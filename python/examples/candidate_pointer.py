"""Small persisted pointer policy over variable candidate-action sets.

The universal Kindle policy chooses the stable action identity. Environments
with parameterized actions can then expose a variable set of candidate feature
rows; this head scores each current row and returns an index. The head is a
CPU-only two-layer MLP trained with groupwise softmax cross-entropy, so adding
more candidates does not widen every native or Atari policy graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class CandidatePointer:
    """MLP scorer conditioned on a context token and fixed task identity."""

    def __init__(
        self,
        task_ids: Sequence[str],
        context_dim: int = 64,
        candidate_dim: int = 12,
        hidden_dim: int = 128,
        hidden_dim_2: int = 64,
        seed: int = 42,
    ) -> None:
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be non-empty and unique")
        if min(context_dim, candidate_dim, hidden_dim, hidden_dim_2) < 1:
            raise ValueError("pointer dimensions must be positive")
        self.task_ids = list(task_ids)
        self.task_index = {task_id: i for i, task_id in enumerate(task_ids)}
        self.context_dim = int(context_dim)
        self.candidate_dim = int(candidate_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_dim_2 = int(hidden_dim_2)
        input_dim = context_dim + len(task_ids) + candidate_dim
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(
            0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim),
        ).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(
            0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim_2),
        ).astype(np.float32)
        self.b2 = np.zeros(hidden_dim_2, dtype=np.float32)
        self.w3 = rng.normal(
            0.0, np.sqrt(1.0 / hidden_dim_2), (hidden_dim_2, 1),
        ).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    def _inputs(
        self,
        context: Sequence[float],
        task_id: str,
        candidates: Sequence[Sequence[float]],
    ) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        candidate_array = np.asarray(candidates, dtype=np.float32)
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape {context.shape} != ({self.context_dim},)"
            )
        if task_id not in self.task_index:
            raise ValueError(f"pointer has no task identity for {task_id!r}")
        if candidate_array.ndim != 2 or candidate_array.shape[1] != self.candidate_dim:
            raise ValueError(
                f"candidate shape {candidate_array.shape} must be "
                f"[N, {self.candidate_dim}]"
            )
        if not len(candidate_array):
            raise ValueError("candidate set must not be empty")
        if not np.isfinite(context).all() or not np.isfinite(candidate_array).all():
            raise ValueError("pointer inputs must be finite")
        task = np.zeros(len(self.task_ids), dtype=np.float32)
        task[self.task_index[task_id]] = 1.0
        return np.concatenate([
            np.broadcast_to(context, (len(candidate_array), self.context_dim)),
            np.broadcast_to(task, (len(candidate_array), len(task))),
            candidate_array,
        ], axis=1)

    def scores(
        self,
        context: Sequence[float],
        task_id: str,
        candidates: Sequence[Sequence[float]],
    ) -> np.ndarray:
        inputs = self._inputs(context, task_id, candidates)
        hidden = np.maximum(inputs @ self.w1 + self.b1, 0.0)
        hidden_2 = np.maximum(hidden @ self.w2 + self.b2, 0.0)
        return (hidden_2 @ self.w3 + self.b3)[:, 0]

    def choose(
        self,
        context: Sequence[float],
        task_id: str,
        candidates: Sequence[Sequence[float]],
    ) -> int:
        return int(np.argmax(self.scores(context, task_id, candidates)))

    def accuracy(self, samples: Sequence[dict]) -> float:
        correct = sum(
            self.choose(
                sample["observation"],
                sample["game_id"],
                sample["object_candidate_features"],
            ) == int(sample["object_index"])
            for sample in samples
        )
        return correct / max(1, len(samples))

    def fit(
        self,
        samples: Sequence[dict],
        epochs: int = 30_000,
        learning_rate: float = 5e-4,
    ) -> dict:
        """Fit positive candidate indices with equal total weight per task.

        The final third uses a ten-times lower rate with fresh Adam moments.
        The coarse phase learns the broad candidate rule; the fine phase
        resolves close object choices without oscillating across boundaries.
        """
        if not samples:
            raise ValueError("pointer training requires at least one sample")
        if epochs < 0:
            raise ValueError("pointer epochs must be non-negative")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("pointer learning rate must be finite and positive")
        if epochs == 0:
            return {
                "epochs": 0,
                "accuracy": self.accuracy(samples),
                "loss": 0.0,
                "last_loss": 0.0,
            }

        inputs = []
        groups = []
        targets = []
        task_counts: dict[str, int] = {}
        input_offset = 0
        for sample in samples:
            rows = self._inputs(
                sample["observation"],
                sample["game_id"],
                sample["object_candidate_features"],
            )
            target = int(sample["object_index"])
            if not 0 <= target < len(rows):
                raise ValueError(
                    f"object_index {target} outside {len(rows)} candidates"
                )
            begin = input_offset
            inputs.append(rows)
            input_offset += len(rows)
            groups.append((begin, input_offset))
            targets.append(target)
            task_id = str(sample["game_id"])
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        x = np.concatenate(inputs, axis=0)
        targets = np.asarray(targets, dtype=np.int64)
        state_weights = np.asarray([
            len(samples) / (len(task_counts) * task_counts[str(sample["game_id"])])
            for sample in samples
        ], dtype=np.float32)
        weight_sum = float(state_weights.sum())

        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first_moments = [np.zeros_like(param) for param in params]
        second_moments = [np.zeros_like(param) for param in params]
        fine_start = max(1, int(epochs * 2 / 3) + 1)
        best_correct = -1
        best_loss = float("inf")
        best_params = [param.copy() for param in params]
        last_loss = float("inf")
        epochs_run = 0

        for epoch in range(1, epochs + 1):
            epochs_run = epoch
            if epoch == fine_start:
                first_moments = [np.zeros_like(param) for param in params]
                second_moments = [np.zeros_like(param) for param in params]
            adam_step = epoch if epoch < fine_start else epoch - fine_start + 1
            lr = learning_rate if epoch < fine_start else learning_rate * 0.1

            z1 = x @ self.w1 + self.b1
            a1 = np.maximum(z1, 0.0)
            z2 = a1 @ self.w2 + self.b2
            a2 = np.maximum(z2, 0.0)
            scores = (a2 @ self.w3 + self.b3)[:, 0]
            score_gradient = np.zeros_like(scores)
            loss = 0.0
            correct = 0
            for row, ((begin, end), target) in enumerate(zip(groups, targets)):
                logits = scores[begin:end]
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                correct += int(int(np.argmax(probabilities)) == target)
                loss -= state_weights[row] * np.log(
                    max(float(probabilities[target]), 1e-30)
                )
                probabilities[target] -= 1.0
                score_gradient[begin:end] = state_weights[row] * probabilities
            last_loss = loss / weight_sum
            if correct > best_correct or (
                correct == best_correct and last_loss < best_loss
            ):
                best_correct = correct
                best_loss = last_loss
                best_params = [param.copy() for param in params]
            if correct == len(samples):
                break

            score_gradient /= weight_sum
            grad_w3 = a2.T @ score_gradient[:, None]
            grad_b3 = np.asarray([score_gradient.sum()], dtype=np.float32)
            grad_z2 = (score_gradient[:, None] @ self.w3.T) * (z2 > 0.0)
            grad_w2 = a1.T @ grad_z2
            grad_b2 = grad_z2.sum(axis=0)
            grad_z1 = (grad_z2 @ self.w2.T) * (z1 > 0.0)
            grad_w1 = x.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0)
            gradients = [
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3,
            ]
            correction1 = 1.0 - 0.9 ** adam_step
            correction2 = 1.0 - 0.999 ** adam_step
            for index, (param, gradient) in enumerate(zip(params, gradients)):
                first_moments[index] *= 0.9
                first_moments[index] += 0.1 * gradient
                second_moments[index] *= 0.999
                second_moments[index] += 0.001 * gradient * gradient
                param -= lr * (first_moments[index] / correction1) / (
                    np.sqrt(second_moments[index] / correction2) + 1e-8
                )

        for destination, source in zip(params, best_params):
            destination[...] = source
        return {
            "epochs": epochs_run,
            "accuracy": best_correct / len(samples),
            "loss": best_loss,
            "last_loss": last_loss,
        }

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            format=np.asarray("kindle-candidate-pointer-v1"),
            task_ids=np.asarray(self.task_ids),
            context_dim=np.asarray(self.context_dim, dtype=np.int64),
            candidate_dim=np.asarray(self.candidate_dim, dtype=np.int64),
            hidden_dim=np.asarray(self.hidden_dim, dtype=np.int64),
            hidden_dim_2=np.asarray(self.hidden_dim_2, dtype=np.int64),
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
        )

    @classmethod
    def load(cls, path: str | Path) -> CandidatePointer:
        with np.load(Path(path), allow_pickle=False) as payload:
            if str(payload["format"]) != "kindle-candidate-pointer-v1":
                raise ValueError("unsupported candidate-pointer checkpoint")
            pointer = cls(
                [str(value) for value in payload["task_ids"]],
                context_dim=int(payload["context_dim"]),
                candidate_dim=int(payload["candidate_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                hidden_dim_2=int(payload["hidden_dim_2"]),
            )
            for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
                destination = getattr(pointer, name)
                source = np.asarray(payload[name], dtype=np.float32)
                if source.shape != destination.shape:
                    raise ValueError(
                        f"candidate-pointer {name} shape {source.shape} "
                        f"!= {destination.shape}"
                    )
                destination[...] = source
        return pointer
