#!/usr/bin/env python3
"""Aggregate complete Kindle Atari JSONL curves using the D3 score window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kindle._atari_scores import (
    AtariScoreError,
    compare_runtime_scores,
    load_segments,
    load_upstream_d3_segments,
    summarize_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", type=Path)
    parser.add_argument(
        "--upstream-d3-logdir",
        action="append",
        default=[],
        type=Path,
        help=(
            "score a completed provenance-pinned upstream D3 log directory; "
            "may be repeated for multiple seeds and combined with Kindle logs "
            "for a random-floor-adjusted comparison"
        ),
    )
    parser.add_argument("--atari-protocol", default="published")
    parser.add_argument("--mode", default="train")
    parser.add_argument(
        "--minimum-kindle-seeds",
        type=int,
        default=1,
        help="require this many independent seeds in the Kindle JSONL logs",
    )
    parser.add_argument(
        "--minimum-upstream-seeds",
        type=int,
        default=1,
        help="require this many independent seeds in the upstream D3 logs",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.logs and not args.upstream_d3_logdir:
        parser.error("provide Kindle JSONL logs and/or --upstream-d3-logdir")

    try:
        summaries = []
        if args.logs:
            summaries.append(
                summarize_scores(
                    load_segments(args.logs),
                    expected_protocol=args.atari_protocol,
                    expected_mode=args.mode,
                    minimum_seeds=args.minimum_kindle_seeds,
                )
            )
        if args.upstream_d3_logdir:
            summaries.append(
                summarize_scores(
                    load_upstream_d3_segments(args.upstream_d3_logdir),
                    expected_protocol=args.atari_protocol,
                    expected_mode=args.mode,
                    minimum_seeds=args.minimum_upstream_seeds,
                )
            )
        summary = (
            compare_runtime_scores(*summaries) if len(summaries) == 2 else summaries[0]
        )
    except (AtariScoreError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
