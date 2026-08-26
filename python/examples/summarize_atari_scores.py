#!/usr/bin/env python3
"""Aggregate complete Kindle Atari JSONL curves using the D3 score window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kindle._atari_scores import (
    AtariScoreError,
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
            "may be repeated for multiple seeds"
        ),
    )
    parser.add_argument("--atari-protocol", default="published")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.logs) == bool(args.upstream_d3_logdir):
        parser.error(
            "provide either Kindle JSONL logs or --upstream-d3-logdir, "
            "but not both"
        )

    try:
        segments = (
            load_upstream_d3_segments(args.upstream_d3_logdir)
            if args.upstream_d3_logdir
            else load_segments(args.logs)
        )
        summary = summarize_scores(
            segments,
            expected_protocol=args.atari_protocol,
            expected_mode=args.mode,
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
