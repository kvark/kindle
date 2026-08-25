#!/usr/bin/env python3
"""Aggregate complete Kindle Atari JSONL curves using the D3 score window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kindle._atari_scores import AtariScoreError, load_segments, summarize_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--atari-protocol", default="published")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        summary = summarize_scores(
            load_segments(args.logs),
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
