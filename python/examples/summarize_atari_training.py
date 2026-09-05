#!/usr/bin/env python3
"""Summarize an exact interval-aligned Kindle Atari training window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kindle._atari_training import AtariTrainingError, summarize_training_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--start-step", type=int, help="exclusive absolute environment step")
    parser.add_argument("--end-step", type=int, help="inclusive absolute environment step")
    parser.add_argument("--window-steps", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = summarize_training_window(
            args.log,
            start_step=args.start_step,
            end_step=args.end_step,
            default_window_steps=args.window_steps,
        )
    except (AtariTrainingError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
