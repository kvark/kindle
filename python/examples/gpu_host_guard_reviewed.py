"""Host guard that accepts kernel fault records a human has reviewed for this boot.

`gpu_host_guard.py` refuses a boot whose kernel journal holds any GPU fault.
This wrapper runs the same guard, unchanged, after removing from its journal
reads the exact records listed in the file named by
`KINDLE_REVIEWED_KERNEL_FAULTS` (same boot, same journal cursor, same message).
Each accepted record is retained as a `reviewed_kernel_fault` event. Any fault
record that is not listed, including every fault after the review, still stops
the guard. No retries, host recovery or driver action.
"""

import json
import os
from pathlib import Path
import sys

import gpu_guard as retained
import gpu_host_guard as base


def load(path):
    review = json.loads(Path(path).read_text())
    if set(review) != {"boot_id", "reviewed_by", "reviewed_at", "decision", "records"}:
        raise ValueError("reviewed-fault file requires boot_id, reviewed_by, reviewed_at, decision and records")
    if review["boot_id"] != retained.BOOT.read_text().strip():
        raise ValueError("the review applies to another boot")
    rows = review["records"]
    if (not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict) or set(row) != {"cursor", "message"}
            or any(not isinstance(value, str) or not value for value in row.values())
            or not retained.FAULT.search(row["message"]) for row in rows)):
        raise ValueError("reviewed faults require exact journal cursors and fault messages")
    return {(row["cursor"], row["message"]) for row in rows}


def install(path):
    reviewed = load(path)
    digest = retained.fingerprint(path)
    parse, check = retained.parse_journal, retained.check_kernel
    accepted = []

    def parse_journal(text, boot, limit):
        records = parse(text, boot, limit)
        kept = []
        for row in records:
            if (row["__CURSOR"], row["MESSAGE"]) in reviewed:
                accepted.append(row)
            else:
                kept.append(row)
        return kept

    def check_kernel(evidence, boot, cursor=None, *args, **kwargs):
        accepted.clear()
        try:
            return check(evidence, boot, cursor, *args, **kwargs)
        finally:
            for row in accepted:
                evidence.event("reviewed_kernel_fault", record=row, review=str(Path(path).resolve()),
                               review_sha256=digest, baseline=cursor is None)

    retained.parse_journal, retained.check_kernel = parse_journal, check_kernel


if __name__ == "__main__":
    install(os.environ["KINDLE_REVIEWED_KERNEL_FAULTS"])
    sys.exit(base.main())
