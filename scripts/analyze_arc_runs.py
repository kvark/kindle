#!/usr/bin/env python3
"""Parse arc_agi3_grpo logs and compare runs side-by-side."""
import re
import sys
from pathlib import Path

LOG_PAT = re.compile(
    r"macro=\s*(\d+)\s+micro_total=\s*(\d+)\s+eps=\s*(\d+)\s+evt=\s*(\d+)\s+\(([^)]+)\)"
)
LEVEL_PAT = re.compile(r"L(\d+):(\d+)")


def parse_log(path: Path) -> dict:
    out = {
        "path": str(path),
        "final_macro": 0,
        "final_evt": 0,
        "final_per_game": {},
        "summary": [],  # list of dicts per game
        "total_evt_by_level": {},
        "throughput_env_s": None,
        "elapsed_s": None,
    }
    if not path.exists():
        return out
    text = path.read_text()
    last_macro_evt = None
    for m in LOG_PAT.finditer(text):
        macro = int(m.group(1))
        evt = int(m.group(4))
        per_game = m.group(5)
        out["final_macro"] = macro
        out["final_evt"] = evt
        per_game_now = {}
        if per_game not in ("(none)", "none", "(none"):
            for pair in per_game.split(","):
                pair = pair.strip()
                if ":" in pair:
                    g, n = pair.split(":")
                    try:
                        per_game_now[g] = int(n)
                    except ValueError:
                        pass
        out["final_per_game"] = per_game_now
    # Per-game summary block at end
    in_summary = False
    for line in text.splitlines():
        if "Per-game summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.startswith("games with"):
            break
        # game     eps  evt max_lvl  win  L1:n L2:m
        parts = line.split()
        if len(parts) >= 5 and parts[0].isalnum() and len(parts[0]) == 4:
            try:
                g = parts[0]
                eps = int(parts[1])
                evt = int(parts[2])
                max_lvl = int(parts[3])
                wlvl = int(parts[4])
                # Per-level events from L*:N tokens
                per_lvl = {}
                for tok in parts[5:]:
                    m = LEVEL_PAT.match(tok)
                    if m:
                        per_lvl[int(m.group(1))] = int(m.group(2))
                out["summary"].append({
                    "game": g, "eps": eps, "evt": evt,
                    "max_lvl": max_lvl, "win_levels": wlvl,
                    "per_lvl": per_lvl,
                })
                for lvl, n in per_lvl.items():
                    out["total_evt_by_level"][lvl] = out["total_evt_by_level"].get(lvl, 0) + n
            except (ValueError, IndexError):
                pass
    # Throughput + elapsed
    m = re.search(r"throughput:\s*(\d+)\s*env/s\s*\(([\d.]+)s\s*total\)", text)
    if m:
        out["throughput_env_s"] = int(m.group(1))
        out["elapsed_s"] = float(m.group(2))
    return out


def fmt_per_lvl(d):
    if not d:
        return "-"
    return " ".join(f"L{lvl}:{d[lvl]}" for lvl in sorted(d))


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py log1 [log2 ...]")
        return 1
    runs = [parse_log(Path(arg)) for arg in sys.argv[1:]]
    # Final-state header
    print(f"{'run':<22} {'macros':>7} {'evt':>5}  {'L0':>5} {'L1':>5} {'L2':>5} {'L3':>5}  throughput  elapsed")
    for r in runs:
        name = Path(r["path"]).stem
        lvl = r["total_evt_by_level"]
        l0 = lvl.get(0, 0)
        l1 = lvl.get(1, 0)
        l2 = lvl.get(2, 0)
        l3 = lvl.get(3, 0)
        thr = r["throughput_env_s"] or "?"
        el = r["elapsed_s"] or 0
        print(f"{name:<22} {r['final_macro']:>7} {r['final_evt']:>5}  "
              f"{l0:>5} {l1:>5} {l2:>5} {l3:>5}  {thr:>6}env/s {el:>6.0f}s")
    # Per-game L1/L2/L3 breakdown
    print()
    print("--- per-game ---")
    games = sorted({s["game"] for r in runs for s in r["summary"]})
    print(f"{'game':<6}", end="")
    for r in runs:
        print(f" {Path(r['path']).stem[:14]:<16}", end="")
    print()
    for g in games:
        print(f"{g:<6}", end="")
        for r in runs:
            s = next((s for s in r["summary"] if s["game"] == g), None)
            if s:
                lvls = fmt_per_lvl(s["per_lvl"])
                print(f" e{s['evt']:>4}/maxL{s['max_lvl']} {lvls:<8}", end="")
            else:
                print(f" {'-':<16}", end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
