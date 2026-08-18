#!/usr/bin/env python3
"""Inspect one set of traces on its own - no comparison, no second backend.

Answers two questions per case, before Forge is involved at all:

  1. Does the target neuron actually fire, and on which token?
  2. Which neuron wins the layer, and by how much?

Both matter for whether a case can be graded later. The comparison script only
counts a peak-position or destination decision when the winner beats the
runner-up by a comfortable margin; a case where the top two are nearly tied is
reported as INDETERMINATE, because at that separation the ordering is not
resolvable at float32 precision. This script shows you which cases are in that
state while the fixture is still cheap to change.

Usage:
  python inspect_traces.py --fixture fixture --backend cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import trace_format as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="directory containing fixture.json")
    ap.add_argument("--backend", default="cpu", help="subdirectory of traces/")
    ap.add_argument("--neuron", type=int, default=None, help="override target neuron")
    ap.add_argument("--layer", type=int, default=None, help="override layer")
    ap.add_argument("--out", default=None,
                    help="write a TSV of these results (easier to share than console output)")
    args = ap.parse_args()

    fixdir = Path(args.fixture)
    with open(fixdir / "fixture.json", encoding="utf-8") as fh:
        fixture = json.load(fh)

    targets = fixture.get("targets", [])
    layer = args.layer if args.layer is not None else targets[0]["layer"]
    neuron = args.neuron if args.neuron is not None else targets[0]["neuron"]
    tdir = fixdir / "traces" / args.backend

    print(f"layer {layer}, target neuron {neuron}, backend {args.backend}\n")
    header = (f"{'case':<5} {'role':<11} {'tok':>4} "
              f"{'N':>8} {'peak tok':<14} {'margin':>8}   "
              f"{'winner':>6} {'val':>8} {'signed':>8}  note")
    print(header)
    print("-" * len(header))

    rows, records = [], []
    for case in fixture["cases"]:
        fp = tdir / tf.trace_filename(case["id"], layer)
        if not fp.exists():
            continue
        _, act = tf.read_trace(fp)

        col = act[:, neuron].astype(np.float64)
        pk = int(col.argmax())
        pk_val = float(col[pk])
        pk_margin = float(pk_val - np.partition(col, -2)[-2]) if col.size > 1 else float("inf")
        tok = case.get("token_strs", [""] * (pk + 1))[pk]

        per_neuron_max = act.max(axis=0).astype(np.float64)
        win = int(per_neuron_max.argmax())
        win_val = float(per_neuron_max[win])
        # v11 signed margin: peak(target) minus the highest peak of ANY other
        # layer-5 neuron. Positive means the target is the destination.
        others = np.delete(per_neuron_max, neuron)
        signed_margin = float(per_neuron_max[neuron] - others.max())

        flags = []
        if pk_val < 0.01:
            flags.append("N541 silent")
        if win == neuron:
            flags.append("N541 WINS")
        else:
            flags.append(f"lost to {win}")

        print(f"{case['id']:<5} {str(case.get('role','')):<11} {case['n_tokens']:>4} "
              f"{pk_val:>8.4f} {repr(tok):<14} {pk_margin:>8.4f}   "
              f"{win:>6} {win_val:>8.4f} {signed_margin:>+8.4f}  {', '.join(flags)}")
        rows.append((case, pk_val, win == neuron))
        records.append({
            "case_id": case["id"], "role": case.get("role"),
            "n_tokens": case["n_tokens"], "peak_value": pk_val,
            "peak_position": pk, "peak_token": tok, "peak_margin": pk_margin,
            "winner_neuron": win, "winner_value": win_val,
            "signed_margin": signed_margin, "target_wins": win == neuron,
        })

    fired = [c for c, v, _ in rows if v >= 0.01]
    won = [c for c, v, w in rows if w]
    print(f"\n{len(rows)} cases | N541 fires in {len(fired)} | N541 wins layer {layer} in {len(won)}")
    not_won = [c["id"] for c, v, w in rows if v >= 0.01 and not w]
    if not_won:
        print(f"fires but does NOT win (these test the destination path): {', '.join(not_won)}")
    else:
        print("NOTE: no case where N541 fires but loses the layer. The destination\n"
              "      check will not be testing anything N541's own trace doesn't\n"
              "      already cover. Consider adding one.")

    if args.out:
        _write_tsv(args.out, records, layer, neuron, args.backend)
        print(f"\nwrote {args.out}")


def _write_tsv(path, records, layer, neuron, backend):
    cols = ["case_id", "role", "n_tokens", "peak_value", "peak_position", "peak_token",
            "peak_margin", "winner_neuron", "winner_value", "signed_margin", "target_wins"]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# layer={layer} neuron={neuron} backend={backend}\n")
        fh.write("\t".join(cols) + "\n")
        for r in records:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
