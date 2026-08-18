#!/usr/bin/env python3
"""Part 3 of 3: compare Forge traces against the Hugging Face reference.

  python compare_traces.py --fixture fixture \
      --ref cpu --alt-ref cuda \
      --test fixture/traces/forge \
      --out report_v1

Tolerance policy
----------------
If two reference backends are present, per-case tolerance is derived from the
observed cpu-vs-cuda disagreement rather than from a number someone picked.
That is the point of exporting both: it establishes what "same implementation,
different backend" already costs, so Forge's deviation can be judged against an
empirical envelope. With only one reference backend the script falls back to
the config floors and labels every verdict PROVISIONAL.

Two distinct questions are reported separately, per the handoff:
  * trace agreement for the target neuron (does N541 do the same thing?)
  * destination agreement for the whole layer (max over positions per neuron,
    then argmax over neurons - does the same neuron win?)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import trace_format as tf


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency (average ranks for ties)."""
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average tied ranks
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def pearson(a, b) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def top2_margin(v: np.ndarray):
    if v.size < 2:
        return float("inf"), int(np.argmax(v)) if v.size else -1
    idx = int(np.argmax(v))
    part = np.partition(v, -2)
    return float(part[-1] - part[-2]), idx


def alignment_probe(ref: np.ndarray, test: np.ndarray):
    """Detect the classic off-by-one from BOS handling or a dropped position."""
    out = {"shape_ref": list(ref.shape), "shape_test": list(test.shape), "flag": None}
    if ref.shape[1] != test.shape[1]:
        out["flag"] = "WIDTH_MISMATCH"
        return out
    if ref.shape[0] != test.shape[0]:
        diff = test.shape[0] - ref.shape[0]
        out["flag"] = "LENGTH_MISMATCH"
        out["length_delta"] = int(diff)
        # try trimming the leading position from whichever side is longer
        if diff == 1:
            err = float(np.abs(ref - test[1:]).max())
            out["error_if_test_leading_position_dropped"] = err
            out["hint"] = ("Forge emitted one extra leading position. Most likely "
                           "Forge added BOS itself while the fixture ids already "
                           "contain it, or vice versa.")
        elif diff == -1:
            err = float(np.abs(ref[1:] - test).max())
            out["error_if_ref_leading_position_dropped"] = err
            out["hint"] = ("Forge emitted one fewer position. Most likely Forge "
                           "consumed its own tokenizer output instead of the "
                           "fixture ids, or dropped BOS.")
        return out
    base = float(np.abs(ref - test).max())
    shifted = float(np.abs(ref[1:] - test[:-1]).max()) if ref.shape[0] > 1 else float("inf")
    out["error_shift_0"] = base
    out["error_shift_1"] = shifted
    if shifted * 10 < base:
        out["flag"] = "SHIFT_SUSPECTED"
        out["hint"] = ("Agreement improves ~10x under a one-position shift: the "
                       "sequences are misaligned, not numerically different.")
    return out


def compare_case(case, layer, ref, test, alt, cfg, targets):
    tolcfg = cfg["tolerance"]
    crit = cfg["criteria"]
    res = {"case_id": case["id"], "role": case.get("role"), "layer": layer,
           "n_tokens": case["n_tokens"], "prepend_bos": case.get("prepend_bos")}

    align = alignment_probe(ref, test)
    res["alignment"] = align
    if align["flag"] in ("WIDTH_MISMATCH", "LENGTH_MISMATCH"):
        res["verdict"] = "ABORT_SHAPE"
        return res

    # --- tolerance -------------------------------------------------------
    if alt is not None and alt.shape == ref.shape:
        env = np.abs(ref.astype(np.float64) - alt.astype(np.float64))
        env_max = float(env.max())
        env_p999 = float(np.percentile(env, 99.9))
        tol = max(tolcfg["envelope_multiplier"] * env_max, tolcfg["abs_floor"])
        tol_source = "backend_envelope"
    else:
        env_max = env_p999 = None
        tol = tolcfg["abs_floor"]
        tol_source = "abs_floor_only"
    res["tolerance"] = {"value": tol, "source": tol_source,
                        "backend_envelope_max": env_max,
                        "backend_envelope_p99_9": env_p999}

    # --- whole-layer numeric error ---------------------------------------
    d = np.abs(ref.astype(np.float64) - test.astype(np.float64))
    res["layer_error"] = {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "p99_9_abs": float(np.percentile(d, 99.9)),
        "max_abs_over_tolerance": float(d.max() / tol) if tol > 0 else float("inf"),
    }

    exceed = np.argwhere(d > tol)
    if exceed.size:
        t0, n0 = int(exceed[0][0]), int(exceed[0][1])
        res["first_divergence"] = {
            "position": t0, "neuron": n0,
            "token_id": case["token_ids"][t0],
            "token_str": case.get("token_strs", [None] * (t0 + 1))[t0],
            "ref": float(ref[t0, n0]), "test": float(test[t0, n0]),
            "delta": float(test[t0, n0] - ref[t0, n0]),
            "n_positions_clean_before": t0,
        }
    else:
        res["first_divergence"] = None

    # --- destination decision (max over positions, then argmax over neurons)
    ref_peak_per_neuron = ref.max(axis=0)
    test_peak_per_neuron = test.max(axis=0)
    margin, ref_dest = top2_margin(ref_peak_per_neuron)
    test_dest = int(np.argmax(test_peak_per_neuron))
    margin_ratio = margin / tol if tol > 0 else float("inf")
    testable = margin_ratio >= crit["min_margin_ratio"]
    k = int(crit["top_k"])
    ref_topk = set(np.argsort(ref_peak_per_neuron)[-k:].tolist())
    test_topk = set(np.argsort(test_peak_per_neuron)[-k:].tolist())
    top50 = np.argsort(ref_peak_per_neuron)[-50:]
    res["destination"] = {
        "ref_neuron": int(ref_dest),
        "test_neuron": test_dest,
        "match": ref_dest == test_dest,
        "winner_margin": margin,
        "margin_over_tolerance": margin_ratio,
        "testable": bool(testable),
        f"top{k}_overlap": len(ref_topk & test_topk),
        "spearman_top50": spearman(ref_peak_per_neuron[top50], test_peak_per_neuron[top50]),
    }

    # --- per-target neuron traces ----------------------------------------
    res["neurons"] = {}
    for tgt in targets:
        if tgt["layer"] != layer:
            continue
        n = int(tgt["neuron"])
        rv, tv = ref[:, n].astype(np.float64), test[:, n].astype(np.float64)
        dn = np.abs(rv - tv)
        # v11 signed margin: peak(target) - highest peak of any OTHER neuron.
        # Reported so the numbers here line up with the existing probe results.
        ref_others = np.delete(ref_peak_per_neuron, n)
        test_others = np.delete(test_peak_per_neuron, n)
        signed_ref = float(ref_peak_per_neuron[n] - ref_others.max())
        signed_test = float(test_peak_per_neuron[n] - test_others.max())
        pmargin, rpeak = top2_margin(rv)
        tpeak = int(np.argmax(tv))
        first = int(np.argmax(dn > tol)) if bool((dn > tol).any()) else None
        res["neurons"][tgt["name"]] = {
            "neuron": n,
            "max_abs_err": float(dn.max()),
            "mean_abs_err": float(dn.mean()),
            "max_rel_err": float((dn / np.maximum(np.abs(rv), 1e-6)).max()),
            "max_abs_err_over_tolerance": float(dn.max() / tol) if tol > 0 else float("inf"),
            "pearson": pearson(rv, tv),
            "spearman_positions": spearman(rv, tv),
            "peak_position_ref": int(rpeak),
            "peak_position_test": tpeak,
            "peak_match": int(rpeak) == tpeak,
            "peak_value_ref": float(rv[rpeak]),
            "peak_value_test": float(tv[tpeak]),
            "peak_margin_ref": pmargin,
            "peak_testable": bool(pmargin / tol >= crit["min_margin_ratio"]) if tol > 0 else True,
            "first_divergent_position": first,
            "signed_margin_ref": signed_ref,
            "signed_margin_test": signed_test,
            "signed_margin_delta": signed_test - signed_ref,
            "signed_margin_sign_flip": (signed_ref > 0) != (signed_test > 0),
        }

    # --- verdict ---------------------------------------------------------
    fails, notes = [], []
    if align["flag"] == "SHIFT_SUSPECTED":
        fails.append("alignment_shift")
    dst = res["destination"]
    if crit["destination_must_match"]:
        if dst["testable"]:
            if not dst["match"]:
                fails.append("destination_mismatch")
        else:
            notes.append("destination_indeterminate_low_margin")
    if dst[f"top{k}_overlap"] < crit["min_topk_overlap"]:
        fails.append(f"top{k}_overlap")
    if not (dst["spearman_top50"] >= crit["min_spearman_top50"]):
        fails.append("spearman_top50")
    for name, m in res["neurons"].items():
        if m["signed_margin_sign_flip"]:
            fails.append(f"{name}_signed_margin_sign_flip")
        if crit["peak_position_must_match"] and not m["peak_match"]:
            if m["peak_testable"]:
                fails.append(f"{name}_peak_position")
            else:
                notes.append(f"{name}_peak_indeterminate_low_margin")
    res["failures"] = fails
    res["notes"] = notes
    res["verdict"] = "FAIL" if fails else ("PASS_WITH_NOTES" if notes else "PASS")
    return res


def markdown_table(results, targets):
    tname = targets[0]["name"] if targets else None
    head = ["case", "role", "tok", "tol", "layer max abs err", "x tol",
            f"{tname} max abs err", f"{tname} peak", "dest ref/test", "margin x tol", "verdict"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in results:
        if r["verdict"] == "ABORT_SHAPE":
            lines.append(f"| {r['case_id']} | {r.get('role','')} | {r['n_tokens']} | "
                         f"- | - | - | - | - | - | - | **ABORT_SHAPE** |")
            continue
        m = r["neurons"].get(tname, {})
        d = r["destination"]
        peak = ("ok " + str(m.get("peak_position_ref"))) if m.get("peak_match") \
            else f"{m.get('peak_position_ref')} / {m.get('peak_position_test')}"
        lines.append(
            f"| {r['case_id']} | {r.get('role','')} | {r['n_tokens']} | "
            f"{r['tolerance']['value']:.2e} | {r['layer_error']['max_abs']:.3e} | "
            f"{r['layer_error']['max_abs_over_tolerance']:.2f} | "
            f"{m.get('max_abs_err', float('nan')):.3e} | {peak} | "
            f"{d['ref_neuron']} / {d['test_neuron']}{'' if d['match'] else ' MISMATCH'} | "
            f"{d['margin_over_tolerance']:.1f} | {r['verdict']} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="directory containing fixture.json")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--ref", default="cpu", help="primary reference backend name")
    ap.add_argument("--alt-ref", default=None, help="second reference backend (calibrates tolerance)")
    ap.add_argument("--test", default=None, help="directory of Forge traces (default: <fixture>/traces/forge)")
    ap.add_argument("--out", default=None, help="directory for report.json / report.md")
    ap.add_argument("--only-roles", default=None, help="comma-separated roles to include")
    args = ap.parse_args()

    fixdir = Path(args.fixture)
    cfg = load_json(args.config)
    fixture = load_json(fixdir / "fixture.json")
    ref_dir = fixdir / "traces" / args.ref
    alt_dir = (fixdir / "traces" / args.alt_ref) if args.alt_ref else None
    test_dir = Path(args.test) if args.test else fixdir / "traces" / "forge"
    for d in [ref_dir, test_dir] + ([alt_dir] if alt_dir else []):
        if not d.is_dir():
            print(f"ERROR: missing trace directory {d}", file=sys.stderr)
            raise SystemExit(2)

    targets = fixture.get("targets", cfg["targets"])
    layers = fixture.get("capture_layers", cfg["capture_layers"])
    roles = set(args.only_roles.split(",")) if args.only_roles else None

    results, missing = [], []
    for case in fixture["cases"]:
        if roles and case.get("role") not in roles:
            continue
        for layer in layers:
            fn = tf.trace_filename(case["id"], layer)
            if not (test_dir / fn).exists():
                missing.append(fn)
                continue
            _, ref = tf.read_trace(ref_dir / fn)
            _, test = tf.read_trace(test_dir / fn)
            alt = None
            if alt_dir and (alt_dir / fn).exists():
                _, alt = tf.read_trace(alt_dir / fn)
            results.append(compare_case(case, layer, ref, test, alt, cfg, targets))

    verdicts = {}
    for r in results:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    provisional = any(r.get("tolerance", {}).get("source") == "abs_floor_only" for r in results)

    summary = {
        "compared": len(results),
        "missing_test_traces": missing,
        "verdicts": verdicts,
        "overall": ("FAIL" if (verdicts.get("FAIL") or verdicts.get("ABORT_SHAPE"))
                    else ("PASS" if results else "NO_DATA")),
        "provisional": provisional,
        "reference_backend": args.ref,
        "tolerance_calibration_backend": args.alt_ref,
        "checkpoint": fixture.get("checkpoint"),
        "environments": fixture.get("environments"),
    }

    md = ["# Forge / Hugging Face conformance report", "",
          f"- reference backend: `{args.ref}`"
          + (f", tolerance calibrated against `{args.alt_ref}`" if args.alt_ref
             else ", **no second backend: tolerance is a declared floor, verdicts are PROVISIONAL**"),
          f"- cases compared: {len(results)}",
          f"- verdicts: {verdicts}", ""]
    if missing:
        md += [f"- **missing Forge traces ({len(missing)})**: "
               + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""), ""]
    md += [markdown_table(results, targets), ""]
    fails = [r for r in results if r["verdict"] in ("FAIL", "ABORT_SHAPE")]
    if fails:
        md += ["## Failures", ""]
        for r in fails:
            reasons = r.get("failures") or [r["verdict"].lower()]
            md.append(f"### {r['case_id']} - {', '.join(reasons)}")
            fd = r.get("first_divergence")
            if fd:
                md.append(f"- first divergence at position {fd['position']}, neuron "
                          f"{fd['neuron']} (token {fd['token_id']} {fd['token_str']!r}): "
                          f"ref {fd['ref']:+.6f} vs test {fd['test']:+.6f} "
                          f"(delta {fd['delta']:+.2e}); {fd['n_positions_clean_before']} "
                          f"positions clean before it")
            if r["alignment"].get("hint"):
                md.append(f"- alignment: {r['alignment']['hint']}")
            md.append("")

    # Write artifacts FIRST. If the console cannot render something, the report
    # must still survive; a failed print should never cost you the results.
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "report.json", "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "config": cfg, "cases": results}, fh, indent=2)
        (out / "report.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    if args.out:
        print(f"\nwrote {out/'report.json'} and {out/'report.md'}")

    raise SystemExit(1 if summary["overall"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
