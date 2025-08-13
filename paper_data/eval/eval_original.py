#!/usr/bin/env python3
"""
evaluate_generated_dataset.py   –   rev D (2025-07-02)

Adds a third disambiguation layer:
    SMILES  →  Solvent  →  DOI
Logs are split into:
    no_match_entries.txt        (#ref_matches == 0)
    multi_match_entries.txt     (#ref_matches  > 1)
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

# ────────────────────────── helpers ───────────────────────────────────────── #
DOI_RX = re.compile(r"10\.\S+")


def canonicalise_paper_code(code: str) -> str | float:
    """Convert `1016.j.jlumin.2017.03.042` → `10.1016/j.jlumin.2017.03.042`."""
    if pd.isna(code) or code.strip() == "":
        return np.nan
    first_dot = code.find(".")
    return f"10.{code}" if first_dot == -1 else f"10.{code[:first_dot]}/{code[first_dot+1:]}"


def extract_doi(text) -> str | float:
    """Pull pure DOI (10.xxxx/...) from the Reference cell."""
    if pd.isna(text):
        return np.nan
    m = DOI_RX.search(str(text))
    return m.group(0) if m else np.nan


def pct_diff(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    denom = abs(b)
    return np.nan if denom == 0 else abs(a - b) / denom * 100.0


def safe_mean(series: pd.Series) -> float:
    return series.dropna().mean()


# ────────────────────────── main routine ──────────────────────────────────── #
def evaluate(original_csv: Path, generated_csv: Path, out_dir: Path = Path(".")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # output files
    results_path   = out_dir / "final_results.txt"
    no_match_log   = out_dir / "no_match_entries.txt"
    multi_match_log = out_dir / "multi_match_entries.txt"

    # 1 ─── read & normalise ──────────────────────────────────────────────── #
    na_vals = ["null", "NULL", "", "NaN", "nan"]

    ref = pd.read_csv(original_csv, sep=None, engine="python",
                      na_values=na_vals, keep_default_na=True)
    gen = pd.read_csv(generated_csv, sep=None, engine="python",
                      na_values=na_vals, keep_default_na=True)

    ref = ref.rename(columns={"Chromophore": "smiles_ref",
                              "Reference": "ref_reference"})
    gen = gen.rename(columns={"molecule_name": "smiles_gen",
                              "paper": "paper_code"})

    ref["doi"] = ref["ref_reference"].apply(extract_doi)
    gen["doi"] = gen["paper_code"].apply(canonicalise_paper_code)

    # 2 ─── coverage metrics (unchanged) ─────────────────────────────────── #
    gen_valid = gen[gen["smiles_gen"].astype(str).str.lower().ne("failed")]
    gen_counts = gen_valid.groupby("doi").size()
    ref_counts = ref.groupby("doi").size()

    coverage = (pd.DataFrame({"gen_n": gen_counts, "ref_n": ref_counts})
                .fillna(0).astype(int).reset_index().rename(columns={"index": "doi"}))
    coverage["ratio"] = np.where(coverage["ref_n"] == 0,
                                 np.nan,
                                 coverage["gen_n"] / coverage["ref_n"])
    ratio_all      = safe_mean(coverage["ratio"])
    ratio_nonempty = safe_mean(coverage.loc[coverage["gen_n"] > 0, "ratio"])

    # 3 ─── accuracy prep ────────────────────────────────────────────────── #
    numeric_cols = [
        "Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)",
        "Quantum yield", "log(e/mol-1 dm3 cm-1)",
        "abs FWHM (cm-1)", "emi FWHM (cm-1)",
        "abs FWHM (nm)", "emi FWHM (nm)",
        "Molecular weight (g mol-1)"
    ]
    for col in numeric_cols:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")
        if col in gen.columns:
            gen[col] = pd.to_numeric(gen[col], errors="coerce")

    diff_dict   = {c: [] for c in numeric_cols}
    missed_vals = {c: 0  for c in numeric_cols}
    extra_vals  = {c: 0  for c in numeric_cols}

    # quick helpers for logs
    header = "idx\tSMILES\tSolvent\tGen_DOI\t#ref_matches\n"
    no_match_lines, multi_match_lines = [], []

    # 4 ─── iterate through generated rows ───────────────────────────────── #
    # Pre-lowercase columns used for comparison to avoid repeated .str.lower()
    ref_smiles = ref["smiles_ref"].astype(str).str.strip().str.lower()
    ref_solvent = ref["Solvent"].astype(str).str.strip().str.lower()

    for idx, row in gen.iterrows():
        smi_raw = str(row.get("smiles_gen", "")).strip()
        if smi_raw.lower() in {"failed", "nan", ""}:
            continue

        smi = smi_raw.lower()
        solv = str(row.get("solvent", "")).strip().lower()
        gen_doi = row.get("doi", np.nan)

        # ----------------  Disambiguation pipeline  ---------------- #
        # (1) SMILES
        cand_mask = ref_smiles == smi
        candidates = ref[cand_mask]

        # No SMILES match at all
        if len(candidates) == 0:
            no_match_lines.append(f"{idx}\t{smi_raw}\t{row.get('solvent','')}\t{gen_doi or ''}\t0\n")
            continue

        # (2) Solvent, if provided
        if solv and len(candidates) > 1:
            cand_mask &= ref_solvent == solv
            cand_solvent = ref[cand_mask]
            if len(cand_solvent) >= 1:
                candidates = cand_solvent

        # (3) DOI, if provided
        if pd.notna(gen_doi) and len(candidates) > 1:
            cand_doi = candidates[candidates["doi"] == gen_doi]
            if len(cand_doi) >= 1:
                candidates = cand_doi

        # final decision
        if len(candidates) == 0:
            no_match_lines.append(f"{idx}\t{smi_raw}\t{row.get('solvent','')}\t{gen_doi or ''}\t0\n")
            continue
        if len(candidates) > 1:
            multi_match_lines.append(
                f"{idx}\t{smi_raw}\t{row.get('solvent','')}\t{gen_doi or ''}\t{len(candidates)}\n"
            )
            continue

        # === unique match found ===
        ref_row = candidates.iloc[0]

        for col in numeric_cols:
            g_val = row.get(col, np.nan)
            r_val = ref_row.get(col, np.nan)

            if pd.isna(g_val) and pd.isna(r_val):
                continue
            if pd.isna(g_val) and not pd.isna(r_val):
                missed_vals[col] += 1
            elif not pd.isna(g_val) and pd.isna(r_val):
                extra_vals[col] += 1
            else:
                diff_dict[col].append(pct_diff(g_val, r_val))

    # 5 ─── aggregate accuracies ────────────────────────────────────────── #
    col_mean_err = {c: safe_mean(pd.Series(v)) for c, v in diff_dict.items()}
    col_acc      = {c: 100 - e if not pd.isna(e) else np.nan
                    for c, e in col_mean_err.items()}
    overall_acc  = safe_mean(pd.Series(col_acc.values()))

    # 6 ─── write logs & summary ───────────────────────────────────────── #
    no_match_log.write_text(header + "".join(no_match_lines), encoding="utf-8")
    multi_match_log.write_text(header + "".join(multi_match_lines), encoding="utf-8")

    summary = [
        "===== COVERAGE =====",
        f"Average ratio (including zero-extractions): {ratio_all:.3f}",
        f"Average ratio (excluding zero-extractions): {ratio_nonempty:.3f}",
        "",
        "===== ACCURACY (per property) =====",
    ]
    for col in numeric_cols:
        summary.append(
            f"{col:35}: acc={col_acc[col]:6.2f}% | mean |%Δ|={col_mean_err[col]:6.2f}% "
            f"| missed={missed_vals[col]:4d} | extra={extra_vals[col]:4d}"
        )

    summary += [
        "",
        f"===== OVERALL ACCURACY =====  {overall_acc:6.2f}%",
        "",
        f"No-match entries   → {no_match_log.name}",
        f"Multi-match entries → {multi_match_log.name}",
    ]

    results_path.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


# ────────────────────────── CLI wrapper ──────────────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate a machine-generated photophysical dataset "
                    "against the human-curated reference with SMILES, solvent "
                    "and DOI disambiguation.")
    ap.add_argument("original_csv", help="Reference CSV file")
    ap.add_argument("generated_csv", help="Generated CSV file to evaluate")
    ap.add_argument("-o", "--outdir", default="evaluation_results",
                    help="Directory to store result logs")
    args = ap.parse_args()
    evaluate(Path(args.original_csv), Path(args.generated_csv), Path(args.outdir))

