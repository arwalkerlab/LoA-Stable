#!/usr/bin/env python3
"""
evaluate_generated_dataset.py   –   rev F (2025-07-02)

Improvements vs rev E
---------------------
1. Adds InChIKey-connectivity fallback (chromophore *and* solvent) for
   structural equivalence within the same DOI.
2. Writes a third log  invalid_smiles.txt  for rows whose generated
   chromophore or solvent SMILES cannot be parsed by RDKit.
3. Leaves coverage maths, accuracy stats, and CLI unchanged.

Output directory:
    final_results.txt
    no_match_entries.txt
    multi_match_entries.txt
    invalid_smiles.txt
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except ImportError as e:
    raise SystemExit("RDKit is required.  Install via  pip install rdkit-pypi") from e

# ───────────────────────── helpers ────────────────────────── #
DOI_RX = re.compile(r"10\.\S+")


def canonicalise_paper_code(code: str) -> str | float:
    """Convert '1021.acs.jpcb.5b09905' → '10.1021/acs.jpcb.5b09905'."""
    if pd.isna(code) or code.strip() == "":
        return np.nan
    first_dot = code.find(".")
    return f"10.{code}" if first_dot == -1 else f"10.{code[:first_dot]}/{code[first_dot+1:]}"


def extract_doi(text) -> str | float:
    if pd.isna(text):
        return np.nan
    m = DOI_RX.search(str(text))
    return m.group(0) if m else np.nan


def pct_diff(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return abs(a - b) / abs(b) * 100.0 if b != 0 else np.nan


def safe_mean(series: pd.Series) -> float:
    return series.dropna().mean()


# ---------- RDKit helpers with caching ---------- #
_canon_cache: dict[str, str | None] = {}


def inchikey_connectivity(smi: str) -> str | None:
    """
    Return the 14-char connectivity block (first segment) of the InChIKey
    for the given SMILES, or None if RDKit fails.
    """
    if smi in _canon_cache:
        return _canon_cache[smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _canon_cache[smi] = None
        return None
    ik = Chem.MolToInchiKey(mol)  # e.g. 'KHPHRYYPVSCNMM-UHFFFAOYSA-N'
    conn = ik.split("-")[0]       # 'KHPHRYYPVSCNMM'
    _canon_cache[smi] = conn
    return conn


# ───────────────────────── main evaluation ───────────────────────── #
def evaluate(original_csv: Path, generated_csv: Path, out_dir: Path = Path(".")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # output files
    results_path    = out_dir / "final_results.txt"
    no_match_log    = out_dir / "no_match_entries.txt"
    multi_match_log = out_dir / "multi_match_entries.txt"
    invalid_log     = out_dir / "invalid_smiles.txt"

    # 1 ─── read & normalise ────────────────────────────────────────── #
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

    # pre-compute InChIKey connectivity for the reference set
    ref["inchi_chr"]  = ref["smiles_ref"].astype(str).apply(inchikey_connectivity)
    ref["inchi_solv"] = ref["Solvent"].astype(str).apply(inchikey_connectivity)

    # 2 ─── coverage (row-level) ──────────────────────────────────── #
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

    # 3 ─── numeric prep ──────────────────────────────────────────── #
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

    # log lines
    hdr = "idx\tSMILES\tSolvent\tGen_DOI\tReason\t#ref_matches\n"
    no_lines, multi_lines, invalid_lines = [], [], []

    # quick lowercase solvent in reference for exact comparison
    ref["solv_lc"] = ref["Solvent"].astype(str).str.strip().str.lower()

    # 4 ─── iterate over generated rows ───────────────────────────── #
    for idx, row in gen.iterrows():
        smi_gen_raw = str(row.get("smiles_gen", "")).strip()
        if smi_gen_raw.lower() in {"failed", "nan", ""}:
            continue

        solv_gen_raw = str(row.get("solvent", "")).strip()
        gen_doi = row.get("doi", np.nan)

        # ----- exact string match (SMILES + optional solvent) -----
        mask = (ref["smiles_ref"] == smi_gen_raw)
        if solv_gen_raw:
            mask &= ref["solv_lc"] == solv_gen_raw.lower()
        candidates = ref[mask]

        # ----- DOI filter if ambiguous (or none) -----
        if len(candidates) != 1 and pd.notna(gen_doi):
            candidates = candidates[candidates["doi"] == gen_doi] if len(candidates) else \
                         ref[(ref["doi"] == gen_doi) & (ref["smiles_ref"] == smi_gen_raw)]

        # ----- InChIKey fallback within DOI -----
        if len(candidates) != 1 and pd.notna(gen_doi):
            # first compute connectivity keys for generated row
            ik_chr_gen  = inchikey_connectivity(smi_gen_raw)
            ik_solv_gen = inchikey_connectivity(solv_gen_raw) if solv_gen_raw else None

            if ik_chr_gen is None:         # invalid SMILES – log and skip
                invalid_lines.append(
                    f"{idx}\t{smi_gen_raw}\t{solv_gen_raw}\t{gen_doi or ''}\tinvalid_smiles\t0\n")
                continue

            ref_same_doi = ref[ref["doi"] == gen_doi]
            cand = ref_same_doi[ref_same_doi["inchi_chr"] == ik_chr_gen]

            if len(cand) > 1 and ik_solv_gen is not None:
                cand = cand[cand["inchi_solv"] == ik_solv_gen]

            candidates = cand

        # ----- decision -----
        if len(candidates) == 0:
            no_lines.append(
                f"{idx}\t{smi_gen_raw}\t{solv_gen_raw}\t{gen_doi or ''}\tno_match\t0\n")
            continue
        if len(candidates) > 1:
            multi_lines.append(
                f"{idx}\t{smi_gen_raw}\t{solv_gen_raw}\t{gen_doi or ''}\tmulti\t{len(candidates)}\n")
            continue

        # unique match
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

    # 5 ─── aggregate accuracy ───────────────────────────────────── #
    col_mean_err = {c: safe_mean(pd.Series(v)) for c, v in diff_dict.items()}
    col_acc      = {c: 100 - e if not pd.isna(e) else np.nan
                    for c, e in col_mean_err.items()}
    overall_acc  = safe_mean(pd.Series(col_acc.values()))

    # 6 ─── write logs & summary ─────────────────────────────────── #
    no_match_log.write_text(hdr + "".join(no_lines),     encoding="utf-8")
    multi_match_log.write_text(hdr + "".join(multi_lines), encoding="utf-8")
    invalid_log.write_text(hdr + "".join(invalid_lines), encoding="utf-8")

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
        f"No-match entries    → {no_match_log.name}",
        f"Multi-match entries  → {multi_match_log.name}",
        f"Invalid SMILES rows → {invalid_log.name}",
    ]

    results_path.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


# ─────────────────────────── CLI ──────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate a machine-generated photophysical dataset "
                    "against a reference set with SMILES, solvent, DOI, "
                    "and InChIKey-based structural fallback.")
    ap.add_argument("original_csv", help="Reference CSV")
    ap.add_argument("generated_csv", help="Generated CSV to evaluate")
    ap.add_argument("-o", "--outdir", default="evaluation_results",
                    help="Directory for output logs")
    args = ap.parse_args()
    evaluate(Path(args.original_csv), Path(args.generated_csv), Path(args.outdir))

