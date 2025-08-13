#!/usr/bin/env python3
"""
evaluate_generated_dataset.py   –   rev K  (2025-07-02)

• Fixes FileNotFoundError when saving plots: filenames are now sanitised.
• Rest is identical to rev J.
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── external deps ───────────────────────────────────────────────────────── #
try:
    import pubchempy as pcp
except ImportError as e:
    raise SystemExit("pubchempy not installed.  pip install pubchempy") from e

try:
    from rdkit import Chem
except ImportError as e:
    raise SystemExit("RDKit not installed.  pip install rdkit-pypi") from e

# ── regex & caches ─────────────────────────────────────────────────────── #
DOI_RX = re.compile(r"10\.\S+")
FNAME_RX = re.compile(r"[^A-Za-z0-9_-]+")

cid_cache: dict[str, int | None] = {}
inchi_cache: dict[str, str | None] = {}


def safe_filename(text: str) -> str:
    """Convert arbitrary text into safe filename segment."""
    return FNAME_RX.sub("_", text).strip("_")


# ── cache helpers ──────────────────────────────────────────────────────── #
def load_cid_cache(path: Path) -> None:
    global cid_cache
    if path.is_file():
        try:
            cid_cache = json.loads(path.read_text())
        except Exception:
            cid_cache = {}


def save_cid_cache(path: Path) -> None:
    with path.open("w") as fh:
        json.dump(cid_cache, fh)


# ── chem helpers ───────────────────────────────────────────────────────── #
def pubchem_cid(smiles: str) -> int | None:
    if smiles in cid_cache:
        return cid_cache[smiles]
    try:
        comps = pcp.get_compounds(smiles, "smiles", "identity")
        cid = comps[0].cid if comps else None
    except Exception:
        cid = None
    cid_cache[smiles] = cid
    return cid


def inchi_connectivity(smiles: str) -> str | None:
    if smiles in inchi_cache:
        return inchi_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    conn = None if mol is None else Chem.MolToInchiKey(mol).split("-")[0]
    inchi_cache[smiles] = conn
    return conn


# ── misc helpers ───────────────────────────────────────────────────────── #
def canonicalise_paper_code(code: str) -> str | float:
    if pd.isna(code) or code.strip() == "":
        return np.nan
    dot = code.find(".")
    return f"10.{code}" if dot == -1 else f"10.{code[:dot]}/{code[dot+1:]}"


def extract_doi(text) -> str | float:
    if pd.isna(text):
        return np.nan
    m = DOI_RX.search(str(text))
    return m.group(0) if m else np.nan


def pct_diff(a: float, b: float) -> float:
    return np.nan if pd.isna(a) or pd.isna(b) or b == 0 else abs(a - b) / abs(b) * 100.0


def safe_mean(series: pd.Series) -> float:
    return series.dropna().mean()


# ── plotting utilities ─────────────────────────────────────────────────── #
def make_scatter(df: pd.DataFrame, col: str, outdir: Path) -> None:
    ref = df[f"{col}_ref"].dropna()
    gen = df.loc[ref.index, f"{col}_gen"]
    if ref.empty:
        return
    plt.figure()
    plt.scatter(ref, gen, alpha=0.6, s=20)
    mn, mx = ref.min(), ref.max()
    if mx == mn:
        mn *= 0.95
        mx *= 1.05
    plt.plot([mn, mx], [mn, mx], ls="--")
    plt.xlabel(f"Reference {col}")
    plt.ylabel(f"Generated {col}")
    plt.title(f"Predicted vs. Reference – {col}")
    plt.tight_layout()
    fname = outdir / f"{safe_filename(col)}_scatter.png"
    plt.savefig(fname, dpi=300)
    plt.close()


def make_hist(df: pd.DataFrame, col: str, outdir: Path) -> None:
    err = df[f"{col}_err"].dropna()
    if err.empty:
        return
    plt.figure()
    plt.hist(err, bins=30, alpha=0.7)
    plt.xlabel("|%Δ|")
    plt.ylabel("Count")
    plt.title(f"Error distribution – {col}")
    plt.tight_layout()
    fname = outdir / f"{safe_filename(col)}_hist.png"
    plt.savefig(fname, dpi=300)
    plt.close()


def make_error_boxplot(df: pd.DataFrame, props: list[str], outdir: Path) -> None:
    data, labels = [], []
    for p in props:
        d = df[f"{p}_err"].dropna()
        if len(d):
            data.append(d)
            labels.append(p)
    if not data:
        return
    plt.figure(figsize=(max(6, len(data) * 1.2), 4))
    plt.boxplot(data, labels=labels)
    plt.ylabel("|%Δ|")
    plt.title("Error distribution across properties")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "error_boxplot.png", dpi=300)
    plt.close()


# ── evaluation core ─────────────────────────────────────────────────────── #
def evaluate(original_csv: Path, generated_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cid_cache_path = out_dir / "cid_cache.json"
    load_cid_cache(cid_cache_path)

    # output paths
    res_path   = out_dir / "final_results.txt"
    no_path    = out_dir / "no_match_entries.txt"
    multi_path = out_dir / "multi_match_entries.txt"
    invalid_path = out_dir / "invalid_smiles.txt"

    # 1 ── load CSVs
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

    ref["ik_chr"]  = ref["smiles_ref"].astype(str).apply(inchi_connectivity)
    ref["ik_solv"] = ref["Solvent"].astype(str).apply(inchi_connectivity)
    ref["solv_lc"] = ref["Solvent"].astype(str).str.strip().str.lower()

    # 2 ── numeric props
    props = [
        "Absorption max (nm)", "Emission max (nm)", "Lifetime (ns)",
        "Quantum yield", "log(e/mol-1 dm3 cm-1)",
        "abs FWHM (cm-1)", "emi FWHM (cm-1)",
        "abs FWHM (nm)", "emi FWHM (nm)",
        "Molecular weight (g mol-1)"
    ]
    for col in props:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")
        if col in gen.columns:
            gen[col] = pd.to_numeric(gen[col], errors="coerce")

    comp_records: list[dict[str, float]] = []
    diff, missed, extra = ({p: [] for p in props},
                           {p: 0  for p in props},
                           {p: 0  for p in props})

    hdr = "idx\tSMILES\tSolvent\tGen_DOI\tReason\t#ref_matches\n"
    no_lines, multi_lines, invalid_lines = [], [], []

    # coverage
    gen_ok = gen[gen["smiles_gen"].astype(str).str.lower().ne("failed")]
    cov = (pd.DataFrame({"gen": gen_ok.groupby("doi").size(),
                         "ref": ref.groupby("doi").size()})
           .fillna(0).astype(int).reset_index().rename(columns={"index": "doi"}))
    cov["ratio"] = np.where(cov["ref"] == 0, np.nan, cov["gen"] / cov["ref"])
    ratio_all = safe_mean(cov["ratio"])
    ratio_nonempty = safe_mean(cov.loc[cov["gen"] > 0, "ratio"])

    # 3 ── iterate rows
    for idx, row in gen.iterrows():
        smi_g = str(row.get("smiles_gen", "")).strip()
        if smi_g.lower() in {"failed", "nan", ""}:
            continue
        solv_g = str(row.get("solvent", "")).strip()
        doi_g  = row.get("doi", np.nan)

        # exact
        mask = (ref["smiles_ref"] == smi_g)
        if solv_g:
            mask &= ref["solv_lc"] == solv_g.lower()
        cand = ref[mask]

        # DOI
        if len(cand) != 1 and pd.notna(doi_g):
            cand = cand[cand["doi"] == doi_g] if len(cand) else \
                   ref[(ref["doi"] == doi_g) & (ref["smiles_ref"] == smi_g)]

        # PubChem
        if len(cand) != 1 and pd.notna(doi_g):
            cid_chr_g  = pubchem_cid(smi_g)
            cid_solv_g = pubchem_cid(solv_g) if solv_g else None
            if cid_chr_g is not None:
                cand_doi = ref[ref["doi"] == doi_g]
                cand = cand_doi[cand_doi["smiles_ref"].apply(pubchem_cid) == cid_chr_g]
                if len(cand) > 1 and cid_solv_g is not None:
                    cand = cand[cand["Solvent"].apply(pubchem_cid) == cid_solv_g]

        # RDKit
        if len(cand) != 1 and pd.notna(doi_g):
            ik_chr_g  = inchi_connectivity(smi_g)
            ik_solv_g = inchi_connectivity(solv_g) if solv_g else None
            if ik_chr_g is None and pubchem_cid(smi_g) is None:
                invalid_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tinvalid_smiles\t0\n")
                continue
            cand_doi = ref[ref["doi"] == doi_g]
            cand = cand_doi[cand_doi["ik_chr"] == ik_chr_g]
            if len(cand) > 1 and ik_solv_g is not None:
                cand = cand[cand["ik_solv"] == ik_solv_g]

        # decision
        if len(cand) == 0:
            no_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tno_match\t0\n")
            continue
        if len(cand) > 1:
            multi_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tmulti\t{len(cand)}\n")
            continue

        ref_row = cand.iloc[0]
        rec: dict[str, float] = {}

        for col in props:
            g = row.get(col, np.nan)
            r = ref_row.get(col, np.nan)
            err = pct_diff(g, r)

            rec[f"{col}_gen"] = g
            rec[f"{col}_ref"] = r
            rec[f"{col}_err"] = err

            if pd.isna(g) and pd.isna(r):
                continue
            if pd.isna(g) and not pd.isna(r):
                missed[col] += 1
            elif not pd.isna(g) and pd.isna(r):
                extra[col] += 1
            else:
                diff[col].append(err)

        comp_records.append(rec)

    comp_df = pd.DataFrame(comp_records)

    # 4 ── metrics
    mean_err = {c: safe_mean(pd.Series(v)) for c, v in diff.items()}
    acc      = {c: 100 - e if not pd.isna(e) else np.nan for c, e in mean_err.items()}
    overall  = safe_mean(pd.Series(acc.values()))

    # 5 ── logs
    no_path.write_text(hdr + "".join(no_lines), encoding="utf-8")
    multi_path.write_text(hdr + "".join(multi_lines), encoding="utf-8")
    invalid_path.write_text(hdr + "".join(invalid_lines), encoding="utf-8")

    # 6 ── plots
    for col in props:
        if f"{col}_gen" in comp_df.columns:
            make_scatter(comp_df, col, out_dir)
            make_hist(comp_df, col, out_dir)
    make_error_boxplot(comp_df, props, out_dir)

    # 7 ── report
    report = [
        "===== COVERAGE =====",
        f"Average ratio (including zero-extractions): {ratio_all:.3f}",
        f"Average ratio (excluding zero-extractions): {ratio_nonempty:.3f}",
        "",
        "===== ACCURACY (per property) =====",
    ]
    for col in props:
        report.append(
            f"{col:35}: acc={acc[col]:6.2f}% | mean |%Δ|={mean_err[col]:6.2f}% "
            f"| missed={missed[col]:4d} | extra={extra[col]:4d}"
        )
    report += [
        "",
        f"===== OVERALL ACCURACY =====  {overall:6.2f}%",
        "",
        f"No-match entries     → {no_path.name}",
        f"Multi-match entries   → {multi_path.name}",
        f"Invalid SMILES rows  → {invalid_path.name}",
        "",
        "Figures saved as PNG in the output directory.",
    ]
    res_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    # 8 ── save CID cache
    save_cid_cache(cid_cache_path)


# ── CLI ─────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Evaluate a generated photophysical dataset with "
                    "multi-stage matching, CID caching, and automatic plots.")
    ap.add_argument("original_csv")
    ap.add_argument("generated_csv")
    ap.add_argument("-o", "--outdir", default="evaluation_results")
    args = ap.parse_args()
    evaluate(Path(args.original_csv), Path(args.generated_csv), Path(args.outdir))

