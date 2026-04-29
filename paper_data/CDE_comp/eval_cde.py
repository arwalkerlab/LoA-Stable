#!/usr/bin/env python3
"""
evaluate_generated_dataset.py

Updated to support both:
1) the original generated CSV schema
2) the newer extraction CSV schema (e.g. cde_100_results.csv)

Key additions for the new schema
--------------------------------
- Detects and normalizes new generated columns:
    paper, doi, compound, absorption_max, emission_max,
    quantum_yield, lifetime, molar_absorptivity, status
- Uses DOI directly when it is already present.
- Filters new-schema rows to status == "extracted" by default.
- Falls back to DOI-level numeric assignment when the generated file
  does not contain molecule identifiers / solvent columns compatible
  with the reference file.
- Preserves the old identifier-based matching path for the legacy CSV.

Matching strategy
-----------------
- Legacy schema: exact / DOI / PubChem / InChI matching (same as before)
- New schema: within each DOI, assign generated rows to reference rows
  by minimizing property disagreement over overlapping numeric properties

Outputs are kept compatible with the original script.
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── optional external deps ──────────────────────────────────────────────── #
try:
    import pubchempy as pcp
except ImportError:
    pcp = None

try:
    from rdkit import Chem
except ImportError:
    Chem = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

# ── regex & caches ─────────────────────────────────────────────────────── #
DOI_RX = re.compile(r"10\.\S+")
FNAME_RX = re.compile(r"[^A-Za-z0-9_-]+")

cid_cache: dict[str, int | None] = {}
inchi_cache: dict[str, str | None] = {}

# Canonical internal property names (reference-side names)
PROPS = [
    "Absorption max (nm)",
    "Emission max (nm)",
    "Lifetime (ns)",
    "Quantum yield",
    "log(e/mol-1 dm3 cm-1)",
    "abs FWHM (cm-1)",
    "emi FWHM (cm-1)",
    "abs FWHM (nm)",
    "emi FWHM (nm)",
    "Molecular weight (g mol-1)",
]

# Known generated-column aliases -> internal names
GEN_PROPERTY_ALIASES = {
    "Absorption max (nm)": "Absorption max (nm)",
    "absorption_max": "Absorption max (nm)",
    "absorption max": "Absorption max (nm)",

    "Emission max (nm)": "Emission max (nm)",
    "emission_max": "Emission max (nm)",
    "emission max": "Emission max (nm)",

    "Lifetime (ns)": "Lifetime (ns)",
    "lifetime": "Lifetime (ns)",

    "Quantum yield": "Quantum yield",
    "quantum_yield": "Quantum yield",
    "quantum yield": "Quantum yield",

    "log(e/mol-1 dm3 cm-1)": "log(e/mol-1 dm3 cm-1)",
    "molar_absorptivity": "log(e/mol-1 dm3 cm-1)",
    "molar absorptivity": "log(e/mol-1 dm3 cm-1)",

    "abs FWHM (cm-1)": "abs FWHM (cm-1)",
    "emi FWHM (cm-1)": "emi FWHM (cm-1)",
    "abs FWHM (nm)": "abs FWHM (nm)",
    "emi FWHM (nm)": "emi FWHM (nm)",

    "Molecular weight (g mol-1)": "Molecular weight (g mol-1)",
    "molecular_weight": "Molecular weight (g mol-1)",
}


def safe_filename(text: str) -> str:
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
    if pcp is None:
        return None
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
    if Chem is None:
        return None
    if smiles in inchi_cache:
        return inchi_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    conn = None if mol is None else Chem.MolToInchiKey(mol).split("-")[0]
    inchi_cache[smiles] = conn
    return conn


# ── misc helpers ───────────────────────────────────────────────────────── #
def _coerce_value_from_flag(raw, flag, prop):
    """
    Convert raw extracted value + flag into a usable float.
    Returns np.nan if unusable.
    """

    if pd.isna(raw):
        return np.nan

    s = str(raw).strip()

    # --- normalize dashes ---
    s = s.replace("–", "-").replace("—", "-")

    # --- RANGE: 329-334 ---
    if flag == "range_value" or "-" in s:
        parts = re.split(r"\s*-\s*", s)
        try:
            nums = [float(p) for p in parts if p != ""]
            if len(nums) >= 2:
                return float(np.mean(nums))
        except:
            return np.nan

    # --- MULTIPLE VALUES: "400, 420" or "400 and 420" ---
    if flag == "multiple_values":
        try:
            parts = re.split(r"[,\s]+|and", s)
            nums = [float(p) for p in parts if p.strip() != ""]
            if len(nums) > 0:
                return float(np.mean(nums))
        except:
            return np.nan

    # --- INVALID FORMAT (try to salvage) ---
    if flag == "invalid_format":
        try:
            # remove parentheses like 0.86(48)
            s_clean = re.sub(r"\(.*?\)", "", s)
            # remove commas
            s_clean = s_clean.replace(",", "")
            # extract first number
            m = re.search(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", s_clean, re.I)
            if m:
                return float(m.group(0))
        except:
            return np.nan
        return np.nan

    # --- OUT OF RANGE ---
    if isinstance(flag, str) and flag.startswith("out_of_range"):
        # extract numeric inside parentheses if possible
        m = re.search(r"\((.*?)\)", flag)
        if m:
            try:
                return float(m.group(1))
            except:
                return np.nan
        return np.nan

    # --- NORMAL NUMERIC ---
    try:
        val = float(s)
    except:
        return np.nan

    # --- Special handling: Quantum yield scaling ---
    if prop == "Quantum yield":
        if val > 1:  # assume percentage
            val = val / 100.0

    return val

def canonicalise_paper_code(code: str) -> str | float:
    if pd.isna(code) or str(code).strip() == "":
        return np.nan
    code = str(code).strip()
    if code.startswith("10."):
        return code
    dot = code.find(".")
    return f"10.{code}" if dot == -1 else f"10.{code[:dot]}/{code[dot+1:]}"


def extract_doi(text) -> str | float:
    if pd.isna(text):
        return np.nan
    m = DOI_RX.search(str(text))
    return m.group(0) if m else np.nan


def pct_diff(a: float, b: float) -> float:
    return np.nan if pd.isna(a) or pd.isna(b) or b == 0 else abs(a - b) / abs(b) * 100.0


def pct_diff_or_scaled(a: float, b: float) -> float:
    """
    Percent difference when possible; otherwise a stable fallback for zeros.
    Used for DOI-only assignment costs.
    """
    if pd.isna(a) or pd.isna(b):
        return np.nan
    if b != 0:
        return abs(a - b) / abs(b) * 100.0
    if a == 0:
        return 0.0
    return abs(a - b) * 100.0


def safe_mean(series: Iterable[float] | pd.Series) -> float:
    s = pd.Series(list(series)) if not isinstance(series, pd.Series) else series
    s = s.dropna()
    return np.nan if s.empty else float(s.mean())


def first_present(df: pd.DataFrame, candidates: list[str], default=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)


# ── plotting utilities ─────────────────────────────────────────────────── #
def make_scatter(df: pd.DataFrame, col: str, outdir: Path) -> None:
    ref = df.get(f"{col}_ref", pd.Series(dtype=float)).dropna()
    if ref.empty:
        return
    gen = df.loc[ref.index, f"{col}_gen"]
    if gen.empty:
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
    plt.savefig(outdir / f"{safe_filename(col)}_scatter.png", dpi=300)
    plt.close()


def make_hist(df: pd.DataFrame, col: str, outdir: Path) -> None:
    err = df.get(f"{col}_err", pd.Series(dtype=float)).dropna()
    if err.empty:
        return
    plt.figure()
    plt.hist(err, bins=30, alpha=0.7)
    plt.xlabel("|%Δ|")
    plt.ylabel("Count")
    plt.title(f"Error distribution – {col}")
    plt.tight_layout()
    plt.savefig(outdir / f"{safe_filename(col)}_hist.png", dpi=300)
    plt.close()


def make_error_boxplot(df: pd.DataFrame, props: list[str], outdir: Path, fname: str = "error_boxplot.png") -> None:
    data, labels = [], []
    for p in props:
        d = df.get(f"{p}_err", pd.Series(dtype=float)).dropna()
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
    plt.savefig(outdir / fname, dpi=300)
    plt.close()


def make_publication_figure(
    df: pd.DataFrame,
    props: list[str],
    outdir: Path,
    qy_col: str = "Quantum yield",
    fname: str = "publication_figure.png",
) -> None:
    from matplotlib.gridspec import GridSpec

    props_wo_qy = [p for p in props if p != qy_col]
    data_wo, labels_wo = [], []
    for p in props_wo_qy:
        d = df.get(f"{p}_err", pd.Series(dtype=float)).dropna()
        if len(d):
            data_wo.append(d)
            labels_wo.append(p)

    qy_err = df.get(f"{qy_col}_err", pd.Series(dtype=float)).dropna()
    qy_ref = df.get(f"{qy_col}_ref", pd.Series(dtype=float)).dropna()
    qy_gen = df.get(f"{qy_col}_gen", pd.Series(dtype=float))
    qy_gen = qy_gen.loc[qy_ref.index] if not qy_ref.empty else pd.Series(dtype=float)

    if not data_wo and qy_err.empty and (qy_ref.empty or qy_gen.empty):
        return

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], figure=fig)

    if data_wo:
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.boxplot(data_wo, labels=labels_wo)
        ax_top.set_ylabel("|%Δ|")
        ax_top.set_title("Error distribution – (all except Quantum yield)")
        for label in ax_top.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

    if not qy_err.empty:
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_bl.boxplot([qy_err], labels=[qy_col])
        ax_bl.set_ylabel("|%Δ|")
        ax_bl.set_title("Error distribution – Quantum yield")

    if not qy_ref.empty and not qy_gen.empty:
        ax_br = fig.add_subplot(gs[1, 1])
        ax_br.scatter(qy_ref, qy_gen, alpha=0.6, s=20)
        mn, mx = float(qy_ref.min()), float(qy_ref.max())
        if mx == mn:
            mn *= 0.95
            mx *= 1.05
        ax_br.plot([mn, mx], [mn, mx], ls="--")
        ax_br.set_xlabel(f"Reference {qy_col}")
        ax_br.set_ylabel(f"Generated {qy_col}")
        ax_br.set_title(f"Predicted vs. Reference – {qy_col}")

    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=300)
    plt.close()


# ── raw results I/O ─────────────────────────────────────────────────────── #
def save_raw_results(out_dir: Path, comp_df: pd.DataFrame, coverage_df: pd.DataFrame, props: list[str]) -> None:
    comp_df.to_parquet(out_dir / "comp_df.parquet", index=False)
    coverage_df.to_csv(out_dir / "coverage.csv", index=False)
    (out_dir / "props.json").write_text(json.dumps({"props": props}, indent=2), encoding="utf-8")


def load_raw_results(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    comp_path = out_dir / "comp_df.parquet"
    cov_path = out_dir / "coverage.csv"
    meta_path = out_dir / "props.json"
    if not (comp_path.is_file() and cov_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError(
            "Raw results not found. Expected comp_df.parquet, coverage.csv, and props.json in the output directory."
        )
    comp_df = pd.read_parquet(comp_path)
    coverage_df = pd.read_csv(cov_path)
    props = list(json.loads(meta_path.read_text(encoding="utf-8")).get("props", []))
    return comp_df, coverage_df, props


# ── schema normalization ───────────────────────────────────────────────── #
def normalize_reference(ref: pd.DataFrame) -> pd.DataFrame:
    ref = ref.copy()
    ref = ref.rename(columns={"Chromophore": "smiles_ref", "Reference": "ref_reference"})
    ref["doi"] = ref["ref_reference"].apply(extract_doi) if "ref_reference" in ref.columns else np.nan
    ref["ik_chr"] = ref["smiles_ref"].astype(str).apply(inchi_connectivity) if "smiles_ref" in ref.columns else np.nan

    if "Solvent" in ref.columns:
        ref["ik_solv"] = ref["Solvent"].astype(str).apply(inchi_connectivity)
        ref["solv_lc"] = ref["Solvent"].astype(str).str.strip().str.lower()
    else:
        ref["ik_solv"] = np.nan
        ref["solv_lc"] = np.nan

    for col in PROPS:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")
        else:
            ref[col] = np.nan
    return ref


def normalize_generated(gen: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Returns (normalized_generated_df, matching_mode)
    matching_mode ∈ {"legacy_identifier", "doi_numeric"}
    """
    gen = gen.copy()

    lowered = {c.lower().strip(): c for c in gen.columns}

    # Detect the new schema
    is_new_schema = {"paper", "doi", "compound"}.issubset(set(lowered.keys()))

    if is_new_schema:
        gen["source_row_status"] = first_present(gen, ["status"]).astype(str).str.strip().str.lower()
        gen = gen[gen["source_row_status"].eq("extracted")].copy()

        # direct DOI if already present
        gen["doi"] = first_present(gen, ["doi"]).apply(extract_doi)

        # best-effort free-text fields
        gen["smiles_gen"] = np.nan
        gen["solvent"] = np.nan
        gen["paper_code"] = first_present(gen, ["paper"]).astype(str)

        # property normalization
        for raw_name, canon_name in GEN_PROPERTY_ALIASES.items():
            if raw_name in gen.columns:
                gen[canon_name] = pd.to_numeric(gen[raw_name], errors="coerce")

        for col in PROPS:
            if col not in gen.columns:
                gen[col] = np.nan

        gen["matching_mode"] = "doi_numeric"
        return gen, "doi_numeric"

    # Legacy/original schema
    gen = gen.rename(columns={"molecule_name": "smiles_gen", "paper": "paper_code"})
    if "doi" in gen.columns:
        gen["doi"] = gen["doi"].apply(extract_doi)
    else:
        gen["doi"] = gen["paper_code"].apply(canonicalise_paper_code) if "paper_code" in gen.columns else np.nan

    if "solvent" not in gen.columns:
        gen["solvent"] = np.nan

    for raw_name, canon_name in GEN_PROPERTY_ALIASES.items():
        if raw_name in gen.columns and canon_name not in gen.columns:
            gen[canon_name] = pd.to_numeric(gen[raw_name], errors="coerce")

    for col in PROPS:
        if col in gen.columns:
            gen[col] = pd.to_numeric(gen[col], errors="coerce")
        else:
            gen[col] = np.nan

    gen["matching_mode"] = "legacy_identifier"
    return gen, "legacy_identifier"


# ── matching helpers ───────────────────────────────────────────────────── #
def build_record(gen_row: pd.Series, ref_row: pd.Series, props: list[str]) -> tuple[dict[str, float], dict[str, list | int]]:
    rec: dict[str, float] = {}
    diff_local = {p: [] for p in props}
    missed_local = {p: 0 for p in props}
    extra_local = {p: 0 for p in props}

    for col in props:
        g = gen_row.get(col, np.nan)
        r = ref_row.get(col, np.nan)
        err = pct_diff(g, r)

        rec[f"{col}_gen"] = g
        rec[f"{col}_ref"] = r
        rec[f"{col}_err"] = err

        if pd.isna(g) and pd.isna(r):
            continue
        if pd.isna(g) and not pd.isna(r):
            missed_local[col] += 1
        elif not pd.isna(g) and pd.isna(r):
            extra_local[col] += 1
        else:
            diff_local[col].append(err)

    return rec, {"diff": diff_local, "missed": missed_local, "extra": extra_local}


def merge_stats(diff, missed, extra, local_stats):
    for p, vals in local_stats["diff"].items():
        diff[p].extend(vals)
    for p, v in local_stats["missed"].items():
        missed[p] += v
    for p, v in local_stats["extra"].items():
        extra[p] += v


def row_pair_cost(gen_row: pd.Series, ref_row: pd.Series, props: list[str]) -> float:
    costs = []
    for col in props:
        g = gen_row.get(col, np.nan)
        r = ref_row.get(col, np.nan)
        if pd.isna(g) or pd.isna(r):
            continue
        d = pct_diff_or_scaled(g, r)
        if pd.notna(d):
            costs.append(min(float(d), 500.0))
    if not costs:
        return 1e9
    return float(np.mean(costs))


def greedy_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    pairs = []
    used_rows, used_cols = set(), set()
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort(key=lambda x: x[0])
    for c, i, j in flat:
        if i in used_rows or j in used_cols:
            continue
        if not np.isfinite(c) or c >= 1e9:
            continue
        pairs.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return pairs


def assign_within_doi(gen_doi: pd.DataFrame, ref_doi: pd.DataFrame, props: list[str]) -> list[tuple[int, int]]:
    if gen_doi.empty or ref_doi.empty:
        return []

    cost = np.full((len(gen_doi), len(ref_doi)), 1e9, dtype=float)
    gen_idx = list(gen_doi.index)
    ref_idx = list(ref_doi.index)

    for i, gi in enumerate(gen_idx):
        for j, rj in enumerate(ref_idx):
            cost[i, j] = row_pair_cost(gen_doi.loc[gi], ref_doi.loc[rj], props)

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        pairs = []
        for i, j in zip(rows, cols):
            if np.isfinite(cost[i, j]) and cost[i, j] < 1e9:
                pairs.append((gen_idx[i], ref_idx[j]))
        return pairs

    pairs_local = greedy_assignment(cost)
    return [(gen_idx[i], ref_idx[j]) for i, j in pairs_local]


def evaluate_legacy_rows(
    ref: pd.DataFrame,
    gen: pd.DataFrame,
    props: list[str],
    no_lines: list[str],
    multi_lines: list[str],
    invalid_lines: list[str],
) -> tuple[list[dict[str, float]], dict, dict, dict]:
    comp_records: list[dict[str, float]] = []
    diff = {p: [] for p in props}
    missed = {p: 0 for p in props}
    extra = {p: 0 for p in props}

    for idx, row in gen.iterrows():
        smi_g = str(row.get("smiles_gen", "")).strip()
        if smi_g.lower() in {"failed", "nan", ""}:
            continue

        solv_g = str(row.get("solvent", "")).strip()
        doi_g = row.get("doi", np.nan)

        mask = (ref["smiles_ref"] == smi_g)
        if solv_g and "solv_lc" in ref.columns:
            mask &= ref["solv_lc"] == solv_g.lower()
        cand = ref[mask]

        if len(cand) != 1 and pd.notna(doi_g):
            cand = cand[cand["doi"] == doi_g] if len(cand) else ref[(ref["doi"] == doi_g) & (ref["smiles_ref"] == smi_g)]

        if len(cand) != 1 and pd.notna(doi_g):
            cid_chr_g = pubchem_cid(smi_g)
            cid_solv_g = pubchem_cid(solv_g) if solv_g else None
            if cid_chr_g is not None:
                cand_doi = ref[ref["doi"] == doi_g]
                cand = cand_doi[cand_doi["smiles_ref"].apply(pubchem_cid) == cid_chr_g]
                if len(cand) > 1 and cid_solv_g is not None and "Solvent" in cand.columns:
                    cand = cand[cand["Solvent"].apply(pubchem_cid) == cid_solv_g]

        if len(cand) != 1 and pd.notna(doi_g):
            ik_chr_g = inchi_connectivity(smi_g)
            ik_solv_g = inchi_connectivity(solv_g) if solv_g else None
            if ik_chr_g is None and pubchem_cid(smi_g) is None:
                invalid_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tinvalid_smiles\t0\n")
                continue
            cand_doi = ref[ref["doi"] == doi_g]
            cand = cand_doi[cand_doi["ik_chr"] == ik_chr_g]
            if len(cand) > 1 and ik_solv_g is not None:
                cand = cand[cand["ik_solv"] == ik_solv_g]

        if len(cand) == 0:
            no_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tno_match\t0\n")
            continue
        if len(cand) > 1:
            multi_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tmulti\t{len(cand)}\n")
            continue

        ref_row = cand.iloc[0]
        rec, local_stats = build_record(row, ref_row, props)
        comp_records.append(rec)
        merge_stats(diff, missed, extra, local_stats)

    return comp_records, diff, missed, extra


def evaluate_doi_numeric_rows(
    ref: pd.DataFrame,
    gen: pd.DataFrame,
    props: list[str],
    no_lines: list[str],
) -> tuple[list[dict[str, float]], dict, dict, dict]:
    comp_records: list[dict[str, float]] = []
    diff = {p: [] for p in props}
    missed = {p: 0 for p in props}
    extra = {p: 0 for p in props}

    ref_by_doi = {doi: grp.copy() for doi, grp in ref.groupby("doi", dropna=True)}
    gen_by_doi = {doi: grp.copy() for doi, grp in gen.groupby("doi", dropna=True)}

    for doi_g, ggrp in gen_by_doi.items():
        rgrp = ref_by_doi.get(doi_g)
        if rgrp is None or rgrp.empty:
            for idx, row in ggrp.iterrows():
                no_lines.append(f"{idx}\t{row.get('compound', '')}\t\t{doi_g or ''}\tno_reference_doi\t0\n")
            continue

        pairs = assign_within_doi(ggrp, rgrp, props)
        matched_gen = set()
        for gi, rj in pairs:
            matched_gen.add(gi)
            rec, local_stats = build_record(ggrp.loc[gi], rgrp.loc[rj], props)
            comp_records.append(rec)
            merge_stats(diff, missed, extra, local_stats)

        unmatched = set(ggrp.index) - matched_gen
        for gi in unmatched:
            row = ggrp.loc[gi]
            no_lines.append(f"{gi}\t{row.get('compound', '')}\t\t{doi_g or ''}\tunassigned_within_doi\t0\n")

    # also log rows with missing DOI
    missing_doi_rows = gen[gen["doi"].isna()]
    for idx, row in missing_doi_rows.iterrows():
        no_lines.append(f"{idx}\t{row.get('compound', '')}\t\t\tmissing_doi\t0\n")

    return comp_records, diff, missed, extra


# ── evaluation core ─────────────────────────────────────────────────────── #
def evaluate(original_csv: Path, generated_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cid_cache_path = out_dir / "cid_cache.json"
    load_cid_cache(cid_cache_path)

    res_path = out_dir / "final_results.txt"
    no_path = out_dir / "no_match_entries.txt"
    multi_path = out_dir / "multi_match_entries.txt"
    invalid_path = out_dir / "invalid_smiles.txt"

    na_vals = ["null", "NULL", "", "NaN", "nan"]
    ref_raw = pd.read_csv(original_csv, sep=None, engine="python", na_values=na_vals, keep_default_na=True)
    gen_raw = pd.read_csv(generated_csv, sep=None, engine="python", na_values=na_vals, keep_default_na=True)

    ref = normalize_reference(ref_raw)
    gen, matching_mode = normalize_generated(gen_raw)

    props = [p for p in PROPS if p in ref.columns]

    hdr = "idx\tSMILES_or_compound\tSolvent\tGen_DOI\tReason\t#ref_matches\n"
    no_lines, multi_lines, invalid_lines = [], [], []

    # coverage
    cov = (
        pd.DataFrame({"gen": gen.groupby("doi", dropna=True).size(), "ref": ref.groupby("doi", dropna=True).size()})
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": "doi"})
    )
    cov["ratio"] = np.where(cov["ref"] == 0, np.nan, cov["gen"] / cov["ref"])
    ratio_all = safe_mean(cov["ratio"])
    ratio_nonempty = safe_mean(cov.loc[cov["gen"] > 0, "ratio"])

    if matching_mode == "legacy_identifier":
        comp_records, diff, missed, extra = evaluate_legacy_rows(ref, gen, props, no_lines, multi_lines, invalid_lines)
    else:
        comp_records, diff, missed, extra = evaluate_doi_numeric_rows(ref, gen, props, no_lines)

    comp_df = pd.DataFrame(comp_records)

    mean_err = {c: safe_mean(pd.Series(v)) for c, v in diff.items()}
    acc = {c: 100 - e if not pd.isna(e) else np.nan for c, e in mean_err.items()}
    overall = safe_mean(pd.Series(acc.values()))

    no_path.write_text(hdr + "".join(no_lines), encoding="utf-8")
    multi_path.write_text(hdr + "".join(multi_lines), encoding="utf-8")
    invalid_path.write_text(hdr + "".join(invalid_lines), encoding="utf-8")

    for col in props:
        if f"{col}_gen" in comp_df.columns:
            make_scatter(comp_df, col, out_dir)
            make_hist(comp_df, col, out_dir)
    make_error_boxplot(comp_df, props, out_dir)
    make_publication_figure(comp_df, props, out_dir, qy_col="Quantum yield", fname="publication_figure.png")

    report = [
        "===== MODE =====",
        f"Matching mode: {matching_mode}",
        "",
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
        f"No-match entries      → {no_path.name}",
        f"Multi-match entries   → {multi_path.name}",
        f"Invalid SMILES rows   → {invalid_path.name}",
        "",
        "Figures saved as PNG in the output directory.",
        "Publication figure    → publication_figure.png",
    ]
    res_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    save_cid_cache(cid_cache_path)
    save_raw_results(out_dir, comp_df, cov, props)
    return comp_df, cov, props


def figures_only(out_dir: Path) -> None:
    comp_df, cov, props = load_raw_results(out_dir)
    for col in props:
        if f"{col}_gen" in comp_df.columns:
            make_scatter(comp_df, col, out_dir)
            make_hist(comp_df, col, out_dir)
    make_error_boxplot(comp_df, props, out_dir)
    make_publication_figure(comp_df, props, out_dir, qy_col="Quantum yield", fname="publication_figure.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a generated photophysical dataset. Supports both the legacy generated CSV "
            "schema and the newer extraction CSV schema."
        )
    )
    ap.add_argument("original_csv", nargs="?", help="Original reference CSV (ignored with --reuse)")
    ap.add_argument("generated_csv", nargs="?", help="Generated CSV to evaluate (ignored with --reuse)")
    ap.add_argument("-o", "--outdir", default="evaluation_results", help="Output directory")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="Skip recomputation: load raw results from outdir and regenerate figures only.",
    )
    args = ap.parse_args()

    out_dir = Path(args.outdir)

    if args.reuse:
        try:
            figures_only(out_dir)
            print(f"[OK] Figures regenerated from raw results in: {out_dir}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}\nRun a full evaluation first (without --reuse).", file=sys.stderr)
            sys.exit(1)
    else:
        if args.original_csv is None or args.generated_csv is None:
            print("ERROR: original_csv and generated_csv are required unless using --reuse.", file=sys.stderr)
            sys.exit(2)
        evaluate(Path(args.original_csv), Path(args.generated_csv), out_dir)
