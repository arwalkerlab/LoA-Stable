#!/usr/bin/env python3
"""
evaluate_generated_dataset.py

New-schema behavior:
- only compounds in the dataset being checked are resolved to SMILES
- reference dataset is NOT sent to PubChem for name lookup
- reference Chromophore values are canonicalized locally with RDKit only
- generated compound resolution prefers PubChem for name-like strings
  and RDKit only for probable SMILES strings
- DOI is still used to restrict candidate reference rows

Important:
- Old cached failed resolutions can prevent new PubChem attempts.
  This script supports --refresh-name-cache to ignore/delete that cache.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pubchempy as pcp
except ImportError:
    pcp = None

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    Chem = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


WORST_ROWS_PER_PROPERTY = 5

DOI_RX = re.compile(r"10\.\S+")
FNAME_RX = re.compile(r"[^A-Za-z0-9_-]+")
NUM_RX = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
TIMES10_RX = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:x|×)\s*10\s*\^?\s*([-+]?\d+)\s*$",
    re.IGNORECASE,
)

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

FLAG_COLUMNS = {
    "Absorption max (nm)": "absorption_max_flag",
    "Emission max (nm)": "emission_max_flag",
    "Lifetime (ns)": "lifetime_flag",
    "Quantum yield": "quantum_yield_flag",
    "log(e/mol-1 dm3 cm-1)": "molar_absorptivity_flag",
    "abs FWHM (cm-1)": "abs_fwhm_cm_flag",
    "emi FWHM (cm-1)": "emi_fwhm_cm_flag",
    "abs FWHM (nm)": "abs_fwhm_nm_flag",
    "emi FWHM (nm)": "emi_fwhm_nm_flag",
    "Molecular weight (g mol-1)": "molecular_weight_flag",
}

SANITY_BOUNDS = {
    "Absorption max (nm)": (150.0, 2000.0),
    "Emission max (nm)": (150.0, 2500.0),
    "Lifetime (ns)": (0.0, 1e7),
    "Quantum yield": (0.0, 1.0),
    "log(e/mol-1 dm3 cm-1)": (0.0, 8.0),
    "abs FWHM (cm-1)": (1.0, 50000.0),
    "emi FWHM (cm-1)": (1.0, 50000.0),
    "abs FWHM (nm)": (0.1, 5000.0),
    "emi FWHM (nm)": (0.1, 5000.0),
    "Molecular weight (g mol-1)": (1.0, 10000.0),
}

cid_cache: dict[str, int | None] = {}
inchi_cache: dict[str, str | None] = {}
name_resolve_cache: dict[str, dict[str, object]] = {}


def safe_filename(text: str) -> str:
    return FNAME_RX.sub("_", text).strip("_")


def load_json_cache(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_json_cache(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def load_caches(out_dir: Path, refresh_name_cache: bool = False) -> None:
    global cid_cache, inchi_cache, name_resolve_cache
    cid_cache = load_json_cache(out_dir / "cid_cache.json")
    inchi_cache = load_json_cache(out_dir / "inchi_cache.json")
    if refresh_name_cache:
        name_resolve_cache = {}
    else:
        name_resolve_cache = load_json_cache(out_dir / "name_resolve_cache.json")


def save_caches(out_dir: Path) -> None:
    save_json_cache(out_dir / "cid_cache.json", cid_cache)
    save_json_cache(out_dir / "inchi_cache.json", inchi_cache)
    save_json_cache(out_dir / "name_resolve_cache.json", name_resolve_cache)


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
    
def doi_from_paper_filename(text) -> str | float:
    if pd.isna(text):
        return np.nan

    s = str(text).strip()
    if not s:
        return np.nan

    # remove trailing .pdf if present
    if s.lower().endswith(".pdf"):
        s = s[:-4]

    # first dot becomes slash
    if "." in s:
        left, right = s.split(".", 1)
        s = f"{left}/{right}"

    # add leading 10.
    s = f"10.{s}"

    return extract_doi(s)


def pct_diff(a: float, b: float) -> float:
    return np.nan if pd.isna(a) or pd.isna(b) or b == 0 else abs(a - b) / abs(b) * 100.0


def pct_diff_or_scaled(a: float, b: float) -> float:
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


def parse_number_like(text) -> float | None:
    if pd.isna(text):
        return None
    s = str(text).strip()
    if not s:
        return None

    s = s.replace(",", "")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("—", "-")
    s = s.replace("·", ".")
    s = s.replace("µ", "u")

    m = TIMES10_RX.match(s)
    if m:
        try:
            base = float(m.group(1))
            exp = int(m.group(2))
            return float(base * (10 ** exp))
        except Exception:
            pass

    try:
        return float(s)
    except Exception:
        pass

    nums = NUM_RX.findall(s)
    if len(nums) == 1:
        try:
            return float(nums[0])
        except Exception:
            return None
    return None


def parse_range_or_single(text) -> list[float]:
    if pd.isna(text):
        return []
    s = str(text).strip()
    if not s:
        return []

    s2 = s.replace(",", "")
    s2 = s2.replace("−", "-").replace("–", "-").replace("—", "-")
    s2 = s2.replace(" to ", "-").replace(" TO ", "-").replace(" To ", "-")

    if TIMES10_RX.match(s2):
        val = parse_number_like(s2)
        return [] if val is None else [val]

    nums = NUM_RX.findall(s2)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def sanity_filter(prop: str, value: float | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    lo, hi = SANITY_BOUNDS[prop]
    if value < lo or value > hi:
        return np.nan
    return float(value)


def clean_property_value(raw_value, flag_value, prop: str) -> float:
    flag = "" if pd.isna(flag_value) else str(flag_value).strip().lower()
    raw = raw_value

    if "invalid_format" in flag:
        return np.nan
    if "multiple_values" in flag:
        return np.nan

    if prop == "Quantum yield":
        val = parse_number_like(raw)
        if val is None:
            vals = parse_range_or_single(raw)
            if len(vals) >= 2:
                val = float(np.mean(vals[:2]))
            elif len(vals) == 1:
                val = vals[0]
            else:
                return np.nan

        if 0.0 <= val <= 1.0:
            return sanity_filter(prop, val)
        if 1.0 < val <= 100.0:
            return sanity_filter(prop, val / 100.0)
        return np.nan

    if "out_of_range" in flag:
        return np.nan

    if "range_value" in flag:
        vals = parse_range_or_single(raw)
        if len(vals) >= 2:
            return sanity_filter(prop, float(np.mean(vals[:2])))
        if len(vals) == 1:
            return sanity_filter(prop, vals[0])
        return np.nan

    vals = parse_range_or_single(raw)
    if len(vals) >= 2:
        return sanity_filter(prop, float(np.mean(vals[:2])))

    if len(vals) == 1:
        val = vals[0]
    else:
        val = parse_number_like(raw)

    if val is None:
        return np.nan

    if prop == "log(e/mol-1 dm3 cm-1)" and val > 20:
        if val <= 0:
            return np.nan
        val = math.log10(val)

    return sanity_filter(prop, val)


def rdkit_canonical_smiles(text: str) -> str | None:
    if Chem is None:
        return None
    if text is None or pd.isna(text):
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def inchi_connectivity_from_smiles(smiles: str) -> str | None:
    if Chem is None:
        return None
    if smiles in inchi_cache:
        return inchi_cache[smiles]
    try:
        mol = Chem.MolFromSmiles(smiles)
        conn = None if mol is None else Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        conn = None
    inchi_cache[smiles] = conn
    return conn


def is_probable_smiles(text: str) -> bool:
    if text is None or pd.isna(text):
        return False
    s = str(text).strip()
    if not s:
        return False

    # spaces almost always mean names / phrases, not SMILES
    if " " in s or "\t" in s:
        return False

    # common indicators of names / prose
    lower = s.lower()
    if any(ch.isalpha() for ch in s):
        # if it contains lowercase letters beyond allowed aromatic symbols,
        # it's probably a name rather than a SMILES string
        letters = re.findall(r"[A-Za-z]", s)
        allowed_aromatic = set("BCNOFPSIbcnofpsiklbr")
        if any(ch not in allowed_aromatic for ch in letters):
            return False

    # characters commonly seen in SMILES
    smiles_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#@+-\\/%.")
    if not all(ch in smiles_chars for ch in s):
        return False

    # very short plain all-caps tokens are more likely labels than SMILES
    if re.fullmatch(r"[A-Z0-9]{2,8}", s):
        return False

    return True


def pubchem_name_to_smiles(name: str) -> str | None:
    if pcp is None:
        return None
    q = str(name).strip()
    if not q:
        return None
    try:
        comps = pcp.get_compounds(q, "name")
        if not comps:
            return None
        smi = getattr(comps[0], "canonical_smiles", None)
        if smi:
            return rdkit_canonical_smiles(smi) or smi
        return None
    except Exception:
        return None


def resolve_identifier(text) -> dict[str, object]:
    """
    Resolve ONLY dataset-under-test compounds.

    Logic:
    - if it looks like SMILES, try RDKit first
    - otherwise go straight to PubChem name lookup
    - if probable SMILES failed in RDKit, still try PubChem as a fallback
    """
    if pd.isna(text):
        return {"gen_smiles": np.nan, "gen_ik": np.nan, "resolved_via": np.nan}

    raw = str(text).strip()
    if not raw:
        return {"gen_smiles": np.nan, "gen_ik": np.nan, "resolved_via": np.nan}

    if raw in name_resolve_cache:
        return name_resolve_cache[raw]

    result = {"gen_smiles": np.nan, "gen_ik": np.nan, "resolved_via": np.nan}

    probable_smiles = is_probable_smiles(raw)
    if probable_smiles:
        can = rdkit_canonical_smiles(raw)
        if can:
            result = {
                "gen_smiles": can,
                "gen_ik": inchi_connectivity_from_smiles(can),
                "resolved_via": "rdkit_smiles",
            }
            name_resolve_cache[raw] = result
            return result

    name_smiles = pubchem_name_to_smiles(raw)
    if name_smiles:
        result = {
            "gen_smiles": name_smiles,
            "gen_ik": inchi_connectivity_from_smiles(name_smiles),
            "resolved_via": "pubchem_name",
        }
        name_resolve_cache[raw] = result
        return result

    name_resolve_cache[raw] = result
    return result


def reference_identity_from_smiles(smiles) -> dict[str, object]:
    if pd.isna(smiles):
        return {"ref_smiles_canonical": np.nan, "ref_ik": np.nan}
    can = rdkit_canonical_smiles(str(smiles).strip())
    if not can:
        return {"ref_smiles_canonical": np.nan, "ref_ik": np.nan}
    return {
        "ref_smiles_canonical": can,
        "ref_ik": inchi_connectivity_from_smiles(can),
    }


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


def normalize_reference(ref: pd.DataFrame) -> pd.DataFrame:
    ref = ref.copy()
    ref = ref.rename(columns={"Chromophore": "smiles_ref", "Reference": "ref_reference"})
    ref["doi"] = ref["ref_reference"].apply(extract_doi) if "ref_reference" in ref.columns else np.nan

    ids = ref["smiles_ref"].apply(reference_identity_from_smiles).apply(pd.Series) if "smiles_ref" in ref.columns else pd.DataFrame(index=ref.index)
    for c in ["ref_smiles_canonical", "ref_ik"]:
        ref[c] = ids[c] if c in ids.columns else np.nan

    if "Solvent" in ref.columns:
        ref["solv_lc"] = ref["Solvent"].astype(str).str.strip().str.lower()
    else:
        ref["solv_lc"] = np.nan

    for col in PROPS:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")
        else:
            ref[col] = np.nan
    return ref


def normalize_generated(gen: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    gen = gen.copy()
    lowered = {c.lower().strip(): c for c in gen.columns}
    is_new_schema = {"paper", "doi", "compound"}.issubset(set(lowered.keys()))

    if is_new_schema:
        gen["source_row_status"] = first_present(gen, ["status"]).astype(str).str.strip().str.lower()
        gen = gen[gen["source_row_status"].eq("extracted")].copy()

        raw_doi = first_present(gen, ["doi"])
        paper_fallback = first_present(gen, ["paper"])

        gen["doi"] = raw_doi.apply(extract_doi)
        missing_mask = gen["doi"].isna()

        if missing_mask.any():
            gen.loc[missing_mask, "doi"] = paper_fallback.loc[missing_mask].apply(doi_from_paper_filename)
        gen["paper_code"] = first_present(gen, ["paper"]).astype(str)
        gen["smiles_gen"] = np.nan
        gen["solvent"] = np.nan

        for col in PROPS:
            gen[f"__raw__{col}"] = np.nan
            gen[f"__flag__{col}"] = np.nan
            gen[col] = np.nan

        for raw_name, canon_name in GEN_PROPERTY_ALIASES.items():
            if raw_name in gen.columns:
                gen[f"__raw__{canon_name}"] = gen[raw_name]

        for prop, flag_col in FLAG_COLUMNS.items():
            if flag_col in gen.columns:
                gen[f"__flag__{prop}"] = gen[flag_col]

        for prop in PROPS:
            gen[prop] = [
                clean_property_value(raw, flag, prop)
                for raw, flag in zip(gen[f"__raw__{prop}"], gen[f"__flag__{prop}"])
            ]

        print(f"[INFO] Resolving generated compounds for {len(gen)} extracted rows...", flush=True)
        resolved = gen["compound"].apply(resolve_identifier).apply(pd.Series)
        for c in ["gen_smiles", "gen_ik", "resolved_via"]:
            gen[c] = resolved[c] if c in resolved.columns else np.nan

        rdkit_hits = int((gen["resolved_via"] == "rdkit_smiles").sum())
        pubchem_hits = int((gen["resolved_via"] == "pubchem_name").sum())
        unresolved = int(gen["gen_smiles"].isna().sum())
        print(f"[INFO] Compound resolution summary: RDKit={rdkit_hits}, PubChem={pubchem_hits}, unresolved={unresolved}", flush=True)

        gen["matching_mode"] = "resolved_generated_to_reference_smiles"
        return gen, "resolved_generated_to_reference_smiles"

    gen = gen.rename(columns={"molecule_name": "smiles_gen", "paper": "paper_code"})
    if "doi" in gen.columns:
        gen["doi"] = gen["doi"].apply(extract_doi)
    else:
        gen["doi"] = gen["paper_code"].apply(canonicalise_paper_code) if "paper_code" in gen.columns else np.nan

    if "solvent" not in gen.columns:
        gen["solvent"] = np.nan

    for col in PROPS:
        gen[f"__raw__{col}"] = gen[col] if col in gen.columns else np.nan
        gen[f"__flag__{col}"] = np.nan
        if col in gen.columns:
            gen[col] = pd.to_numeric(gen[col], errors="coerce")
        else:
            gen[col] = np.nan

    if "smiles_gen" in gen.columns:
        gen["gen_smiles"] = gen["smiles_gen"].apply(rdkit_canonical_smiles)
        gen["gen_ik"] = gen["gen_smiles"].apply(inchi_connectivity_from_smiles)
        gen["resolved_via"] = np.where(gen["gen_smiles"].notna(), "legacy_smiles", np.nan)
    else:
        gen["gen_smiles"] = np.nan
        gen["gen_ik"] = np.nan
        gen["resolved_via"] = np.nan

    gen["matching_mode"] = "legacy_identifier"
    return gen, "legacy_identifier"


def build_record(
    gen_row: pd.Series,
    ref_row: pd.Series,
    props: list[str],
    *,
    match_cost: float | None = None,
    matching_mode: str = "",
) -> tuple[dict[str, float], dict[str, list | int]]:
    rec: dict[str, float] = {
        "_matching_mode": matching_mode,
        "_gen_index": gen_row.name,
        "_ref_index": ref_row.name,
        "_doi": gen_row.get("doi", ref_row.get("doi", np.nan)),
        "_compound": gen_row.get("compound", np.nan),
        "_smiles_gen": gen_row.get("gen_smiles", gen_row.get("smiles_gen", np.nan)),
        "_smiles_ref": ref_row.get("ref_smiles_canonical", ref_row.get("smiles_ref", np.nan)),
        "_solvent": gen_row.get("solvent", np.nan),
        "_paper": gen_row.get("paper_code", np.nan),
        "_match_cost": match_cost,
        "_resolved_via": gen_row.get("resolved_via", np.nan),
    }

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
        rec[f"{col}_gen_raw"] = gen_row.get(f"__raw__{col}", np.nan)
        rec[f"{col}_gen_flag"] = gen_row.get(f"__flag__{col}", np.nan)

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


def assign_within_doi(gen_doi: pd.DataFrame, ref_doi: pd.DataFrame, props: list[str]) -> list[tuple[int, int, float]]:
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
                pairs.append((gen_idx[i], ref_idx[j], float(cost[i, j])))
        return pairs

    pairs_local = greedy_assignment(cost)
    return [(gen_idx[i], ref_idx[j], float(cost[i, j])) for i, j in pairs_local]


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
            ik_chr_g = inchi_connectivity_from_smiles(smi_g)
            if ik_chr_g is None:
                invalid_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tinvalid_smiles\t0\n")
                continue
            cand_doi = ref[ref["doi"] == doi_g]
            cand = cand_doi[cand_doi["ref_ik"] == ik_chr_g]

        if len(cand) == 0:
            no_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tno_match\t0\n")
            continue
        if len(cand) > 1:
            multi_lines.append(f"{idx}\t{smi_g}\t{solv_g}\t{doi_g or ''}\tmulti\t{len(cand)}\n")
            continue

        ref_row = cand.iloc[0]
        rec, local_stats = build_record(row, ref_row, props, matching_mode="legacy_identifier")
        comp_records.append(rec)
        merge_stats(diff, missed, extra, local_stats)

    return comp_records, diff, missed, extra


def evaluate_resolved_generated_rows(
    ref: pd.DataFrame,
    gen: pd.DataFrame,
    props: list[str],
    no_lines: list[str],
    multi_lines: list[str],
    allow_doi_fallback: bool,
) -> tuple[list[dict[str, float]], dict, dict, dict]:
    comp_records: list[dict[str, float]] = []
    diff = {p: [] for p in props}
    missed = {p: 0 for p in props}
    extra = {p: 0 for p in props}

    gen_matchable = gen[gen["gen_smiles"].notna()].copy()
    print(f"[INFO] Matchable generated rows after compound resolution: {len(gen_matchable)}", flush=True)

    used_ref_idx = set()
    unresolved_rows = []

    for idx, row in gen_matchable.iterrows():
        doi_g = row.get("doi", np.nan)
        if pd.isna(doi_g):
            no_lines.append(f"{idx}\t{row.get('compound','')}\t\t\tmissing_doi\t0\n")
            continue

        cand = ref[ref["doi"] == doi_g].copy()
        if cand.empty:
            no_lines.append(f"{idx}\t{row.get('compound','')}\t\t{doi_g}\tno_reference_doi\t0\n")
            continue

        g_smi = row.get("gen_smiles", np.nan)
        g_ik = row.get("gen_ik", np.nan)

        cand_exact = cand[cand["ref_smiles_canonical"] == g_smi]
        cand_exact = cand_exact.loc[~cand_exact.index.isin(used_ref_idx)]
        if len(cand_exact) == 1:
            chosen = cand_exact.iloc[0]
            rec, local_stats = build_record(row, chosen, props, matching_mode="resolved_generated_to_reference_smiles")
            comp_records.append(rec)
            merge_stats(diff, missed, extra, local_stats)
            used_ref_idx.add(chosen.name)
            continue

        cand_ik = cand[cand["ref_ik"] == g_ik] if pd.notna(g_ik) else cand.iloc[0:0]
        cand_ik = cand_ik.loc[~cand_ik.index.isin(used_ref_idx)]
        if len(cand_ik) == 1:
            chosen = cand_ik.iloc[0]
            rec, local_stats = build_record(row, chosen, props, matching_mode="resolved_generated_to_reference_smiles")
            comp_records.append(rec)
            merge_stats(diff, missed, extra, local_stats)
            used_ref_idx.add(chosen.name)
            continue

        if len(cand_exact) > 1 or len(cand_ik) > 1:
            multi_lines.append(f"{idx}\t{row.get('compound','')}\t\t{doi_g}\tmulti_identity_match\t{max(len(cand_exact), len(cand_ik))}\n")
            continue

        unresolved_rows.append(idx)

    if allow_doi_fallback and unresolved_rows:
        fallback_gen = gen_matchable.loc[unresolved_rows].copy()
        for doi_g, ggrp in fallback_gen.groupby("doi", dropna=True):
            rgrp = ref[(ref["doi"] == doi_g) & (~ref.index.isin(used_ref_idx))].copy()
            if rgrp.empty:
                for idx, row in ggrp.iterrows():
                    no_lines.append(f"{idx}\t{row.get('compound','')}\t\t{doi_g}\tno_reference_after_identity\t0\n")
                continue

            pairs = assign_within_doi(ggrp, rgrp, props)
            matched_gen = set()
            for gi, rj, pair_cost in pairs:
                matched_gen.add(gi)
                used_ref_idx.add(rj)
                rec, local_stats = build_record(
                    ggrp.loc[gi], rgrp.loc[rj], props,
                    match_cost=pair_cost, matching_mode="doi_numeric_fallback"
                )
                comp_records.append(rec)
                merge_stats(diff, missed, extra, local_stats)

            for gi in set(ggrp.index) - matched_gen:
                row = ggrp.loc[gi]
                no_lines.append(f"{gi}\t{row.get('compound','')}\t\t{doi_g}\tunmatched_after_fallback\t0\n")
    else:
        for idx in unresolved_rows:
            row = gen_matchable.loc[idx]
            no_lines.append(f"{idx}\t{row.get('compound','')}\t\t{row.get('doi','')}\tunresolved_identity_match\t0\n")

    unres = gen[gen["gen_smiles"].isna()]
    for idx, row in unres.iterrows():
        no_lines.append(f"{idx}\t{row.get('compound','')}\t\t{row.get('doi','')}\tcompound_not_resolved\t0\n")

    return comp_records, diff, missed, extra


def format_debug_value(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float) and np.isfinite(v):
        return f"{v:.6g}"
    return str(v)


def write_worst_cases(comp_df: pd.DataFrame, props: list[str], out_dir: Path, top_n: int = WORST_ROWS_PER_PROPERTY) -> dict[str, pd.DataFrame]:
    worst_tables: dict[str, pd.DataFrame] = {}
    summary_lines: list[str] = []

    for prop in props:
        err_col = f"{prop}_err"
        if err_col not in comp_df.columns:
            continue

        subset = comp_df.loc[comp_df[err_col].notna()].copy()
        if subset.empty:
            continue

        keep_cols = [
            "_matching_mode", "_gen_index", "_ref_index", "_doi", "_compound", "_smiles_gen",
            "_smiles_ref", "_solvent", "_paper", "_match_cost", "_resolved_via",
            f"{prop}_gen_raw", f"{prop}_gen_flag", f"{prop}_gen", f"{prop}_ref", f"{prop}_err",
        ]
        keep_cols = [c for c in keep_cols if c in subset.columns]
        worst = subset.sort_values(err_col, ascending=False).head(top_n)[keep_cols].copy()
        worst = worst.rename(columns={
            f"{prop}_gen_raw": "generated_raw",
            f"{prop}_gen_flag": "generated_flag",
            f"{prop}_gen": "generated_clean",
            f"{prop}_ref": "reference",
            f"{prop}_err": "abs_pct_error",
            "_matching_mode": "matching_mode",
            "_gen_index": "gen_index",
            "_ref_index": "ref_index",
            "_doi": "doi",
            "_compound": "compound",
            "_smiles_gen": "gen_smiles",
            "_smiles_ref": "ref_smiles",
            "_solvent": "solvent",
            "_paper": "paper",
            "_match_cost": "match_cost",
            "_resolved_via": "resolved_via",
        })

        out_csv = out_dir / f"{safe_filename(prop)}_worst_cases.csv"
        worst.to_csv(out_csv, index=False)
        worst_tables[prop] = worst

        summary_lines.append(f"===== WORST CASES: {prop} =====")
        for _, row in worst.iterrows():
            summary_lines.append(
                " | ".join([
                    f"err={format_debug_value(row.get('abs_pct_error'))}%",
                    f"gen_clean={format_debug_value(row.get('generated_clean'))}",
                    f"ref={format_debug_value(row.get('reference'))}",
                    f"raw={format_debug_value(row.get('generated_raw'))}",
                    f"flag={format_debug_value(row.get('generated_flag'))}",
                    f"doi={format_debug_value(row.get('doi'))}",
                    f"compound={format_debug_value(row.get('compound'))}",
                    f"gen_smiles={format_debug_value(row.get('gen_smiles'))}",
                    f"ref_smiles={format_debug_value(row.get('ref_smiles'))}",
                    f"resolved_via={format_debug_value(row.get('resolved_via'))}",
                    f"paper={format_debug_value(row.get('paper'))}",
                ])
            )
        summary_lines.append("")

    (out_dir / "worst_cases_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return worst_tables


def evaluate(original_csv: Path, generated_csv: Path, out_dir: Path, allow_doi_fallback: bool = False, refresh_name_cache: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    load_caches(out_dir, refresh_name_cache=refresh_name_cache)

    res_path = out_dir / "final_results.txt"
    no_path = out_dir / "no_match_entries.txt"
    multi_path = out_dir / "multi_match_entries.txt"
    invalid_path = out_dir / "invalid_smiles.txt"

    na_vals = ["null", "NULL", "", "NaN", "nan"]
    ref_raw = pd.read_csv(original_csv, sep=None, engine="python", na_values=na_vals, keep_default_na=True)
    gen_raw = pd.read_csv(generated_csv, sep=None, engine="python", na_values=na_vals, keep_default_na=True)

    ref = normalize_reference(ref_raw)
    gen, matching_mode = normalize_generated(gen_raw)
    
    if matching_mode == "resolved_generated_to_reference_smiles":
        bad = gen[gen["doi"].isna()][["paper_code", "compound"]].copy()
        if "doi" in gen_raw.columns:
            bad["raw_doi_column"] = gen_raw.loc[bad.index, "doi"]
        print("\n[DEBUG] Rows with missing parsed DOI:")
        print(bad.head(20).to_string(index=True))
    
    props = [p for p in PROPS if p in ref.columns]

    hdr = "idx\tSMILES_or_compound\tSolvent\tGen_DOI\tReason\t#ref_matches\n"
    no_lines, multi_lines, invalid_lines = [], [], []

    if matching_mode == "resolved_generated_to_reference_smiles":
        gen_cov = gen[gen["gen_smiles"].notna()].copy()
    else:
        gen_cov = gen[gen["smiles_gen"].astype(str).str.lower().ne("failed")].copy()

    cov = (
        pd.DataFrame({
            "gen": gen_cov.groupby("doi", dropna=True).size(),
            "ref": ref.groupby("doi", dropna=True).size(),
        })
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
        comp_records, diff, missed, extra = evaluate_resolved_generated_rows(
            ref, gen, props, no_lines, multi_lines, allow_doi_fallback
        )

    comp_df = pd.DataFrame(comp_records)
    worst_tables = write_worst_cases(comp_df, props, out_dir) if not comp_df.empty else {}

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
        f"DOI numeric fallback enabled: {allow_doi_fallback}",
        f"Refresh name cache: {refresh_name_cache}",
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

    if not comp_df.empty:
        report += ["", f"===== WORST MATCHED VALUES (top {WORST_ROWS_PER_PROPERTY} per property) ====="]
        for col in props:
            worst = worst_tables.get(col)
            if worst is None or worst.empty:
                continue
            report.append(f"{col}:")
            for _, row in worst.head(WORST_ROWS_PER_PROPERTY).iterrows():
                report.append(
                    "  "
                    + f"err={format_debug_value(row.get('abs_pct_error'))}% | "
                    + f"gen_clean={format_debug_value(row.get('generated_clean'))} | "
                    + f"ref={format_debug_value(row.get('reference'))} | "
                    + f"raw={format_debug_value(row.get('generated_raw'))} | "
                    + f"flag={format_debug_value(row.get('generated_flag'))} | "
                    + f"doi={format_debug_value(row.get('doi'))} | "
                    + f"compound={format_debug_value(row.get('compound'))} | "
                    + f"gen_smiles={format_debug_value(row.get('gen_smiles'))} | "
                    + f"ref_smiles={format_debug_value(row.get('ref_smiles'))} | "
                    + f"resolved_via={format_debug_value(row.get('resolved_via'))}"
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
        "Worst-case summary    → worst_cases_summary.txt",
    ]

    res_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    save_caches(out_dir)
    save_raw_results(out_dir, comp_df, cov, props)
    return comp_df, cov, props


def figures_only(out_dir: Path) -> None:
    comp_df, _, props = load_raw_results(out_dir)
    for col in props:
        if f"{col}_gen" in comp_df.columns:
            make_scatter(comp_df, col, out_dir)
            make_hist(comp_df, col, out_dir)
    make_error_boxplot(comp_df, props, out_dir)
    make_publication_figure(comp_df, props, out_dir, qy_col="Quantum yield", fname="publication_figure.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a generated photophysical dataset. "
            "For the new extraction CSV schema, only compounds in the dataset under test "
            "are resolved via RDKit/PubChem and matched to reference Chromophore SMILES."
        )
    )
    ap.add_argument("original_csv", nargs="?", help="Original reference CSV (ignored with --reuse)")
    ap.add_argument("generated_csv", nargs="?", help="Generated CSV to evaluate (ignored with --reuse)")
    ap.add_argument("-o", "--outdir", default="evaluation_results", help="Output directory")
    ap.add_argument("--reuse", action="store_true", help="Regenerate figures from saved raw results only.")
    ap.add_argument(
        "--allow-doi-fallback",
        action="store_true",
        help="Allow DOI+numeric fallback matching for generated rows whose resolved SMILES could not be matched.",
    )
    ap.add_argument(
        "--refresh-name-cache",
        action="store_true",
        help="Ignore previously cached compound-name resolution results and resolve names again.",
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
        evaluate(
            Path(args.original_csv),
            Path(args.generated_csv),
            out_dir,
            allow_doi_fallback=args.allow_doi_fallback,
            refresh_name_cache=args.refresh_name_cache,
        )
