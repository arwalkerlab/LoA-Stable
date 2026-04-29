#!/usr/bin/env python3
"""
Usage:
    python match_polymers_openchemie_compare.py \
        --reference "BCPs_sampled_100_bigsmiles_threshold50_combined.csv" \
        --extracted "polymer.csv" \
        --outdir match_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAME_MAP = {
    "polystyrene": "ps",
    "styrene": "ps",
    "ps": "ps",
    "polyisoprene": "pi",
    "isoprene": "pi",
    "pi": "pi",
    "polybutadiene": "pb",
    "butadiene": "pb",
    "pb": "pb",
    "polyethylene oxide": "peo",
    "poly(ethylene oxide)": "peo",
    "polyethylene glycol": "peo",
    "poly(ethylene glycol)": "peo",
    "peg": "peo",
    "peo": "peo",
    "poly(dimethylsiloxane)": "pdms",
    "dimethylsiloxane": "pdms",
    "polydimethylsiloxane": "pdms",
    "pdms": "pdms",
    "poly(2-vinylpyridine)": "p2vp",
    "poly2vinylpyridine": "p2vp",
    "p2vp": "p2vp",
    "poly(4-vinylpyridine)": "p4vp",
    "poly4vinylpyridine": "p4vp",
    "p4vp": "p4vp",
    "poly(ethylene-alt-propylene)": "pep",
    "poly(ethylenepropylene)": "pep",
    "pep": "pep",
    "polyethylethylene": "pee",
    "poly(ethylethylene)": "pee",
    "pee": "pee",
    "poly(lactic acid)": "pla",
    "polylacticacid": "pla",
    "pla": "pla",
    "poly(ethylene)": "pe",
    "pe": "pe",
    "poly(vinylcyclohexane)": "pvch",
    "pvch": "pvch",
}

PHASE_MAP = {
    "lamellar": "lamellar",
    "lamellae": "lamellar",
    "lamella": "lamellar",
    "l": "lamellar",
    "cylinder": "cylinder",
    "cylinders": "cylinder",
    "hexagonally packed cylinders": "cylinder",
    "hexagonal cylinders": "cylinder",
    "hex cyl": "cylinder",
    "c": "cylinder",
    "sphere": "sphere",
    "spheres": "sphere",
    "bcc sphere": "sphere",
    "s": "sphere",
    "gyroid": "gyroid",
    "double gyroid": "gyroid",
    "g": "gyroid",
    "disordered": "disordered",
    "disorder": "disordered",
    "disordered melt": "disordered",
    "dis": "disordered",
    "other": "other",
}

NUMERIC_FIELDS = [
    "Mn", "Mw", "D", "N", "T", "T_meas", "T_alt", "f1", "f_tot1", "w1", "rho1",
    "f2", "f_tot2", "w2", "rho2",
]

DEFAULT_RELAXED_NUMERIC_RULES = {
    "Mn": {"kind": "relative", "threshold": 0.10},
    "comp1_value": {"kind": "absolute", "threshold": 0.05},
}

DEFAULT_STRICT_NUMERIC_RULES = {
    "Mn": {"kind": "relative", "threshold": 0.05},
    "comp1_value": {"kind": "absolute", "threshold": 0.02},
}


def safe_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "failed"}:
        return np.nan
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\(\)\/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(s):
    s = clean_text(s)
    s_no_space = s.replace(" ", "")
    if s in NAME_MAP:
        return NAME_MAP[s]
    if s_no_space in NAME_MAP:
        return NAME_MAP[s_no_space]
    return s_no_space


def normalize_phase(s):
    s = clean_text(s)
    s_no_space = s.replace(" ", "")
    if s in PHASE_MAP:
        return PHASE_MAP[s]
    if s_no_space in PHASE_MAP:
        return PHASE_MAP[s_no_space]
    return s


def composition_value(row):
    for col in ["f1", "f_tot1", "w1"]:
        if col in row.index:
            v = safe_float(row[col])
            if not np.isnan(v):
                return v, col
    return np.nan, None


def text_similarity(a, b):
    a = clean_text(a)
    b = clean_text(b)
    if not a or not b:
        return np.nan
    return SequenceMatcher(None, a, b).ratio()


def prepare(df, kind):
    df = df.copy().reset_index(drop=True)

    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    for col in ["name1", "name2", "phase1", "phase2", "phase_method"]:
        if col not in df.columns:
            df[col] = np.nan

    df["name1_norm"] = df["name1"].apply(normalize_name)
    df["name2_norm"] = df["name2"].apply(normalize_name)
    df["phase1_norm"] = df["phase1"].apply(normalize_phase)
    df["phase2_norm"] = df["phase2"].apply(normalize_phase)
    df["pair_key"] = df.apply(lambda r: "|".join(sorted([r["name1_norm"], r["name2_norm"]])), axis=1)

    comp_vals = []
    comp_cols = []
    for _, row in df.iterrows():
        v, c = composition_value(row)
        comp_vals.append(v)
        comp_cols.append(c)
    df["comp1_value"] = comp_vals
    df["comp1_source_col"] = comp_cols
    df["source_kind"] = kind
    return df


def candidate_pool(ext_row, ref_df):
    pair = ext_row["pair_key"]
    candidates = ref_df[ref_df["pair_key"] == pair].copy()

    if len(candidates) == 0:
        n1 = ext_row["name1_norm"]
        n2 = ext_row["name2_norm"]
        candidates = ref_df[
            ((ref_df["name1_norm"] == n1) & (ref_df["name2_norm"] == n2))
            | ((ref_df["name1_norm"] == n2) & (ref_df["name2_norm"] == n1))
        ].copy()

    if len(candidates) == 0:
        n1 = ext_row["name1_norm"]
        n2 = ext_row["name2_norm"]
        candidates = ref_df[
            ref_df["name1_norm"].isin([n1, n2]) | ref_df["name2_norm"].isin([n1, n2])
        ].copy()

    return candidates


def row_match_score(ext_row, ref_row):
    score = 0.0

    ext_comp = ext_row["comp1_value"]
    ref_comp = ref_row["comp1_value"]
    if not np.isnan(ext_comp) and not np.isnan(ref_comp):
        score += abs(ext_comp - ref_comp) * 100.0
    else:
        score += 20.0

    ext_mn = safe_float(ext_row.get("Mn"))
    ref_mn = safe_float(ref_row.get("Mn"))
    if not np.isnan(ext_mn) and not np.isnan(ref_mn):
        score += min(abs(ext_mn - ref_mn) / max(abs(ref_mn), 1e-12), 5.0) * 10.0
    else:
        score += 5.0

    ext_phase = ext_row.get("phase1_norm", "")
    ref_phase = ref_row.get("phase1_norm", "")
    if ext_phase and ref_phase and ext_phase != ref_phase:
        score += 1.0

    s1 = text_similarity(ext_row.get("name1", ""), ref_row.get("name1", ""))
    s2 = text_similarity(ext_row.get("name2", ""), ref_row.get("name2", ""))
    if not np.isnan(s1):
        score += 1.0 - s1
    if not np.isnan(s2):
        score += 1.0 - s2

    return score


def match_rows(ext_df, ref_df):
    matches = []
    used_ref = set()

    for ext_idx, ext_row in ext_df.iterrows():
        cand = candidate_pool(ext_row, ref_df)
        if len(cand) == 0:
            matches.append({
                "ext_index": ext_idx,
                "ref_index": np.nan,
                "match_found": False,
                "candidate_count": 0,
                "match_score": np.nan,
            })
            continue

        cand_unused = cand.loc[~cand.index.isin(used_ref)]
        if len(cand_unused) > 0:
            cand = cand_unused

        scores = cand.apply(lambda r: row_match_score(ext_row, r), axis=1)
        best_ref_idx = scores.idxmin()
        best_score = float(scores.loc[best_ref_idx])

        used_ref.add(best_ref_idx)
        matches.append({
            "ext_index": ext_idx,
            "ref_index": int(best_ref_idx),
            "match_found": True,
            "candidate_count": int(len(cand)),
            "match_score": best_score,
        })

    return pd.DataFrame(matches)


def numeric_stats(df, pred_col, true_col, tolerances=(0.05, 0.10, 0.20)):
    sub = df[[pred_col, true_col]].copy()
    sub[pred_col] = sub[pred_col].apply(safe_float)
    sub[true_col] = sub[true_col].apply(safe_float)
    sub = sub.dropna()
    if len(sub) == 0:
        return None

    ae = (sub[pred_col] - sub[true_col]).abs()
    re = ae / sub[true_col].abs().clip(lower=1e-12)

    out = {
        "n_compared": int(len(sub)),
        "mae": float(ae.mean()),
        "median_abs_error": float(ae.median()),
        "mean_rel_error": float(re.mean()),
        "median_rel_error": float(re.median()),
    }
    for tol in tolerances:
        out[f"pct_within_{int(tol * 100)}pct"] = float((re <= tol).mean())
    return out


def categorical_stats(df, pred_col, true_col, normalize_fn=None):
    sub = df[[pred_col, true_col]].copy().dropna()
    if len(sub) == 0:
        return None
    if normalize_fn:
        sub[pred_col] = sub[pred_col].apply(normalize_fn)
        sub[true_col] = sub[true_col].apply(normalize_fn)
    return {
        "n_compared": int(len(sub)),
        "accuracy": float((sub[pred_col] == sub[true_col]).mean()),
    }


def build_eval_table(merged):
    stats = {}

    numeric_pairs = [(f"ext_{field}", f"ref_{field}") for field in NUMERIC_FIELDS]
    categorical_pairs = [
        ("ext_phase1", "ref_phase1", normalize_phase),
        ("ext_phase2", "ref_phase2", normalize_phase),
        ("ext_phase_method", "ref_phase_method", clean_text),
        ("ext_name1", "ref_name1", normalize_name),
        ("ext_name2", "ref_name2", normalize_name),
    ]

    for pred_col, true_col in numeric_pairs:
        if pred_col in merged.columns and true_col in merged.columns:
            res = numeric_stats(merged, pred_col, true_col)
            if res:
                stats[pred_col.replace("ext_", "")] = res

    for pred_col, true_col, norm_fn in categorical_pairs:
        if pred_col in merged.columns and true_col in merged.columns:
            res = categorical_stats(merged, pred_col, true_col, normalize_fn=norm_fn)
            if res:
                stats[pred_col.replace("ext_", "")] = res

    return stats


def add_row_quality_columns(
    merged,
    comp_abs_tol=0.05,
    mn_rel_tol=0.10,
    strict_comp_abs_tol=0.02,
    strict_mn_rel_tol=0.05,
):
    merged = merged.copy()

    ext_name_pair = merged.apply(
        lambda r: tuple(sorted([normalize_name(r.get("ext_name1", "")), normalize_name(r.get("ext_name2", ""))])),
        axis=1,
    )
    ref_name_pair = merged.apply(
        lambda r: tuple(sorted([normalize_name(r.get("ref_name1", "")), normalize_name(r.get("ref_name2", ""))])),
        axis=1,
    )
    merged["name_pair_match"] = ext_name_pair == ref_name_pair

    for phase_col in ["phase1", "phase2"]:
        ext_col = f"ext_{phase_col}"
        ref_col = f"ref_{phase_col}"
        if ext_col in merged.columns and ref_col in merged.columns:
            ext_phase = merged[ext_col].apply(normalize_phase)
            ref_phase = merged[ref_col].apply(normalize_phase)
            present = (ext_phase != "") & (ref_phase != "")
            merged[f"{phase_col}_match_when_present"] = (~present) | (ext_phase == ref_phase)
        else:
            merged[f"{phase_col}_match_when_present"] = True

    ext_comp = merged.get("ext_comp1_value", pd.Series(np.nan, index=merged.index)).apply(safe_float)
    ref_comp = merged.get("ref_comp1_value", pd.Series(np.nan, index=merged.index)).apply(safe_float)
    comp_present = ~(ext_comp.isna() | ref_comp.isna())
    comp_abs_err = (ext_comp - ref_comp).abs()
    merged["comp_abs_error"] = comp_abs_err
    merged["composition_match_relaxed"] = (~comp_present) | (comp_abs_err <= comp_abs_tol)
    merged["composition_match_strict"] = (~comp_present) | (comp_abs_err <= strict_comp_abs_tol)

    ext_mn = merged.get("ext_Mn", pd.Series(np.nan, index=merged.index)).apply(safe_float)
    ref_mn = merged.get("ref_Mn", pd.Series(np.nan, index=merged.index)).apply(safe_float)
    mn_present = ~(ext_mn.isna() | ref_mn.isna())
    mn_rel_err = (ext_mn - ref_mn).abs() / ref_mn.abs().clip(lower=1e-12)
    merged["Mn_rel_error"] = mn_rel_err
    merged["mn_match_relaxed"] = (~mn_present) | (mn_rel_err <= mn_rel_tol)
    merged["mn_match_strict"] = (~mn_present) | (mn_rel_err <= strict_mn_rel_tol)

    merged["row_exact_names_phases"] = (
        merged["name_pair_match"]
        & merged["phase1_match_when_present"]
        & merged["phase2_match_when_present"]
    )
    merged["row_relaxed_correct"] = (
        merged["row_exact_names_phases"]
        & merged["composition_match_relaxed"]
        & merged["mn_match_relaxed"]
    )
    merged["row_strict_correct"] = (
        merged["row_exact_names_phases"]
        & merged["composition_match_strict"]
        & merged["mn_match_strict"]
    )

    outcome = []
    for _, row in merged.iterrows():
        if bool(row["row_strict_correct"]):
            outcome.append("strict correct")
        elif bool(row["row_relaxed_correct"]):
            outcome.append("relaxed correct")
        elif bool(row["name_pair_match"]):
            outcome.append("name pair only")
        else:
            outcome.append("matched but incorrect")
    merged["row_outcome"] = outcome
    return merged


def summarize_headline_metrics(merged, n_extracted, n_reference, openchemie_accuracy):
    matched_count = int(len(merged))
    strict_correct = int(merged["row_strict_correct"].sum()) if matched_count else 0
    relaxed_correct = int(merged["row_relaxed_correct"].sum()) if matched_count else 0
    names_phase_correct = int(merged["row_exact_names_phases"].sum()) if matched_count else 0

    summary = {
        "openchemie_reported_accuracy": float(openchemie_accuracy),
        "n_matched_rows": matched_count,
        "row_exact_names_phases_count": names_phase_correct,
        "row_relaxed_correct_count": relaxed_correct,
        "row_strict_correct_count": strict_correct,
        "row_exact_names_phases_accuracy_vs_extracted": float(names_phase_correct / n_extracted) if n_extracted else None,
        "row_exact_names_phases_accuracy_vs_reference": float(names_phase_correct / n_reference) if n_reference else None,
        "row_relaxed_accuracy_vs_extracted": float(relaxed_correct / n_extracted) if n_extracted else None,
        "row_relaxed_accuracy_vs_reference": float(relaxed_correct / n_reference) if n_reference else None,
        "row_strict_accuracy_vs_extracted": float(strict_correct / n_extracted) if n_extracted else None,
        "row_strict_accuracy_vs_reference": float(strict_correct / n_reference) if n_reference else None,
        "row_relaxed_precision": float(relaxed_correct / matched_count) if matched_count else None,
        "row_relaxed_recall": float(relaxed_correct / n_reference) if n_reference else None,
        "row_relaxed_f1": None,
        "row_strict_precision": float(strict_correct / matched_count) if matched_count else None,
        "row_strict_recall": float(strict_correct / n_reference) if n_reference else None,
        "row_strict_f1": None,
        "delta_vs_openchemie_relaxed_extracted": None,
        "delta_vs_openchemie_relaxed_reference": None,
        "delta_vs_openchemie_strict_extracted": None,
        "delta_vs_openchemie_exact_names_phases_extracted": None,
    }

    if summary["row_relaxed_precision"] is not None and summary["row_relaxed_recall"] is not None:
        p = summary["row_relaxed_precision"]
        r = summary["row_relaxed_recall"]
        summary["row_relaxed_f1"] = float(0.0 if p + r == 0 else 2 * p * r / (p + r))

    if summary["row_strict_precision"] is not None and summary["row_strict_recall"] is not None:
        p = summary["row_strict_precision"]
        r = summary["row_strict_recall"]
        summary["row_strict_f1"] = float(0.0 if p + r == 0 else 2 * p * r / (p + r))

    for key in [
        "row_relaxed_accuracy_vs_extracted",
        "row_relaxed_accuracy_vs_reference",
        "row_strict_accuracy_vs_extracted",
        "row_exact_names_phases_accuracy_vs_extracted",
    ]:
        if summary[key] is not None:
            delta = summary[key] - openchemie_accuracy
            if key == "row_relaxed_accuracy_vs_extracted":
                summary["delta_vs_openchemie_relaxed_extracted"] = float(delta)
            elif key == "row_relaxed_accuracy_vs_reference":
                summary["delta_vs_openchemie_relaxed_reference"] = float(delta)
            elif key == "row_strict_accuracy_vs_extracted":
                summary["delta_vs_openchemie_strict_extracted"] = float(delta)
            elif key == "row_exact_names_phases_accuracy_vs_extracted":
                summary["delta_vs_openchemie_exact_names_phases_extracted"] = float(delta)

    return summary


def make_figures_dir(outdir):
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def save_current_figure(path, dpi=200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def paired_numeric_series(merged, pred_col, true_col):
    if pred_col not in merged.columns or true_col not in merged.columns:
        return None
    sub = merged[[pred_col, true_col]].copy()
    sub[pred_col] = sub[pred_col].apply(safe_float)
    sub[true_col] = sub[true_col].apply(safe_float)
    sub = sub.dropna()
    if sub.empty:
        return None
    return sub


def maybe_log_axes(x, y):
    positive_x = (x > 0).all()
    positive_y = (y > 0).all()
    if positive_x and positive_y:
        span_x = x.max() / max(x.min(), 1e-12)
        span_y = y.max() / max(y.min(), 1e-12)
        return span_x >= 100 or span_y >= 100
    return False


def plot_match_score_histogram(match_df, fig_dir, fig_format, dpi):
    sub = match_df["match_score"].dropna()
    if sub.empty:
        return None
    plt.figure(figsize=(7, 4.5))
    plt.hist(sub, bins=min(30, max(8, int(math.sqrt(len(sub)) * 2))))
    plt.xlabel("Match score")
    plt.ylabel("Count")
    plt.title("Distribution of match scores")
    out = os.path.join(fig_dir, f"match_score_histogram.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_candidate_count_histogram(match_df, fig_dir, fig_format, dpi):
    if "candidate_count" not in match_df.columns:
        return None
    sub = match_df["candidate_count"].dropna()
    if sub.empty:
        return None
    plt.figure(figsize=(7, 4.5))
    bins = np.arange(sub.min(), sub.max() + 2) - 0.5
    plt.hist(sub, bins=bins)
    plt.xlabel("Candidate count")
    plt.ylabel("Rows")
    plt.title("Candidate pool size per extracted row")
    out = os.path.join(fig_dir, f"candidate_count_histogram.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_numeric_comparison(merged, field, fig_dir, fig_format, dpi):
    pred_col = f"ext_{field}"
    true_col = f"ref_{field}"
    sub = paired_numeric_series(merged, pred_col, true_col)
    if sub is None:
        return None

    x = sub[true_col].to_numpy()
    y = sub[pred_col].to_numpy()
    stats = numeric_stats(sub, pred_col, true_col)

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(x, y, alpha=0.75)

    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    if np.isfinite(lo) and np.isfinite(hi):
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)

    if maybe_log_axes(pd.Series(x), pd.Series(y)):
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(f"Reference {field}")
    plt.ylabel(f"Extracted {field}")
    plt.title(f"Extracted vs. reference: {field}")

    if stats is not None:
        note = (
            f"n={stats['n_compared']}\n"
            f"MAE={stats['mae']:.4g}\n"
            f"Median rel. err.={stats['median_rel_error']:.3f}"
        )
        plt.annotate(
            note,
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    out = os.path.join(fig_dir, f"compare_{field}.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_error_histogram(merged, field, fig_dir, fig_format, dpi):
    pred_col = f"ext_{field}"
    true_col = f"ref_{field}"
    sub = paired_numeric_series(merged, pred_col, true_col)
    if sub is None:
        return None

    rel_error = (sub[pred_col] - sub[true_col]).abs() / sub[true_col].abs().clip(lower=1e-12)
    plt.figure(figsize=(7, 4.5))
    plt.hist(rel_error, bins=min(30, max(8, int(math.sqrt(len(rel_error)) * 2))))
    plt.xlabel(f"Relative error for {field}")
    plt.ylabel("Count")
    plt.title(f"Relative error distribution: {field}")
    out = os.path.join(fig_dir, f"relative_error_{field}.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_categorical_accuracy(column_stats, fig_dir, fig_format, dpi):
    cat_fields = []
    accuracies = []
    for field, stats in column_stats.items():
        if isinstance(stats, dict) and "accuracy" in stats:
            cat_fields.append(field)
            accuracies.append(stats["accuracy"])
    if not cat_fields:
        return None

    plt.figure(figsize=(8, 4.5))
    plt.bar(cat_fields, accuracies)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Categorical field accuracy")
    plt.xticks(rotation=30, ha="right")
    out = os.path.join(fig_dir, f"categorical_accuracy.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_benchmark_comparison(headline, fig_dir, fig_format, dpi):
    labels = [
        "OpenChemIE reported",
        "Exact names+phases\n(vs extracted)",
        "Relaxed row\n(vs extracted)",
        "Relaxed row\n(vs reference)",
        "Strict row\n(vs extracted)",
    ]
    values = [
        headline["openchemie_reported_accuracy"],
        headline["row_exact_names_phases_accuracy_vs_extracted"],
        headline["row_relaxed_accuracy_vs_extracted"],
        headline["row_relaxed_accuracy_vs_reference"],
        headline["row_strict_accuracy_vs_extracted"],
    ]
    clean_labels = []
    clean_values = []
    for label, value in zip(labels, values):
        if value is not None:
            clean_labels.append(label)
            clean_values.append(value)
    if not clean_values:
        return None

    plt.figure(figsize=(8.5, 4.8))
    positions = np.arange(len(clean_labels))
    plt.bar(positions, clean_values)
    plt.axhline(headline["openchemie_reported_accuracy"], linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Headline score comparison vs. OpenChemIE benchmark")
    plt.xticks(positions, clean_labels, rotation=20, ha="right")
    out = os.path.join(fig_dir, f"benchmark_comparison.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_precision_recall_f1(headline, fig_dir, fig_format, dpi):
    values = [
        headline.get("row_relaxed_precision"),
        headline.get("row_relaxed_recall"),
        headline.get("row_relaxed_f1"),
        headline.get("row_strict_precision"),
        headline.get("row_strict_recall"),
        headline.get("row_strict_f1"),
    ]
    labels = [
        "Relaxed precision", "Relaxed recall", "Relaxed F1",
        "Strict precision", "Strict recall", "Strict F1",
    ]
    clean_labels = []
    clean_values = []
    for label, value in zip(labels, values):
        if value is not None:
            clean_labels.append(label)
            clean_values.append(value)
    if not clean_values:
        return None

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(clean_labels, clean_values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Row-level precision / recall / F1")
    plt.xticks(rotation=20, ha="right")
    out = os.path.join(fig_dir, f"row_prf.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def plot_row_outcomes(merged, fig_dir, fig_format, dpi):
    if "row_outcome" not in merged.columns or merged.empty:
        return None
    counts = merged["row_outcome"].value_counts()
    if counts.empty:
        return None
    plt.figure(figsize=(7.5, 4.6))
    plt.bar(counts.index.tolist(), counts.values.tolist())
    plt.ylabel("Rows")
    plt.title("Matched row outcomes")
    plt.xticks(rotation=20, ha="right")
    out = os.path.join(fig_dir, f"row_outcomes.{fig_format}")
    save_current_figure(out, dpi=dpi)
    return out


def export_figures(match_df, merged, column_stats, headline_metrics, outdir, fig_format="png", dpi=200):
    fig_dir = make_figures_dir(outdir)
    written = []

    for fn in [plot_match_score_histogram, plot_candidate_count_histogram]:
        path = fn(match_df, fig_dir, fig_format, dpi)
        if path:
            written.append(path)

    for field in ["Mn", "Mw", "D", "N", "T", "f1", "f_tot1", "w1", "rho1", "f2", "f_tot2", "w2", "rho2"]:
        compare_path = plot_numeric_comparison(merged, field, fig_dir, fig_format, dpi)
        if compare_path:
            written.append(compare_path)
        err_path = plot_error_histogram(merged, field, fig_dir, fig_format, dpi)
        if err_path:
            written.append(err_path)

    for path in [
        plot_categorical_accuracy(column_stats, fig_dir, fig_format, dpi),
        plot_benchmark_comparison(headline_metrics, fig_dir, fig_format, dpi),
        plot_precision_recall_f1(headline_metrics, fig_dir, fig_format, dpi),
        plot_row_outcomes(merged, fig_dir, fig_format, dpi),
    ]:
        if path:
            written.append(path)

    manifest = {
        "figure_format": fig_format,
        "dpi": dpi,
        "files": [os.path.relpath(path, outdir) for path in written],
    }
    with open(os.path.join(outdir, "figures_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="reference CSV path")
    parser.add_argument("--extracted", required=True, help="extracted CSV path")
    parser.add_argument("--outdir", default="match_results", help="output directory")
    parser.add_argument("--fig-format", default="png", choices=["png", "pdf", "svg"], help="format for exported figures")
    parser.add_argument("--fig-dpi", type=int, default=200, help="figure DPI for raster exports")
    parser.add_argument("--skip-figures", action="store_true", help="disable figure export")
    parser.add_argument("--openchemie-accuracy", type=float, default=0.643, help="benchmark accuracy to compare against, as a fraction")
    parser.add_argument("--comp-abs-tol", type=float, default=0.05, help="absolute tolerance for relaxed composition correctness")
    parser.add_argument("--mn-rel-tol", type=float, default=0.10, help="relative tolerance for relaxed Mn correctness")
    parser.add_argument("--strict-comp-abs-tol", type=float, default=0.02, help="absolute tolerance for strict composition correctness")
    parser.add_argument("--strict-mn-rel-tol", type=float, default=0.05, help="relative tolerance for strict Mn correctness")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ref = pd.read_csv(args.reference)
    ext = pd.read_csv(args.extracted)

    ref_p = prepare(ref, "reference")
    ext_p = prepare(ext, "extracted")

    match_df = match_rows(ext_p, ref_p)
    matched = match_df[match_df["match_found"]].copy()

    ext_join = ext_p.copy()
    ext_join["ext_index"] = ext_join.index
    ext_join = ext_join.add_prefix("ext_")
    ext_join = ext_join.rename(columns={"ext_ext_index": "ext_index"})

    ref_join = ref_p.copy()
    ref_join["ref_index"] = ref_join.index
    ref_join = ref_join.add_prefix("ref_")
    ref_join = ref_join.rename(columns={"ref_ref_index": "ref_index"})

    merged = matched.merge(ext_join, on="ext_index", how="left").merge(ref_join, on="ref_index", how="left")
    merged = add_row_quality_columns(
        merged,
        comp_abs_tol=args.comp_abs_tol,
        mn_rel_tol=args.mn_rel_tol,
        strict_comp_abs_tol=args.strict_comp_abs_tol,
        strict_mn_rel_tol=args.strict_mn_rel_tol,
    )

    summary = {
        "n_reference_rows": int(len(ref_p)),
        "n_extracted_rows": int(len(ext_p)),
        "n_matched_rows": int(merged["ref_index"].notna().sum()),
        "match_rate_vs_extracted": float(merged["ref_index"].notna().mean()) if len(ext_p) else 0.0,
        "mean_match_score": float(merged["match_score"].mean()) if len(merged) else None,
        "median_match_score": float(merged["match_score"].median()) if len(merged) else None,
    }

    if len(merged):
        comp_abs = []
        mn_rel = []

        for _, r in merged.iterrows():
            ext_comp = safe_float(r.get("ext_comp1_value"))
            ref_comp = safe_float(r.get("ref_comp1_value"))
            if not np.isnan(ext_comp) and not np.isnan(ref_comp):
                comp_abs.append(abs(ext_comp - ref_comp))

            ext_mn = safe_float(r.get("ext_Mn"))
            ref_mn = safe_float(r.get("ref_Mn"))
            if not np.isnan(ext_mn) and not np.isnan(ref_mn):
                mn_rel.append(abs(ext_mn - ref_mn) / max(abs(ref_mn), 1e-12))

        summary["composition_pairs_with_data"] = int(len(comp_abs))
        summary["composition_abs_error_mean"] = float(np.mean(comp_abs)) if comp_abs else None
        summary["composition_abs_error_median"] = float(np.median(comp_abs)) if comp_abs else None
        summary["mn_pairs_with_data"] = int(len(mn_rel))
        summary["mn_rel_error_mean"] = float(np.mean(mn_rel)) if mn_rel else None
        summary["mn_rel_error_median"] = float(np.median(mn_rel)) if mn_rel else None

    column_stats = build_eval_table(merged)
    headline_metrics = summarize_headline_metrics(
        merged=merged,
        n_extracted=len(ext_p),
        n_reference=len(ref_p),
        openchemie_accuracy=args.openchemie_accuracy,
    )

    config = {
        "openchemie_accuracy": args.openchemie_accuracy,
        "relaxed_rules": {
            "name_pair": "unordered normalized exact match",
            "phase1": "must match when both are present",
            "phase2": "must match when both are present",
            "composition_abs_tolerance": args.comp_abs_tol,
            "Mn_relative_tolerance": args.mn_rel_tol,
        },
        "strict_rules": {
            "name_pair": "unordered normalized exact match",
            "phase1": "must match when both are present",
            "phase2": "must match when both are present",
            "composition_abs_tolerance": args.strict_comp_abs_tol,
            "Mn_relative_tolerance": args.strict_mn_rel_tol,
        },
    }

    match_df.to_csv(os.path.join(args.outdir, "matches_only.csv"), index=False)
    merged.to_csv(os.path.join(args.outdir, "matched_merged.csv"), index=False)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.outdir, "column_stats.json"), "w") as f:
        json.dump(column_stats, f, indent=2)

    with open(os.path.join(args.outdir, "headline_metrics.json"), "w") as f:
        json.dump(headline_metrics, f, indent=2)

    with open(os.path.join(args.outdir, "benchmark_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    figure_paths = []
    if not args.skip_figures:
        figure_paths = export_figures(
            match_df=match_df,
            merged=merged,
            column_stats=column_stats,
            headline_metrics=headline_metrics,
            outdir=args.outdir,
            fig_format=args.fig_format,
            dpi=args.fig_dpi,
        )

    print("\n=== Matching summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Headline metrics for rough benchmark comparison ===")
    for k, v in headline_metrics.items():
        print(f"{k}: {v}")

    print("\n=== Column stats ===")
    if not column_stats:
        print("No overlapping comparable columns found after matching.")
    else:
        for col, stats in column_stats.items():
            print(f"\n[{col}]")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    print(f"\nWrote outputs to: {args.outdir}")
    print("  - matches_only.csv")
    print("  - matched_merged.csv")
    print("  - summary.json")
    print("  - column_stats.json")
    print("  - headline_metrics.json")
    print("  - benchmark_config.json")
    if not args.skip_figures:
        print("  - figures_manifest.json")
        print(f"  - figures/ ({len(figure_paths)} files)")


if __name__ == "__main__":
    main()
