"""Validate the saved resolution series and visualize all latent/observed activity maps.

Read-only with respect to simulation inputs. No simulation or REGAL fit is run.
Diagnostic PNGs and their source tables are written to simulation/outputs/resolution/simulation_check by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.dataset_registry import DatasetSpec  # noqa: E402
from utils.h5ad_schema import check_rss_h5ad_schema  # noqa: E402

if __package__:
    from .generate_region_layout import ACTIVITY_COLORS
    from .generate_resolution_simulation import DEFAULT_CONFIG, OUTPUT_ROOT, sha256
else:
    from generate_region_layout import ACTIVITY_COLORS
    from generate_resolution_simulation import DEFAULT_CONFIG, OUTPUT_ROOT, sha256

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
}
TRUTH_KEYS = (
    "spot_ids", "gene_ids", "activity_ids", "instance_ids", "instance_activity_ids", "weights", "specific",
    "activation", "activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction",
)


def require(checks: list[dict], name: str, passed: bool, detail="") -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def read_sample(data_root: Path, prefix: str, nside: int, checks: list[dict]) -> dict:
    sample_id = f"{prefix}_{nside}"
    folder = data_root / sample_id / "simulation"
    manifest = json.loads((folder / "manifest.json").read_text())
    hashes = {key: sha256(folder / key) == value for key, value in manifest["files"].items()}
    require(checks, f"{nside}:manifest_files", all(hashes.values()), hashes)
    h5ad = data_root / f"{sample_id}.h5ad"
    spec = DatasetSpec("external_dataset", "SIMULATION", (sample_id,), species="synthetic",
                       expression_state="raw_counts")
    schema = check_rss_h5ad_schema(h5ad, spec, sample_id)
    require(checks, f"{nside}:input_schema", schema.ok, schema.format_report())
    data = ad.read_h5ad(h5ad)
    counts = data.X.tocsr()
    with np.load(folder / "truth/arrays.npz") as archive:
        truth = {key: archive[key] for key in TRUTH_KEYS}
    require(checks, f"{nside}:integer_nonnegative_counts", counts.dtype.kind in "iu" and (counts.data >= 0).all())
    require(checks, f"{nside}:aligned_ids", counts.shape == (nside**2, len(truth["gene_ids"]))
            and np.array_equal(data.obs_names, truth["spot_ids"]) and np.array_equal(data.var_names, truth["gene_ids"]))
    axis = (np.arange(nside) + 0.5) * 128 / nside
    x, y = np.meshgrid(axis, axis)
    coords = np.column_stack((x.ravel(), y.ravel()))
    require(checks, f"{nside}:coordinates", np.array_equal(data.obsm["spatial"], coords))
    require(checks, f"{nside}:input_has_no_truth", set(data.obs) == {"sample_id"}
            and set(data.var) == {"gene_id", "gene_name"} and set(data.obsm) == {"spatial"}
            and set(data.layers) == {"log1p_norm"} and not data.varm)
    normalized = data.layers["log1p_norm"].tocsr()
    totals_float = np.asarray(counts.astype(np.float32).sum(axis=1)).ravel()
    target = data.uns["normalization"]["target_sum"]
    expected = (sparse.diags(target / np.maximum(totals_float, 1)) @ counts.astype(np.float32)).tocsr()
    expected.data = np.log1p(expected.data)
    difference = normalized - expected
    require(checks, f"{nside}:normalization", np.isfinite(normalized.data).all()
            and (not difference.nnz or np.abs(difference.data).max() <= 2e-6))
    core = truth["specific"]
    core_scores = np.column_stack([
        np.asarray(normalized[:, core[:, k]].mean(axis=1)).ravel() for k in range(core.shape[1])
    ])
    totals = np.asarray(counts.sum(axis=1)).ravel()
    detected = np.asarray((counts > 0).sum(axis=1)).ravel()
    for key in ("activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction"):
        require(checks, f"{nside}:{key}_range", np.isfinite(truth[key]).all()
                and ((truth[key] >= 0) & (truth[key] <= 1)).all())
    require(checks, f"{nside}:activity_fraction_identity", np.array_equal(
        truth["activity_fraction"].sum(axis=1), 1 + truth["overlap_fraction"].sum(axis=1)))
    require(checks, f"{nside}:full_base_coverage", np.array_equal(
        truth["base_region_fraction"].sum(axis=1), np.ones(nside**2)))
    require(checks, f"{nside}:positive_activation_support", np.array_equal(
        truth["activation"] > 0, truth["activity_fraction"] > 0))
    require(checks, f"{nside}:nonempty_positions", (totals > 0).all())
    return {"sample_id": sample_id, "nside": nside, "folder": folder, "counts": counts, "truth": truth,
            "coords": coords, "totals": totals, "detected": detected, "core_scores": core_scores,
            "normalized": normalized,
            "manifest": manifest, "input_manifest_sha256": sha256(folder / "manifest.json")}


def high_to_coarse_operator(high: dict, coarse: dict, checks: list[dict]) -> sparse.csr_matrix:
    mapping = pd.read_csv(coarse["folder"] / "high_resolution_spot_mapping.csv")
    n_high, n_coarse = high["counts"].shape[0], coarse["counts"].shape[0]
    source_index = mapping.high_resolution_spot_index.to_numpy()
    target_index = mapping.spot_index.to_numpy()
    factor = high["nside"] // coarse["nside"]
    row, col = np.divmod(source_index, high["nside"])
    expected = row // factor * coarse["nside"] + col // factor
    require(checks, f"{coarse['nside']}:position_mapping", np.array_equal(source_index, np.arange(n_high))
            and np.array_equal(target_index, expected)
            and np.array_equal(mapping.high_resolution_spot_id, high["truth"]["spot_ids"])
            and np.array_equal(mapping.spot_id, coarse["truth"]["spot_ids"][target_index]))
    return sparse.csr_matrix(
        (np.ones(n_high, dtype=np.int64), (target_index, source_index)), shape=(n_coarse, n_high),
    )


def check_series(samples: list[dict], checks: list[dict]) -> None:
    high = samples[0]
    high_truth = high["truth"]
    require(checks, "series:11_activities_16_instances", len(high_truth["activity_ids"]) == 11
            and len(high_truth["instance_ids"]) == 16)
    for sample in samples:
        model_keys = ("gene_ids", "activity_ids", "instance_ids", "instance_activity_ids", "weights", "specific")
        require(checks, f"{sample['nside']}:same_gene_model", all(
            np.array_equal(sample["truth"][key], high_truth[key]) for key in model_keys))
        for k, activity in enumerate(high_truth["activity_ids"]):
            columns = high_truth["instance_activity_ids"] == activity
            require(checks, f"{sample['nside']}:{activity}_instance_union", np.array_equal(
                sample["truth"]["instance_fraction"][:, columns].sum(axis=1),
                sample["truth"]["activity_fraction"][:, k]))
    with np.load(high["folder"] / "truth/gene_activity_fields.npz") as fields_file:
        fields = fields_file["fields"]
        genes, activities = fields_file["gene_indices"], fields_file["activity_indices"]
        require(checks, "high:gene_activity_fields", fields.shape == (high["counts"].shape[0], len(genes))
                and len(genes) == len(activities) and np.isfinite(fields).all() and (fields >= 0).all()
                and (high_truth["weights"][genes, activities] > 0).all()
                and not np.any(fields[~high_truth["activity_fraction"][:, activities].astype(bool)]))
    del fields
    with np.load(high["folder"] / "truth/arrays.npz") as truth_file:
        expected_high = truth_file["expected_counts"]
    require(checks, "high:finite_positive_conditional_means", np.isfinite(expected_high).all()
            and (expected_high > 0).all())
    for coarse in samples[1:]:
        operator = high_to_coarse_operator(high, coarse, checks)
        factor = (high["nside"] // coarse["nside"])**2
        require(checks, f"{coarse['nside']}:exact_raw_count_sum",
                (operator @ high["counts"] != coarse["counts"]).nnz == 0)
        require(checks, f"{coarse['nside']}:per_gene_total_conservation", np.array_equal(
            np.asarray(high["counts"].sum(axis=0)), np.asarray(coarse["counts"].sum(axis=0))))
        for key in ("activation", "activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction"):
            expected = operator @ high_truth[key] / factor
            require(checks, f"{coarse['nside']}:{key}_aggregation", np.allclose(
                coarse["truth"][key], expected, rtol=2e-7, atol=1e-7))
        for key in ("activity_fraction", "instance_fraction", "overlap_fraction"):
            require(checks, f"{coarse['nside']}:{key}_physical_area", np.array_equal(
                coarse["truth"][key].sum(axis=0) * factor, high_truth[key].sum(axis=0)))
        with np.load(coarse["folder"] / "truth/arrays.npz") as truth_file:
            conditional = truth_file["expected_counts"]
        require(checks, f"{coarse['nside']}:conditional_expected_count_sum", np.allclose(
            conditional, operator @ expected_high, rtol=2e-7, atol=1e-4))


def write_tables(samples: list[dict], out: Path) -> pd.DataFrame:
    summaries, points, activities = [], [], []
    for sample in samples:
        side, truth = sample["nside"], sample["truth"]
        totals, detected = sample["totals"], sample["detected"]
        area = (128 / side)**2
        summaries.append({
            "nside": side, "sample_id": sample["sample_id"], "n_spots": len(totals),
            "n_genes": sample["counts"].shape[1], "total_counts": int(totals.sum()),
            "mean_counts": float(totals.mean()), "median_counts": float(np.median(totals)),
            "median_detected_genes": float(np.median(detected)),
            "nonzero_fraction": sample["counts"].nnz / np.prod(sample["counts"].shape),
        })
        point = pd.DataFrame({"nside": side, "spot_id": truth["spot_ids"], "x": sample["coords"][:, 0],
                              "y": sample["coords"][:, 1], "total_counts": totals, "detected_genes": detected})
        for k, activity in enumerate(truth["activity_ids"]):
            fraction = truth["activity_fraction"][:, k]
            point[f"core_expression_{activity}"] = sample["core_scores"][:, k]
            activities.append({
                "nside": side, "activity_id": activity, "n_specific_genes": int(truth["specific"][:, k].sum()),
                "fractional_occupied_positions": float(fraction.sum()), "support_area": float(fraction.sum() * area),
                "positions_with_any_support": int((fraction > 0).sum()),
                "positions_with_partial_support": int(((fraction > 0) & (fraction < 1)).sum()),
                "latent_activity_max": float(truth["activation"][:, k].max()),
                "core_expression_mean_all_positions": float(sample["core_scores"][:, k].mean()),
            })
        points.append(point)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "sample_summary.csv", index=False)
    pd.concat(points, ignore_index=True).to_csv(out / "spot_qc.csv", index=False)
    pd.DataFrame(activities).to_csv(out / "activity_qc.csv", index=False)
    return summary


def map_axis(ax, values: np.ndarray, side: int, **kwargs):
    image = ax.imshow(values.reshape(side, side, *values.shape[1:]), origin="upper",
                      extent=(0, 128, 128, 0), interpolation="nearest", **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def panel_label(ax, letter: str, *, x: float = -0.14, y: float = 1.04) -> None:
    """Label a panel or a shared row without changing its data."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            ha="left", va="bottom", clip_on=False)


def plot_overview(samples: list[dict], summary: pd.DataFrame, out: Path) -> None:
    """Use all activities/positions; fractional mixtures never become majority labels."""
    if any(np.any(sample["totals"] <= 0) for sample in samples):
        raise ValueError("The log-count overview requires positive counts at every position.")
    with mpl.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(12.4, 11.6), gridspec_kw={"height_ratios": [1, 1, 0.7]})
        fig.subplots_adjust(left=0.07, right=0.91, top=0.965, bottom=0.115, hspace=0.30, wspace=0.20)
        palette = np.array([to_rgb(c) for c in ACTIVITY_COLORS])
        limits = [min(np.log10(s["totals"]).min() for s in samples),
                  max(np.log10(s["totals"]).max() for s in samples)]
        for col, sample in enumerate(samples):
            side, truth = sample["nside"], sample["truth"]
            fractions = truth["activity_fraction"]
            colors = fractions @ palette / fractions.sum(axis=1, keepdims=True)
            map_axis(axes[0, col], colors, side)
            axes[0, col].set_title(f"{side} × {side}\n{side**2:,} locations", fontsize=12, pad=10)
            count_image = map_axis(axes[1, col], np.log10(sample["totals"]), side,
                                   cmap="viridis", vmin=limits[0], vmax=limits[1])
        for row, text in enumerate(("Activity coverage\nArea-weighted colors", "Observed counts\nShared log10 scale")):
            axes[row, 0].set_ylabel(text, fontsize=10, labelpad=13)
        count_position = axes[1, -1].get_position()
        fig.colorbar(count_image, cax=fig.add_axes([0.935, count_position.y0, 0.013, count_position.height]),
                     label="log10(counts)")
        resolution_colors = ["#356FA1", "#C78D57", "#6C9C85"]
        for sample, color in zip(samples, resolution_colors):
            n = len(sample["totals"])
            probability = np.arange(1, n + 1) / n
            axes[2, 0].step(np.sort(sample["totals"]), probability, where="post", color=color,
                            label=f"{sample['nside']} × {sample['nside']}", linewidth=1.4)
            axes[2, 1].step(np.sort(sample["detected"]), probability, where="post", color=color, linewidth=1.4)
        axes[2, 0].set_xscale("log")
        axes[2, 0].set(xlabel="Raw counts per location (log scale)", ylabel="Cumulative fraction", ylim=(0, 1.02))
        axes[2, 1].set(xlabel="Detected genes per location", ylabel="Cumulative fraction", ylim=(0, 1.02))
        axes[2, 0].legend(frameon=False, fontsize=8, loc="upper left")
        axes[2, 2].set_axis_off()
        cells = [[f"{int(r.nside)}", f"{int(r.total_counts):,}", f"{r.mean_counts:,.0f}"]
                 for r in summary.itertuples()]
        table = axes[2, 2].table(cellText=cells, colLabels=["Grid side", "Total counts", "Mean / location"],
                                 loc="center", cellLoc="center", colWidths=[0.23, 0.45, 0.32])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.7)
        axes[2, 2].set_title("Counts summary", fontsize=10)
        for ax, letter in [(axes[0, 0], "a"), (axes[1, 0], "b"),
                           (axes[2, 0], "c"), (axes[2, 1], "d"), (axes[2, 2], "e")]:
            panel_label(ax, letter, x=-.18)
        handles = [Patch(facecolor=c, label=a)
                   for c, a in zip(ACTIVITY_COLORS, samples[0]["truth"]["activity_ids"])]
        fig.legend(handles=handles,
                   loc="lower center", bbox_to_anchor=(0.49, 0.025), ncol=11, frameon=False, fontsize=9,
                   handlelength=1, columnspacing=0.9)
        fig.savefig(out / "overview.png", dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_all_activities(samples: list[dict], out: Path) -> None:
    """Compare every activity's latent intensity and observed specific-gene mean."""
    ids = samples[0]["truth"]["activity_ids"]
    latent_max = max(float(s["truth"]["activation"].max()) for s in samples)
    expression_max = max(float(s["core_scores"].max()) for s in samples)
    latent_cmap = LinearSegmentedColormap.from_list("latent", ["#FFFFFF", "#34795A"])
    expression_cmap = LinearSegmentedColormap.from_list("observed", ["#FFFFFF", "#765798"])
    with mpl.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(ids), 6, figsize=(14.4, 21.5))
        fig.subplots_adjust(left=0.09, right=0.875, top=0.97, bottom=0.025, hspace=0.07, wspace=0.08)
        for row, activity in enumerate(ids):
            for k, sample in enumerate(samples):
                side = sample["nside"]
                latent_image = map_axis(axes[row, 2*k], sample["truth"]["activation"][:, row], side,
                                        cmap=latent_cmap, vmin=0, vmax=latent_max)
                expression_image = map_axis(axes[row, 2*k+1], sample["core_scores"][:, row], side,
                                            cmap=expression_cmap, vmin=0, vmax=expression_max)
                if row == 0:
                    axes[row, 2*k].set_title(f"{side} × {side}\nLatent activity", fontsize=11, pad=10)
                    axes[row, 2*k+1].set_title(f"{side} × {side}\nSpecific-gene expression", fontsize=11, pad=10)
            n_core = int(samples[0]["truth"]["specific"][:, row].sum())
            axes[row, 0].set_ylabel(f"{activity}\n{n_core} genes", fontsize=10, rotation=0,
                                    labelpad=30, va="center")
            panel_label(axes[row, 0], chr(ord("a") + row), x=-.42)
        fig.colorbar(latent_image, cax=fig.add_axes([0.91, 0.62, 0.014, 0.28]), orientation="vertical",
                     label="Latent activity (per unit area)")
        fig.colorbar(expression_image, cax=fig.add_axes([0.91, 0.15, 0.014, 0.28]), orientation="vertical",
                     label="Mean specific-gene log1p-normalized expression")
        fig.savefig(out / "all_activities.png", dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def select_genes(high: dict) -> pd.DataFrame:
    """Select by generating parameters only, independently of observed expression."""
    truth = high["truth"]
    metadata = pd.read_csv(high["folder"] / "truth/genes.csv").fillna({"activities": ""})
    if not np.array_equal(metadata.gene_id, truth["gene_ids"]):
        raise ValueError("Gene metadata is not aligned with the expression matrix.")
    weights = truth["weights"]
    records = []

    def add(indices, parameter, quantile, panel, group, label):
        order = np.argsort(parameter[indices], kind="stable")
        gene = indices[order[int(np.floor(quantile * (len(order) - 1)))]]
        record = metadata.iloc[gene].to_dict()
        record.update(gene_index=int(gene), panel=panel, group=group, label=label,
                      selection_quantile=quantile, selection_parameter=float(parameter[gene]))
        records.append(record)

    for k, activity in enumerate(truth["activity_ids"]):
        indices = np.flatnonzero(truth["specific"][:, k])
        for quantile in (0.25, 0.75):
            add(indices, weights[:, k], quantile, "specific", activity,
                f"{activity} specific gene | loading Q{int(quantile * 100)}")
    shared = metadata[metadata.category == "shared"]
    for memberships in shared.activities.drop_duplicates():
        indices = shared.index[shared.activities == memberships].to_numpy()
        add(indices, weights.sum(axis=1), 0.5, "other", memberships,
            f"Shared gene | {memberships.replace(';', ' + ')}")
    labels = {"broad_background": "Broad background", "nonspatial_background": "Nonspatial background",
              "unstructured_background": "Unstructured background"}
    with np.load(high["folder"] / "truth/arrays.npz") as arrays:
        nonspatial = arrays["nonspatial_weights"].sum(axis=1)
    for category, label in labels.items():
        indices = metadata.index[metadata.category == category].to_numpy()
        parameter = nonspatial if category == "nonspatial_background" else metadata.baseline.to_numpy()
        add(indices, parameter, 0.5, "other", category, label)
    return pd.DataFrame(records)


def plot_genes(samples: list[dict], out: Path) -> None:
    """Show actual individual-gene maps with a shared scale across resolutions."""
    selected = select_genes(samples[0])
    selected.to_csv(out / "tables/selected_genes.csv", index=False)
    indices = selected.gene_index.to_numpy()
    source = {"gene_ids": selected.gene_id.to_numpy(dtype=str)}
    observed = {}
    for sample in samples:
        side = sample["nside"]
        values = sample["normalized"][:, indices].toarray()
        observed[side] = values
        source[f"log1p_norm_{side}"] = values
        source[f"counts_{side}"] = sample["counts"][:, indices].toarray()
        source[f"spot_ids_{side}"] = sample["truth"]["spot_ids"]
    np.savez_compressed(out / "tables/selected_gene_expression.npz", **source)
    titles = {"specific": "Individual specific genes for all 11 activities",
              "other": "Shared genes and three background classes"}
    for panel, title in titles.items():
        subset = selected[selected.panel == panel]
        pairs = 2 if panel == "specific" else 1
        nrows = len(subset) // pairs
        with mpl.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(nrows, 3 * pairs, figsize=(6.2 * pairs + 1, 2.1 * nrows + 1.6),
                                     squeeze=False)
            fig.subplots_adjust(left=0.04, right=0.95, top=0.97, bottom=0.025, hspace=0.45, wspace=0.20)
            for position, (index, gene) in enumerate(subset.iterrows()):
                row, block = divmod(position, pairs)
                vmax = max(float(observed[s["nside"]][:, index].max()) for s in samples)
                for column, sample in enumerate(samples):
                    ax = axes[row, block * 3 + column]
                    image = map_axis(ax, observed[sample["nside"]][:, index], sample["nside"],
                                     cmap="magma", vmin=0, vmax=max(vmax, 1e-8))
                    ax.set_title(f"{sample['nside']}×{sample['nside']}", fontsize=9, pad=3)
                first = axes[row, block * 3]
                panel_label(first, chr(ord("a") + position), x=-.18, y=1.22)
                first.text(0, 1.22, f"{gene.gene_id}  |  {gene.label}", transform=first.transAxes,
                           fontsize=9, ha="left", va="bottom")
                fig.colorbar(image, ax=list(axes[row, block * 3:block * 3 + 3]),
                             fraction=0.022, pad=0.015, ticks=[0, vmax], format="%.1f")
            fig.savefig(out / "figures" / f"genes_{panel}.png", dpi=300, facecolor="white", bbox_inches="tight")
            plt.close(fig)


def write_report(samples: list[dict], summary: pd.DataFrame, checks: list[dict], out: Path) -> None:
    failed = [c for c in checks if not c["passed"]]
    validation = {"status": "FAIL" if failed else "PASS", "n_checks": len(checks), "checks": checks,
                  "figure_language": "English",
                  "checker_sha256": sha256(Path(__file__)), "figures_generated": not failed,
                  "inputs": [{"sample_id": s["sample_id"], "manifest_sha256": s["input_manifest_sha256"]}
                             for s in samples],
                  "observation_score": "mean saved log1p_norm across every activity-specific core gene",
                  "figure_scope": "Four diagnostic PNGs; all activities and positions, parameter-selected gene examples.",
                  "qa": "Large discussion canvases, 300 dpi PNG only; no publication vector/TIFF exports requested."}
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    lines = ["# Simulation checks across resolutions", "", f"Validation: {validation['status']} ({len(checks)} checks).", "",
             "Inputs are read-only. All three resolutions contain the same 5,000 genes and 11 activities; no resimulation or fitting.", ""]
    if not failed:
        lines += ["![Simulation overview](figures/overview.png)", "", "![All activities](figures/all_activities.png)", ""]
        lines += ["![Specific genes](figures/genes_specific.png)", "",
                  "![Shared and background genes](figures/genes_other.png)", "",
                  "Individual-gene maps show observed log1p_norm values, with a common scale across resolutions for each gene.",
                  "Specific genes: Q25/Q75 loading ranks within each activity. Shared genes: median total-loading rank within each group.",
                  "Background genes: median baseline rank for broad/unstructured classes, median nonspatial-loading rank otherwise.",
                  "All 34 example genes are selected independently of observed maps. IDs, classes and selection parameters: tables/selected_genes.csv.",
                  "Raw counts, plotted values and location IDs: tables/selected_gene_expression.npz.", ""]
    else:
        lines += ["Some checks failed; figures were not updated. See validation.json.", ""]
    lines += ["| Grid | Locations | Total counts | Mean counts | Median detected genes |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for r in summary.itertuples():
        lines.append(f"| {r.nside}×{r.nside} | {r.n_spots:,} | {r.total_counts:,} | {r.mean_counts:,.1f} | "
                     f"{r.median_detected_genes:,.0f} |")
    lines += ["", "Activity colors are weighted by true area fractions; no majority labels or overlap inference from color alone.",
              "Activity maps: green denotes latent intensity, purple denotes mean observed expression across all specific genes. Separate shared scales reflect different units.",
              "These expression maps diagnose inputs, not REGAL recovery. Coarse expression mixtures do not by themselves establish activity loss.", "",
              "Source tables contain all 21,504 locations (spot_qc.csv) and all 33 activity-resolution records (activity_qc.csv).", "",
              "This is one descriptive simulation and seed, not independent replication or cross-platform generalization.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "simulation_check")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = args.data_root or Path(config["data_root"])
    out = args.output_root
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    checks = []
    samples = []
    for side in config["resolutions"]:
        print(f"Reading and checking {side}x{side}.", flush=True)
        samples.append(read_sample(data_root, config["sample_prefix"], side, checks))
    check_series(samples, checks)
    summary = write_tables(samples, out / "tables")
    if all(c["passed"] for c in checks):
        plot_overview(samples, summary, out / "figures")
        plot_all_activities(samples, out / "figures")
        plot_genes(samples, out)
    write_report(samples, summary, checks, out)
    failed = [c["name"] for c in checks if not c["passed"]]
    print(json.dumps({"status": "FAIL" if failed else "PASS", "n_checks": len(checks),
                      "failed": failed, "report": str(out / "report.md")}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
