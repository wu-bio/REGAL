from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/regal_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.dataset_registry import get_dataset
from Utils.spatial_plotting import (
    ensure_out_dir,
    load_spatial_coords,
    sample_dir_to_h5ad_path,
    save_figure,
    write_manifest,
)


def _domain_membership_table(domains: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    cols = ["domain_key", "rsmd_id", "rsp_seed_id", "qc_pass", "domain_reliability", "spot_count"]
    keep = [c for c in cols if c in domains.columns]
    return membership.merge(domains[keep].drop_duplicates("domain_key"), on="domain_key", how="left")


def _rsp_activation_plot_table(
    coords: pd.DataFrame,
    rsp_activation: pd.DataFrame,
    rsp_id: str,
) -> pd.DataFrame:
    sub = rsp_activation.loc[rsp_activation["rsp_id"].astype(str).eq(str(rsp_id)), ["spot_id", "activation"]].copy()
    sub["spot_id"] = sub["spot_id"].astype(str)
    sub["activation"] = pd.to_numeric(sub["activation"], errors="coerce").fillna(0.0)
    sub = sub.drop_duplicates("spot_id")
    out = coords.merge(sub, on="spot_id", how="left")
    out["activation"] = out["activation"].fillna(0.0)
    out["rsp_active"] = out["activation"].astype(float) > 0.0
    return out


def _adaptive_panel_grid_shape(n_items: int) -> tuple[int, int]:
    n = max(0, int(n_items))
    if n <= 1:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    while cols <= rows:
        cols += 1
        rows = int(np.ceil(n / cols))
    return rows, cols


def _clear_single_rsmd_pngs(panel_dir: Path) -> None:
    for path in panel_dir.glob("RSMD*.png"):
        path.unlink()


def plot_rsmd_by_rsp(coords: pd.DataFrame, merged: pd.DataFrame, rsp_ids: list[str], out_path: Path) -> None:
    rows, cols = _adaptive_panel_grid_shape(len(rsp_ids))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for ax, rsp_id in zip(axes.ravel(), rsp_ids):
        ax.scatter(coords["x"], coords["y"], s=5, c="#dddddd", linewidths=0, alpha=0.45)
        sub = merged.loc[merged["rsp_seed_id"].astype(str).eq(rsp_id)]
        plot_df = coords.merge(sub[["spot_id", "domain_key", "domain_reliability"]], on="spot_id", how="inner")
        values = plot_df["domain_reliability"].astype(float).fillna(0.0).to_numpy()
        if len(plot_df):
            handle = ax.scatter(
                plot_df["x"],
                plot_df["y"],
                s=8,
                c=values,
                cmap="plasma",
                linewidths=0,
                vmin=0,
                vmax=max(0.01, float(np.nanmax(values))),
            )
            fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rsp_id, loc="left", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(rsp_ids) :]:
        ax.axis("off")
    fig.suptitle("RSMD instances by seed RSP", x=0.02, ha="left", fontsize=12, fontweight="bold")
    save_figure(fig, out_path)


def plot_single_rsmd_panels(
    coords: pd.DataFrame,
    merged: pd.DataFrame,
    rsp_activation: pd.DataFrame,
    metadomains: pd.DataFrame,
    out_dir: Path,
    max_objects: int | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    sort_cols = [c for c in ["rsp_score", "max_domain_reliability", "total_domain_spots"] if c in metadomains.columns]
    md = metadomains.copy()
    if sort_cols:
        md = md.sort_values(sort_cols, ascending=False)
    if max_objects is not None:
        md = md.head(max_objects)

    panel_dir = ensure_out_dir(out_dir / "single_rsmd")
    _clear_single_rsmd_pngs(panel_dir)
    for row in md.itertuples(index=False):
        rsmd_id = str(getattr(row, "rsmd_id"))
        rsp_id = str(getattr(row, "rsp_id", rsmd_id))
        sub = merged.loc[merged["rsmd_id"].astype(str).eq(rsmd_id)].copy()
        activation_df = _rsp_activation_plot_table(coords=coords, rsp_activation=rsp_activation, rsp_id=rsp_id)
        plot_df = coords.merge(
            sub[["spot_id", "domain_key", "domain_reliability", "spot_count"]].drop_duplicates("spot_id"),
            on="spot_id",
            how="left",
        )
        active = plot_df["domain_key"].notna()

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
        axes[0].scatter(activation_df["x"], activation_df["y"], s=5, c="#dddddd", linewidths=0, alpha=0.45)
        rsp_active = activation_df["rsp_active"].astype(bool)
        if rsp_active.any():
            activation_values = activation_df.loc[rsp_active, "activation"].astype(float).fillna(0.0)
            handle = axes[0].scatter(
                activation_df.loc[rsp_active, "x"],
                activation_df.loc[rsp_active, "y"],
                s=8,
                c=activation_values,
                cmap="plasma",
                linewidths=0,
                vmin=0,
                vmax=max(0.01, float(activation_values.max())),
            )
            fig.colorbar(handle, ax=axes[0], fraction=0.046, pad=0.02, label="RSP activation")
        axes[0].set_title(f"{rsp_id} raw activation", loc="left", fontsize=9)

        axes[1].scatter(plot_df["x"], plot_df["y"], s=5, c="#dddddd", linewidths=0, alpha=0.45)
        if active.any():
            values = plot_df.loc[active, "domain_reliability"].astype(float).fillna(0.0)
            handle = axes[1].scatter(
                plot_df.loc[active, "x"],
                plot_df.loc[active, "y"],
                s=8,
                c=values,
                cmap="plasma",
                linewidths=0,
                vmin=0,
                vmax=max(0.01, float(values.max())),
            )
            fig.colorbar(handle, ax=axes[1], fraction=0.046, pad=0.02, label="domain reliability")
        axes[1].set_title(f"{rsmd_id} domain reliability", loc="left", fontsize=9)

        for ax in axes:
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        n_domains = int(getattr(row, "domain_count_pass", sub["domain_key"].nunique()))
        n_spots = int(getattr(row, "total_domain_spots", active.sum()))
        fig.suptitle(
            f"{rsmd_id} / {rsp_id}: {n_domains} passing domains, {n_spots} spots",
            x=0.02,
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        out_path = panel_dir / f"{rsmd_id}.png"
        save_figure(fig, out_path)
        outputs.append(out_path)
    return outputs


def plot_rsmd_summary(metadomains: pd.DataFrame, domains: pd.DataFrame, out_path: Path) -> None:
    df = metadomains.copy()
    if "rsp_id" in df.columns:
        df = df.sort_values("rsp_id")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    xlabels = df.get("rsp_id", df["rsmd_id"]).astype(str).tolist()
    axes[0].bar(xlabels, df["domain_count_pass"].astype(float), color="#2b8cbe")
    axes[0].set_title("Passing domains per RSMD", loc="left", fontweight="bold")
    axes[0].set_ylabel("domain count")
    axes[0].tick_params(axis="x", rotation=45)
    rel = domains.groupby("rsmd_id", as_index=False)["domain_reliability"].mean()
    rel = rel.set_index("rsmd_id").reindex(df["rsmd_id"].astype(str)).reset_index()
    axes[1].bar(xlabels, rel["domain_reliability"].astype(float), color="#f46d43")
    axes[1].set_title("Mean domain reliability", loc="left", fontweight="bold")
    axes[1].set_ylabel("mean reliability")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    save_figure(fig, out_path)


def visualize_sample(
    sample_dir: Path,
    out_dir: Path | None = None,
    max_rsps: int | None = None,
    max_single_objects: int | None = None,
) -> list[Path]:
    rsmd_out = sample_dir / "rsmd_out"
    rsp_activation_path = sample_dir / "rsp_out" / "rsp_activation.parquet"
    domains_path = rsmd_out / "domains.parquet"
    membership_path = rsmd_out / "domain_spot_membership.parquet"
    metadomains_path = rsmd_out / "metadomains.parquet"
    for path in [domains_path, membership_path, metadomains_path, rsp_activation_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing RSMD table: {path}")

    coords = load_spatial_coords(sample_dir_to_h5ad_path(sample_dir))
    domains = pd.read_parquet(domains_path)
    membership = pd.read_parquet(membership_path)
    metadomains = pd.read_parquet(metadomains_path)
    rsp_activation = pd.read_parquet(rsp_activation_path)
    merged = _domain_membership_table(domains, membership)
    rsp_ids = metadomains["rsp_id"].astype(str).tolist() if "rsp_id" in metadomains.columns else sorted(merged["rsp_seed_id"].dropna().astype(str).unique())
    if max_rsps is not None:
        rsp_ids = rsp_ids[:max_rsps]

    final_out = ensure_out_dir(out_dir or sample_dir / "vis" / "rsmd")
    outputs = [
        final_out / "fig01_rsmd_instances_by_rsp.png",
        final_out / "fig02_rsmd_summary.png",
    ]
    plot_rsmd_by_rsp(coords, merged, rsp_ids, outputs[0])
    plot_rsmd_summary(metadomains, domains, outputs[1])
    outputs.extend(
        plot_single_rsmd_panels(
            coords,
            merged,
            rsp_activation,
            metadomains,
            final_out,
            max_objects=max_single_objects,
        )
    )
    write_manifest(final_out / "visualization_manifest.csv", outputs, {"object": "RSMD", "sample_dir": str(sample_dir)})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Regional Specificity Metadomain (RSMD) objects.")
    parser.add_argument("--dataset", default="WU", help="Dataset id from Utils.dataset_registry.")
    parser.add_argument("--sample", default=None, help="Single sample id. Defaults to all samples in dataset.")
    parser.add_argument("--sample-dir", default=None, help="Explicit sample directory.")
    parser.add_argument("--out-dir", default=None, help="Explicit output directory.")
    parser.add_argument("--max-rsps", type=int, default=None, help="Limit number of RSP panels shown.")
    parser.add_argument("--max-single-objects", type=int, default=None, help="Limit number of single RSMD figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_dir:
        sample_dirs = [Path(args.sample_dir)]
    else:
        dataset = get_dataset(args.dataset)
        sample_ids = [args.sample] if args.sample else dataset.sample_ids
        sample_dirs = [dataset.sample_dir(str(sample_id)) for sample_id in sample_ids]
    for sample_dir in sample_dirs:
        outputs = visualize_sample(
            sample_dir,
            Path(args.out_dir) if args.out_dir else None,
            max_rsps=args.max_rsps,
            max_single_objects=args.max_single_objects,
        )
        print(f"[RSMD] {sample_dir.name}: wrote {len(outputs)} figures to {outputs[0].parent}")


if __name__ == "__main__":
    main()
