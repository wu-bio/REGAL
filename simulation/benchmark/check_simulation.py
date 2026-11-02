"""Check simulated data and export layout and expression QC figures; never fit methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.dataset_registry import DatasetSpec
from utils.h5ad_schema import check_rss_h5ad_schema

if __package__:
    from .generate_simulation import gene_counts, input_h5ad_path, sha256, write_json, membership_groups
else:
    from generate_simulation import gene_counts, input_h5ad_path, sha256, write_json, membership_groups


def nested_partitions(c: dict) -> list[dict]:
    """Read actual nested relations, including those in preserved sample metadata."""
    pairs = {(p['a'], p['b']) for p in c['relation_pairs'] if p['relation'] == 'nested'}
    return [p for p in c.get('heterogeneous_partitions', []) if (p['parent'], p['children'][0]) in pairs]


def moran_grid(values: np.ndarray, nside: int) -> np.ndarray:
    """Moran I for all columns, using the same undirected four-neighbor grid."""
    centered = values.astype(np.float64) - values.mean(axis=0)
    grid = centered.reshape(nside, nside, -1)
    numerator = (grid[:-1] * grid[1:]).sum(axis=(0, 1))
    numerator += (grid[:, :-1] * grid[:, 1:]).sum(axis=(0, 1))
    denominator = (centered * centered).sum(axis=0)
    return (nside**2 / (2 * nside * (nside - 1))) * np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def separation_auc(values: np.ndarray, inside: np.ndarray) -> float:
    """P(an inside value exceeds an outside value), with half credit for ties."""
    n_in, n_out = int(inside.sum()), int((~inside).sum())
    ranks = rankdata(values, method="average")
    return float((ranks[inside].sum() - n_in * (n_in + 1) / 2) / (n_in * n_out))


def _check_data(root: Path) -> bool:
    h5ad_path = input_h5ad_path(root)
    root = root / "simulation"
    c = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    qc = root / "qc"
    if qc.exists():
        raise FileExistsError(f"Refusing to replace existing QC: {qc}")
    tests, warnings = [], []

    def require(name: str, ok: bool, detail: object = "") -> None:
        tests.append({"name": name, "ok": bool(ok), "detail": detail})

    require("manifest_hashes", all(sha256(root / p) == digest for p, digest in manifest["files"].items()))
    data = ad.read_h5ad(h5ad_path)
    counts = data.X.toarray()
    t = np.load(root / "truth/arrays.npz", allow_pickle=False)
    activity_table = pd.read_csv(root / "truth/activities.csv")
    gene_table = pd.read_csv(root / "truth/genes.csv", keep_default_na=False)
    deviations = pd.read_csv(root / "truth/local_deviations.csv")
    ids = list(t["activity_ids"])
    support = t["instances"] > 0
    nside = c["nside"]
    require("input_counts", counts.shape == (nside**2, c["n_genes"]) and np.isfinite(counts).all()
            and np.issubdtype(counts.dtype, np.integer) and counts.min() >= 0)
    if "log1p_norm" in data.layers:
        require("input_has_no_truth", list(data.obs.columns) == ["sample_id"]
                and list(data.var.columns) == ["gene_id", "gene_name"] and list(data.obsm) == ["spatial"]
                and list(data.uns) == ["normalization"] and not data.varm and list(data.layers) == ["log1p_norm"])
        target = c.get("normalization_target_sum", 10000.0)
        expected_normalized = np.log1p(counts * (target / np.maximum(counts.sum(axis=1), 1))[:, None])
        require("normalized_layer_matches_counts", np.allclose(
            data.layers["log1p_norm"].toarray(), expected_normalized, rtol=2e-6, atol=1e-6))
        require("normalized_layer_metadata", data.uns["normalization"]["source"] == "X"
                and data.uns["normalization"]["target_sum"] == target)
        require("sample_and_gene_metadata", np.all(data.obs["sample_id"] == c["sample_id"])
                and np.array_equal(data.var["gene_id"], data.var_names)
                and np.array_equal(data.var["gene_name"], data.var_names))
        del expected_normalized
    else:
        require("input_has_no_truth", list(data.obs.columns) == [] and list(data.var.columns) == []
                and list(data.obsm) == ["spatial"] and not data.uns and not data.varm and not data.layers)
    require("unique_and_aligned_ids", data.obs_names.is_unique and data.var_names.is_unique
            and np.array_equal(data.obs_names, t["spot_ids"]) and np.array_equal(data.var_names, t["gene_ids"])
            and np.array_equal(gene_table.gene_id, t["gene_ids"]))
    coords_file = pd.read_csv(root / ("coordinates.csv" if (root / "coordinates.csv").exists()
                                      else "input/coordinates.csv"), index_col=0)
    require("coordinate_alignment", np.array_equal(coords_file.index, data.obs_names)
            and np.allclose(coords_file.to_numpy(), data.obsm["spatial"]))
    spec = DatasetSpec("external_dataset", "SIMULATION", (c["sample_id"],), species="synthetic")
    schema = check_rss_h5ad_schema(h5ad_path, spec, c["sample_id"])
    require("rss_input_schema", schema.ok, schema.format_report())
    require("sixteen_activities", ids == [f"ACT{k:02d}" for k in range(1, 17)]
            and len(activity_table) == 16 and support.shape == (data.n_obs, 16))
    require("activation_support", np.isfinite(t["activation"]).all()
            and np.array_equal(t["activation"] > 0, support))
    require("finite_positive_means", np.isfinite(t["expected_counts"]).all() and t["expected_counts"].min() > 0)
    require("core_membership", np.all(t["specific"].sum(axis=0) == gene_counts(c, "specific_genes_per_activity", 16))
            and np.all((t["weights"] > 0).sum(axis=1)[t["specific"].any(axis=1)] == 1))
    expected_sizes = gene_counts(c, "specific_genes_per_activity", 16)
    for key, size, counts_key in [("shared_pair_groups",2,"shared_genes_per_pair"),("shared_activity_groups",4,"shared_genes_per_group")]:
        for j, ks in enumerate(membership_groups(c, key, size)):
            expected_sizes[ks] += gene_counts(c, counts_key, 16//size)[j]
    require("contributor_counts", np.array_equal((t["weights"] > 0).sum(axis=0), expected_sizes))
    memberships = (t["weights"] > 0).sum(axis=1)
    for category, cardinality, expected_count in [
        ("specific_core", 1, int(gene_counts(c, "specific_genes_per_activity", 16).sum())),
        ("pair_shared", 2, int(gene_counts(c, "shared_genes_per_pair", 8).sum())),
        ("group_shared", 4, int(gene_counts(c, "shared_genes_per_group", 4).sum()))]:
        mask = gene_table.category.to_numpy() == category
        require(f"{category}_unique_gene_count", mask.sum() == expected_count
                and np.all(memberships[mask] == cardinality))
    regional_budget = sum(gene_counts(c, key, length).sum() for key, length in [
        ("specific_genes_per_activity", 16), ("shared_genes_per_pair", 8), ("shared_genes_per_group", 4)])
    require("unique_regional_gene_budget", np.count_nonzero(memberships) == regional_budget
            and regional_budget == c.get("expected_regional_genes", regional_budget))
    require("nonregional_background_budget", np.count_nonzero(memberships == 0) == c["n_genes"] - regional_budget
            and (gene_table.category == "broad_background").sum() == c["broad_background_genes"])
    nonspatial_members = t["nonspatial_weights"] > 0
    require("nonspatial_background_gene_budget", np.all(
        nonspatial_members.sum(axis=0) == c["genes_per_nonspatial_factor"])
        and not np.any((memberships > 0) & nonspatial_members.any(axis=1)))
    require("weighted_shared_genes", all(np.std(t["weights"][t["weights"][:, k] > 0, k]) > 0 for k in range(16))
            and np.any((t["weights"] > 0).sum(axis=1) == 2)
            and np.any((t["weights"] > 0).sum(axis=1) == 4))

    instance_rows = []
    conn = ndimage.generate_binary_structure(2, 1)
    for k, activity in enumerate(c["activities"]):
        components, n_components = ndimage.label(support[:, k].reshape(nside, nside), structure=conn)
        require(f"{ids[k]}_connected_instances", n_components == len(activity["instances"]))
        for m in range(1, len(activity["instances"]) + 1):
            mask = t["instances"][:, k] == m
            component_ids = np.unique(components.ravel()[mask])
            require(f"{ids[k]}_instance{m}_label", len(component_ids) == 1 and component_ids[0] > 0)
            instance_rows.append({"activity_id": ids[k], "instance_id": m, "n_spots": int(mask.sum())})
    instance_table = pd.DataFrame(instance_rows)
    require("total_instances", len(instance_table) == sum(len(a["instances"]) for a in c["activities"]))
    if "landscape_layout" in c:
        primary_ids = [a["id"] for a in c["activities"] if a["relation"] == "primary"]
        primary_indices = [ids.index(activity_id) for activity_id in primary_ids]
        require("primary_partition_covers_slice", np.all(support[:, primary_indices].sum(axis=1) == 1))
        minimum = c["landscape_layout"]["minimum_instance_spots"]
        require("landscape_minimum_instance_spots", instance_table.n_spots.min() >= minimum)
        for activity in c["activities"]:
            fraction = float(support[:, ids.index(activity["id"])].mean())
            low, high = c["landscape_layout"]["scale_fraction_ranges"][activity["scale"]]
            require(f"{activity['id']}_scale_fraction", low <= fraction <= high, fraction)
    if 'maximum_active_layers' in c.get('landscape_layout', {}):
        require('maximum_active_layers', support.sum(axis=1).max() <= c['landscape_layout']['maximum_active_layers'])
        for partition in nested_partitions(c):
            parent = support[:, ids.index(partition['parent'])]
            children = [support[:, ids.index(v)] for v in partition['children']]
            child = children[0]
            fraction = float(child.sum()/parent.sum())
            low, high = c['landscape_layout']['child_fraction_of_parent_range']
            require(f"{partition['parent']}_single_nested_child", not (child & ~parent).any()
                    and (parent & ~child).any() and low <= fraction <= high, fraction)
            if c['landscape_layout'].get('overlays_avoid_heterogeneous_parents', False):
                family = parent | np.logical_or.reduce(children)
                for activity in c['activities']:
                    if activity.get('scale') == 'overlay':
                        require(f"{activity['id']}_avoids_{partition['parent']}_family",
                                not (support[:, ids.index(activity['id'])] & family).any())
        limit = c['landscape_layout']['maximum_overlapping_member_jaccard']
        membership = t['weights'] > 0
        for a in range(16):
            for b in range(a+1,16):
                if (support[:,a] & support[:,b]).any():
                    jaccard = (membership[:,a] & membership[:,b]).sum() / (membership[:,a] | membership[:,b]).sum()
                    require(f'{ids[a]}_{ids[b]}_distinct_members', jaccard <= limit, float(jaccard))
                    wa, wb = t['weights'][:,a], t['weights'][:,b]
                    cosine = np.dot(wa,wb) / (np.linalg.norm(wa)*np.linalg.norm(wb))
                    require(f'{ids[a]}_{ids[b]}_distinct_weights', cosine <= c['landscape_layout']['maximum_overlapping_weight_cosine'], float(cosine))
    # No match to REGAL thresholds is required; record size so future filtering is interpretable.
    relation_rows = []
    for pair in c["relation_pairs"]:
        a, b = ids.index(pair["a"]), ids.index(pair["b"])
        ma, mb = support[:, a], support[:, b]
        overlap = int((ma & mb).sum())
        distance = float(cKDTree(data.obsm["spatial"][ma]).query(data.obsm["spatial"][mb])[0].min())
        relation = pair["relation"]
        if relation == "adjacent":
            ok = overlap == 0 and np.isclose(distance, 64 / nside)
        elif relation == "partial_overlap":
            ok = overlap > 0 and (ma & ~mb).any() and (mb & ~ma).any()
        elif relation == "nested":
            ok = np.all(ma[mb]) and (ma & ~mb).any()
        else:
            raise ValueError(relation)
        require(f"{pair['a']}_{pair['b']}_{relation}", ok)
        relation_rows.append({**pair, "overlap_spots": overlap, "a_only_spots": int((ma & ~mb).sum()),
                              "b_only_spots": int((mb & ~ma).sum()), "minimum_distance": distance})
    declared = {tuple(sorted((ids.index(p["a"]), ids.index(p["b"])))) for p in c["relation_pairs"]}
    require("no_unplanned_overlaps", all(not np.any(support[:, a] & support[:, b])
            for a in range(16) for b in range(a + 1, 16) if (a, b) not in declared))

    # Reconstruct the complete mean independently from saved activity/loadings/local deviations.
    expected = t["activation"].astype(np.float64) @ t["weights"].T.astype(np.float64)
    gene_index = {g: j for j, g in enumerate(t["gene_ids"])}
    for (activity_id, m), group in deviations.groupby(["activity_id", "instance_id"]):
        k = ids.index(activity_id)
        mask = t["instances"][:, k] == m
        gs = np.array([gene_index[g] for g in group.gene_id])
        delta = t["weights"][gs, k] * (group.multiplier.to_numpy() - 1)
        expected[np.ix_(mask, gs)] += np.outer(t["activation"][mask, k], delta)
        require(f"{activity_id}_instance{m}_bounded_local_change",
                len(gs) == round((t["weights"][:, k] > 0).sum() * c["local_deviation_fraction"])
                and np.all(t["weights"][gs, k] > 0)
                and group.multiplier.between(c["local_multiplier_min"], c["local_multiplier_max"]).all())
    if c.get("spatial_heterogeneity", {}).get("enabled", False):
        h = np.load(root / "truth/gene_activity_fields.npz", allow_pickle=False)
        fields, js, ks = h["fields"], h["gene_indices"], h["activity_indices"]
        pairs = np.column_stack((js, ks))
        require("heterogeneity_memberships", np.array_equal(pairs, np.argwhere(t["weights"].T > 0)[:, ::-1]))
        require("heterogeneity_finite_fields", fields.shape == (data.n_obs, len(js))
                and np.isfinite(fields).all() and np.all(fields >= 0))
        reconstructed = np.zeros_like(expected)
        for col, (j, k) in enumerate(pairs):
            reconstructed[:, j] += fields[:, col].astype(np.float64) * float(t["weights"][j, k])
        require("heterogeneity_total_enhancement_conserved", np.allclose(
            reconstructed.sum(axis=0), expected.sum(axis=0), rtol=2e-6, atol=1e-6))
        spec = c["spatial_heterogeneity"]
        for k, activity_id in enumerate(ids):
            columns = np.flatnonzero(ks == k)
            distance = ndimage.distance_transform_edt(~support[:, k].reshape(nside, nside), sampling=64/nside)
            allowed = distance.ravel() <= spec["boundary_width"] + spec["boundary_shift"] + 32/nside
            require(f"{activity_id}_bounded_gene_edges", not np.any(fields[~allowed][:, columns] > 0))
            profiles = fields[support[:, k]][:, columns].astype(np.float64)
            profiles /= profiles.mean(axis=0)
            require(f"{activity_id}_different_member_profiles", float(np.std(profiles, axis=1).mean()) > 0.01)
        expected = reconstructed
    expected *= c["signal_scale"]
    expected += t["baseline"][None, :] + t["nonspatial_activation"] @ t["nonspatial_weights"].T
    expected *= t["depth"][:, None]
    # Keep latent biological effects separate from the perturbed conditional count mean.
    conditional_expected = expected
    nb_shape = c["nb_shape"]
    noise = c.get("expression_noise", {})
    if noise.get("enabled", False):
        rng = np.random.default_rng(noise["seed"])
        multiplier = rng.lognormal(-noise["log_sd"]**2 / 2, noise["log_sd"], size=expected.shape)
        conditional_expected = expected * multiplier
        nb_shape = noise["nb_shape"]
        require("finite_positive_noise", np.isfinite(multiplier).all() and multiplier.min() > 0)
        del multiplier
    delta_max = float(np.max(np.abs(conditional_expected - t["expected_counts"])))
    require("expected_mean_reconstruction", np.allclose(conditional_expected, t["expected_counts"], rtol=2e-6, atol=2e-5), delta_max)

    rates = counts / t["depth"][:, None]
    library_counts = counts.sum(axis=1)
    normalized = counts * (10000 / np.maximum(library_counts, 1))[:, None]
    core_profiles = np.column_stack([rates[:, t["specific"][:, k]].mean(axis=1) for k in range(16)])
    relation_signal_rows = []
    for pair in c["relation_pairs"]:
        a, b = ids.index(pair["a"]), ids.index(pair["b"])
        ma, mb = support[:, a], support[:, b]
        for label, mask in [("a_only", ma & ~mb), ("b_only", mb & ~ma),
                            ("overlap", ma & mb), ("neither", ~ma & ~mb)]:
            if not mask.any():
                continue
            mean_a = float(core_profiles[mask, a].mean())
            mean_b = float(core_profiles[mask, b].mean())
            relation_signal_rows.append({**pair, "region": label, "n_spots": int(mask.sum()),
                                          "a_core_mean_depth_corrected": mean_a,
                                          "b_core_mean_depth_corrected": mean_b})
    signal_rows = []
    instance_signal_rows = []
    gene_signal_rows, baseline_signal_rows = [], []
    for k in range(16):
        core = t["specific"][:, k]
        mask = support[:, k]
        within = rates[mask][:, core].mean(axis=0)
        outside = rates[~mask][:, core].mean(axis=0)
        fc = float(np.log2((within.mean() + 0.01) / (outside.mean() + 0.01)))
        fraction = float(np.mean(within > outside))
        norm_in = float(normalized[mask][:, core].mean())
        norm_out = float(normalized[~mask][:, core].mean())
        signal_rows.append({"activity_id": ids[k], "relation": c["activities"][k]["relation"],
                            "n_contributing_genes": int((t["weights"][:, k] > 0).sum()),
                            "n_specific_core": int(core.sum()),
                            "n_spots": int(mask.sum()), "core_inside_mean_depth_corrected": float(within.mean()),
                            "core_outside_mean_depth_corrected": float(outside.mean()), "core_log2fc": fc,
                            "core_fraction_inside_greater": fraction,
                            "core_detection_inside": float((counts[mask][:, core] > 0).mean()),
                            "core_detection_outside": float((counts[~mask][:, core] > 0).mean()),
                            "core_log2fc_library_normalized": float(np.log2((norm_in + 0.01) / (norm_out + 0.01)))})
        if "program_baseline_classes" in c:
            for tier in c["program_baseline_classes"]:
                members = core & (gene_table.baseline_class.to_numpy() == tier["name"])
                if not members.any():
                    require(f"{ids[k]}_{tier['name']}_present", False)
                    continue
                ti = rates[mask][:, members].mean(axis=0)
                to = rates[~mask][:, members].mean(axis=0)
                baseline_signal_rows.append({"activity_id": ids[k], "baseline_class": tier["name"],
                                             "n_genes": int(members.sum()), "inside_mean": float(ti.mean()),
                                             "outside_mean": float(to.mean()),
                                             "log2fc": float(np.log2((ti.mean() + 0.01) / (to.mean() + 0.01))),
                                             "fraction_inside_greater": float(np.mean(ti > to)),
                                             "detection_inside": float((counts[mask][:, members] > 0).mean()),
                                             "detection_outside": float((counts[~mask][:, members] > 0).mean())})
                for j, mi, mo in zip(np.flatnonzero(members), ti, to):
                    gene_signal_rows.append({"activity_id": ids[k], "gene_id": t["gene_ids"][j],
                                             "baseline_class": tier["name"], "baseline": float(t["baseline"][j]),
                                             "loading": float(t["weights"][j, k]),
                                             "inside_mean": float(mi), "outside_mean": float(mo),
                                             "log2fc": float(np.log2((mi + 0.01) / (mo + 0.01)))})
        for m in range(1, int(t["instances"][:, k].max()) + 1):
            instance_mask = t["instances"][:, k] == m
            fc_m = float(np.log2((rates[instance_mask][:, core].mean() + 0.01) / (outside.mean() + 0.01)))
            instance_signal_rows.append({"activity_id": ids[k], "instance_id": m,
                                         "n_spots": int(instance_mask.sum()), "core_log2fc": fc_m})
    signal_table = pd.DataFrame(signal_rows)
    contributor_rows, core_instance_rows = [], []
    if "regional_enhancement" in c:
        for j in np.flatnonzero((t["weights"] > 0).any(axis=1)):
            parents = np.flatnonzero(t["weights"][j] > 0)
            union = support[:, parents].any(axis=1)
            raw_values = counts[:, j]
            corrected_values = rates[:, j]
            mean_in, mean_out = float(corrected_values[union].mean()), float(corrected_values[~union].mean())
            fold = mean_in / max(mean_out, 1e-12)
            auc_raw = separation_auc(raw_values, union)
            auc_corrected = separation_auc(corrected_values, union)
            # Expected effect is evaluated after local instance multipliers, before technical depth.
            mu_in = expected[union, j] / t["depth"][union]
            minimum_fold = float(mu_in.min() / t["baseline"][j])
            minimum_increment = float(mu_in.min() - t["baseline"][j])
            contributor_rows.append({"gene_id": t["gene_ids"][j], "category": gene_table.category.iloc[j],
                                      "baseline_class": gene_table.baseline_class.iloc[j],
                                      "activities": ";".join(ids[k] for k in parents), "n_inside": int(union.sum()),
                                      "inside_mean": mean_in, "outside_mean": mean_out, "fold": fold,
                                      "raw_auc": auc_raw, "depth_corrected_auc": auc_corrected,
                                      "minimum_expected_fold": minimum_fold,
                                      "minimum_expected_increment": minimum_increment})
        for k in range(16):
            for j in np.flatnonzero(t["specific"][:, k]):
                outside = float(rates[~support[:, k], j].mean())
                for m in range(1, int(t["instances"][:, k].max()) + 1):
                    mask = t["instances"][:, k] == m
                    mean_in = float(rates[mask, j].mean())
                    fold = mean_in / max(outside, 1e-12)
                    core_instance_rows.append({"activity_id": ids[k], "instance_id": m,
                                                "gene_id": t["gene_ids"][j], "fold": fold,
                                                "inside_mean": mean_in, "outside_mean": outside})
    null_mask = ~np.any(t["weights"] > 0, axis=1)
    null_moran = moran_grid(rates[:, null_mask], nside)
    null_summary = {"n_genes": int(null_mask.sum()), "mean_moran": float(null_moran.mean()),
                    "median_abs_moran": float(np.median(np.abs(null_moran))),
                    "p95_abs_moran": float(np.quantile(np.abs(null_moran), 0.95))}
    if null_summary["p95_abs_moran"] > 0.1:
        warnings.append("Background p95 absolute Moran I exceeds 0.1; inspect without reselecting the seed.")
    depth_correlations = [float(np.corrcoef(t["depth"], support[:, k])[0, 1]) for k in range(16)]
    residual = (counts - conditional_expected) / np.sqrt(
        conditional_expected + conditional_expected**2 / nb_shape)
    residual_stats = {"mean": float(residual.mean()), "mean_square": float(np.mean(residual**2))}
    if abs(residual_stats["mean"]) > 0.02 or abs(residual_stats["mean_square"] - 1) > 0.1:
        warnings.append("NB residual moment check outside diagnostic range; do not regenerate to pass.")
    # Create the QC directory only after inputs are read and all calculations complete.
    qc.mkdir()
    pd.DataFrame(core_profiles, index=data.obs_names, columns=ids).to_csv(qc / "core_profiles.csv")
    signal_table.to_csv(qc / "activity_signal.csv", index=False)
    if baseline_signal_rows:
        pd.DataFrame(baseline_signal_rows).to_csv(qc / "baseline_signal.csv", index=False)
        pd.DataFrame(gene_signal_rows).to_csv(qc / "gene_signal.csv", index=False)
    if contributor_rows:
        pd.DataFrame(contributor_rows).to_csv(qc / "contributor_gene_signal.csv", index=False)
        pd.DataFrame(core_instance_rows).to_csv(qc / "core_gene_instance_signal.csv", index=False)
    instance_table.to_csv(qc / "instances.csv", index=False)
    pd.DataFrame(instance_signal_rows).to_csv(qc / "instance_signal.csv", index=False)
    pd.DataFrame(relation_rows).to_csv(qc / "relations.csv", index=False)
    pd.DataFrame(relation_signal_rows).to_csv(qc / "relation_signal.csv", index=False)
    pd.DataFrame({"gene_id": t["gene_ids"][null_mask], "moran_i_depth_corrected": null_moran}).to_csv(
        qc / "background_moran.csv", index=False)
    passed = all(row["ok"] for row in tests)
    summary = {"status": "PASS" if passed else "FAIL", "checks": tests, "warnings": warnings,
               "shape": list(counts.shape), "n_activities": 16, "n_instances": len(instance_table),
               "minimum_instance_spots": int(instance_table.n_spots.min()),
               "core_log2fc_range": [float(signal_table.core_log2fc.min()), float(signal_table.core_log2fc.max())],
               "contributor_count_range": [int(expected_sizes.min()), int(expected_sizes.max())],
               "background": null_summary, "nb_standardized_residual": residual_stats,
               "max_abs_depth_support_correlation": float(np.max(np.abs(depth_correlations))),
               "total_counts_quantiles": np.quantile(library_counts, [0, 0.25, 0.5, 0.75, 1]).tolist(),
               "zero_fraction": float(np.mean(counts == 0)),
               "checker_sha256": sha256(Path(__file__)),
               "scope": "PASS/FAIL covers file integrity, schema and design structure only. Expression effect sizes are descriptive; no method performance, RSS or RSP selection."}
    if contributor_rows:
        contributors = pd.DataFrame(contributor_rows)
        summary["contributor_gene_statistics"] = {
            "n_genes": len(contributors),
            "minimum_observed_fold": float(contributors.fold.min()),
            "minimum_raw_auc": float(contributors.raw_auc.min()),
            "minimum_depth_corrected_auc": float(contributors.depth_corrected_auc.min()),
            "minimum_expected_fold": float(contributors.minimum_expected_fold.min()),
            "minimum_expected_increment": float(contributors.minimum_expected_increment.min()),
            "n_core_gene_instances": len(core_instance_rows)}
    write_json(qc / "summary.json", summary)
    lines = ["# 16 活动模拟数据检查", "", f"状态：{summary['status']}；{len(tests)} 项检查；{len(warnings)} 项诊断提醒。", "",
             f"数据：{data.n_obs:,} 个位置 × {data.n_vars:,} 个基因；16 个活动；{len(instance_table)} 个空间实例。",
             f"最小实例：{instance_table.n_spots.min()} 个位置。",
             f"按技术深度校正后的核心基因区域内外 log2FC：{signal_table.core_log2fc.min():.2f}–{signal_table.core_log2fc.max():.2f}。",
             f"背景基因绝对 Moran I 的95分位数：{null_summary['p95_abs_moran']:.4f}。", "",
             "PASS/FAIL仅针对文件、格式、数值有效性及预设结构；表达倍数、AUC和检测率仅作描述，不设通过门槛。",
             "四类关系、全部活动和全部实例的统计见各CSV及summary.json。",
             "区域特异性使用预设核心基因和区域计算，不使用REGAL RSS或方法输出。",
             "技术深度校正使用生成时的真实倍率，仅用于生成检查；它不进入方法输入。",
             "library-normalized列另外记录总量归一化后的结果，不把生物活动带来的总counts增量当成技术深度。",
             "本轮只有一个基准布局与一个表达种子，不构成跨样本或多种子性能证据。", ""]
    lines += [f"- 未通过：{row['name']}：{row['detail']}" for row in tests if not row["ok"]]
    lines += [f"- 诊断提醒：{warning}" for warning in warnings]
    if baseline_signal_rows:
        lines += ["", f"贡献成员数：{expected_sizes.min()}–{expected_sizes.max()}，各活动不再等大。",
                  "分别记录每活动低／中／高基础表达成员的区内外表达，不设置统一效果门槛。",
                  "具体区内外均值、检测率及每个成员数据见baseline_signal.csv和gene_signal.csv。",
                  "h5ad的X为原始counts，layers['log1p_norm']为总量归一化至10,000后log1p的表达。",
                  "未复制BREAST真实组织注释或图像；合成样本没有这些真值。"]
    if contributor_rows:
        effect = summary["contributor_gene_statistics"]
        lines += ["", f"记录全部{effect['n_genes']}个区域贡献基因的表达统计，不判定通过或失败。",
                  f"实际区内外均值倍数最小{effect['minimum_observed_fold']:.2f}；原始counts AUC最小{effect['minimum_raw_auc']:.3f}。",
                  "共享成员按所属活动区域并集评价，不把其他所属活动区域当阴性区域。",
                  f"同时记录全部{effect['n_core_gene_instances']}个特异核心基因／实例组合的表达统计。",
                  "AUC仅度量单基因值对预设区内外的排序分离，不是方法的恢复准确率，也不是p值。",
                  "不带真值边框的单基因图用于空间可辨认性审阅，不使用平滑或截断色阶。"]
    (qc / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ["status", "shape", "n_activities", "n_instances", "core_log2fc_range", "background"]}))
    return passed


def plot_layout(root: Path) -> None:
    """Render the saved activity truth, using the same data checked by QC."""
    root = root / "simulation"
    c = json.loads((root / "config.json").read_text())
    out = root / "qc/layout_review"
    if out.exists():
        raise FileExistsError(out)
    with np.load(root / "truth/arrays.npz", allow_pickle=False) as truth:
        s = {key: truth[key] for key in ("instances", "weights")}
    n = c["nside"]
    support = s["instances"] > 0
    with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
                     'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        ids=[v['id'] for v in c['activities']];prim=[k for k,v in enumerate(c['activities']) if v['relation']=='primary']
        assert np.all(support[:,prim].sum(axis=1)==1)
        primary=np.argmax(support[:,prim],axis=1)
        colors=['#4079A8','#E2A447','#73A79B','#9C83B7','#C97578','#8E9B59','#9A806D','#5D9F9A'][:len(prim)]
        out.mkdir(parents=True)
        sizes=pd.DataFrame([{'activity_id':v['id'],'scale':v['scale'],'n_spots':int(support[:,k].sum()),
                             'fraction':float(support[:,k].mean()),'n_instances':len(v['instances']),
                             'minimum_instance_spots':min(int((s['instances'][:,k]==j).sum()) for j in range(1,len(v['instances'])+1))}
                             for k,v in enumerate(c['activities'])])
        sizes.to_csv(out/'activity_sizes.csv',index=False)
        def save(fig,name):
            for panel_ax in fig.axes:
                for panel_title in (panel_ax.title, panel_ax._left_title, panel_ax._right_title):
                    label = panel_title.get_text()
                    if len(label) > 2 and label[0].islower() and label[1:3] == "  ":
                        panel_title.set_text(r"$\mathbf{" + label[0] + "}$" + label[1:])
            fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight')
            fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
            fig.savefig(out/f'{name}.svg',bbox_inches='tight')
            plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,5.4),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.15]})
        axes[0].imshow(primary.reshape(n,n),origin='lower',extent=(0,64,0,64),cmap=ListedColormap(colors),vmin=-.5,vmax=len(prim)-.5,interpolation='nearest')
        axes[0].set_title(f'a  {len(prim)} parallel primary activities')
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(facecolor=color,label=ids[k]) for color,k in zip(colors,prim)],ncol=4,loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9)
        additional=[k for k in range(len(ids)) if k not in prim]
        additional_support=support[:,additional].any(axis=1)
        axes[1].imshow(additional_support.reshape(n,n),origin='lower',extent=(0,64,0,64),
                       cmap=ListedColormap(['#FFFFFF','#B94D9C']),vmin=0,vmax=1,interpolation='nearest')
        axes[1].set_title('b  Nested and overlapping activities')
        for k in additional:
            components,n_components=ndimage.label(support[:,k].reshape(n,n))
            for component in range(1,n_components+1):
                mask=components==component
                distance=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
                y,x=np.unravel_index(np.argmax(distance),distance.shape)
                axes[1].text((x+.5)*64/n,(y+.5)*64/n,ids[k],ha='center',va='center',fontsize=9,
                             color='#222222',bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1.2})
        axes[2].barh(range(16),sizes.fraction*100,color='#62666B')
        axes[2].set_yticks(range(16),ids);axes[2].invert_yaxis();axes[2].set_xlabel('Percent of slice');axes[2].set_xlim(0,float(sizes.fraction.max()*100+4))
        for k,row in sizes.iterrows():axes[2].text(row.fraction*100+.4,k,f'{row.n_spots}',va='center',fontsize=8)
        axes[2].set_title('c  Activity sizes')
        for ax in axes[:2]:ax.set_xticks([0,32,64]);ax.set_yticks([0,32,64]);ax.set_xlabel('x (simulation units)')
        axes[0].set_ylabel('y (simulation units)')
        save(fig,'landscape_overview')
        # Combined multi-label map: primary fills with secondary boundaries; all
        # positions are retained and outlines never replace underlying primary labels.
        from scipy import ndimage
        from matplotlib.lines import Line2D
        secondary=[k for k in range(16) if k not in prim]
        fig,ax=plt.subplots(figsize=(9,9),layout='constrained')
        ax.imshow(primary.reshape(n,n),origin='lower',extent=(0,64,0,64),
                  cmap=ListedColormap(colors),vmin=-.5,vmax=len(prim)-.5,interpolation='nearest',alpha=.75)
        def label_inside(mask,label,color='#222222'):
            distance=ndimage.distance_transform_edt(np.pad(mask.reshape(n,n),1))[1:-1,1:-1]
            y,x=np.unravel_index(np.argmax(distance),distance.shape)
            ax.text((x+.5)*64/n,(y+.5)*64/n,label,ha='center',va='center',fontsize=10,
                    color=color,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.3})
        occupied=support[:,secondary].any(axis=1)
        for k in prim:
            components,count_components=ndimage.label(support[:,k].reshape(n,n))
            for instance in range(1,count_components+1):
                mask=components.ravel()==instance
                free=mask & ~occupied
                label_inside(free if free.any() else mask,ids[k])
        for k in secondary:
            nested=c['activities'][k]['scale']=='subregion'
            color='#333333' if nested else '#A34219'
            ax.contour(support[:,k].reshape(n,n),levels=[.5],colors=[color],
                       linestyles='--' if nested else '-',linewidths=1.8,
                       origin='lower',extent=(0,64,0,64))
            label_inside(support[:,k],ids[k],color)
        ax.set(xlim=(0,64),ylim=(0,64),xticks=[0,16,32,48,64],yticks=[0,16,32,48,64],
               xlabel='x (simulation units)',ylabel='y (simulation units)')
        handles=[Patch(facecolor=color,alpha=.75,label=ids[k]) for color,k in zip(colors,prim)]
        handles += [Line2D([0],[0],color='#333333',ls='--',label='Nested activity'),
                    Line2D([0],[0],color='#A34219',label='Partially overlapping activity')]
        ax.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.09),fontsize=10,frameon=False)
        save(fig,'combined_activity_layout')
        fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained',sharex=True,sharey=True)
        for k,ax in enumerate(axes.flat):
            ax.imshow(support[:,k].reshape(n,n),origin='lower',extent=(0,64,0,64),cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
            row=sizes.iloc[k]
            ax.set_title(f'{ids[k]} | {row.scale} | {row.fraction*100:.1f}%\n{row.n_instances} instance(s), {row.n_spots} spots',fontsize=10)
            ax.set_xticks([0,32,64]);ax.set_yticks([0,32,64])
        save(fig,'all_activity_supports')
        (out/'README.md').write_text('Layout-only review, fixed before count sampling and method fitting. All 4096 positions and 16 activities are shown without cropping or smoothing. Primary labels in the overview display only the non-overlapping backbone supports; other activities remain multi-label. All configured connected instances are retained. Size bar labels are spot counts. Review export: 300 dpi PNG and editable PDF/SVG in large layouts; no submission TIFF.\n')
        partitions = nested_partitions(c)
        if partitions:
            from matplotlib.colors import BoundaryNorm
            fig,axes=plt.subplots(len(partitions),4,figsize=(12,4.5*len(partitions)),layout='constrained',squeeze=False)
            for row,part in enumerate(partitions):
                parent=support[:,ids.index(part['parent'])]
                first=support[:,ids.index(part['children'][0])]
                second=parent & ~first
                assert not (first & second).any() and np.array_equal(first | second,parent)
                for col,(values,title) in enumerate([(parent,part['parent']+' parent'),(first,part['children'][0]),
                                                    (second,'Parent only'),
                                                    (first.astype(int)+2*second.astype(int),'Child / parent-only')]):
                    ax=axes[row,col]
                    if col==3:
                        ax.imshow(values.reshape(n,n),origin='lower',extent=(0,64,0,64),interpolation='nearest',cmap=ListedColormap(['#FFFFFF','#73A79B','#E2A447']),vmin=-.5,vmax=2.5)
                    else:
                        ax.imshow(values.reshape(n,n),origin='lower',extent=(0,64,0,64),interpolation='nearest',cmap='Blues',vmin=0,vmax=1)
                    ax.set_title(title);ax.set_xticks([0,32,64]);ax.set_yticks([0,32,64])
            save(fig,'within_parent_heterogeneity')
            member=s['weights']>0
            spatial=np.zeros((16,16));jaccard=np.zeros_like(spatial);cosine=np.zeros_like(spatial)
            weights=s['weights'].astype(float)
            for k in range(16):
                for j in range(16):
                    spatial[k,j]=(support[:,k]&support[:,j]).sum()/max(1,(support[:,k]|support[:,j]).sum())
                    jaccard[k,j]=(member[:,k]&member[:,j]).sum()/(member[:,k]|member[:,j]).sum()
                    cosine[k,j]=np.dot(weights[:,k],weights[:,j])/(np.linalg.norm(weights[:,k])*np.linalg.norm(weights[:,j]))
            fig,axes=plt.subplots(1,3,figsize=(15,6.5),layout='constrained',sharey=True)
            for ax,values,title in zip(axes,[spatial,jaccard,cosine],['Spatial support IoU','Contributor-set Jaccard','Contribution-weight cosine']):
                im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
                ax.set_xticks(range(16),ids,rotation=90,fontsize=8);ax.set_yticks(range(16),ids,fontsize=8);ax.set_title(title)
                pd.DataFrame(values,index=ids,columns=ids).to_csv(out/(title.lower().replace(' ','_')+'.csv'))
            fig.colorbar(im,ax=axes,label='Similarity (0–1)',shrink=.7,fraction=.025)
            save(fig,'spatial_and_gene_similarity')
            pairs=[]
            for k in range(16):
                for j in range(k+1,16):
                    if (support[:,k]&support[:,j]).any():
                        pairs.append({'a':ids[k],'b':ids[j],'spatial_overlap_spots':int((support[:,k]&support[:,j]).sum()),'gene_jaccard':jaccard[k,j],'weight_cosine':cosine[k,j]})
            pd.DataFrame(pairs).to_csv(out/'spatially_overlapping_gene_similarity.csv',index=False)
        print(sizes.to_string(index=False))

LABELS = {"adjacent": "Adjacent", "partial_overlap": "Partial overlap", "nested": "Nested",
          "separated_recurrence": "Separated recurrence"}


def plot_simulation(root: Path, vector: bool = False) -> None:
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False}):
        h5ad_path = input_h5ad_path(root)
        root = root / "simulation"
        c = json.loads((root / "config.json").read_text())
        summary = json.loads((root / "qc/summary.json").read_text())
        t = np.load(root / "truth/arrays.npz", allow_pickle=False)
        genes = pd.read_csv(root / "truth/genes.csv", keep_default_na=False)
        profiles = pd.read_csv(root / "qc/core_profiles.csv", index_col=0)
        data = ad.read_h5ad(h5ad_path)
        counts = data.X.toarray()
        out = root / "qc/figures"
        out.mkdir(exist_ok=False)
        n = c["nside"]
        show_truth_contours = "regional_enhancement" not in c

        def save(fig: plt.Figure, stem: str) -> None:
            for panel_ax in fig.axes:
                for panel_title in (panel_ax.title, panel_ax._left_title, panel_ax._right_title):
                    label = panel_title.get_text()
                    if len(label) > 2 and label[0].islower() and label[1:3] == "  ":
                        panel_title.set_text(r"$\mathbf{" + label[0] + "}$" + label[1:])
            fig.savefig(out / f"{stem}.png", dpi=300)
            if vector:
                fig.savefig(out / f"{stem}.svg")
                fig.savefig(out / f"{stem}.pdf")

        # Full layout on every panel; never crop away other potential activity locations.
        for filename, values, label in [
            ("true_activity_fields", t["activation"], "True activity intensity"),
            ("observed_core_expression", profiles.to_numpy(), "Core mean counts / technical depth"),
        ]:
            fig, axes = plt.subplots(4, 4, figsize=(10.8, 10.4), layout="constrained", sharex=True, sharey=True)
            maximum = float(values.max())
            for k, ax in enumerate(axes.flat):
                image = ax.imshow(values[:, k].reshape(n, n), origin="lower", extent=(0, 64, 0, 64),
                                  cmap="Blues", vmin=0, vmax=maximum, interpolation="nearest")
                if show_truth_contours:
                    ax.contour((t["instances"][:, k] > 0).reshape(n, n), levels=[0.5], colors="#333333",
                               linewidths=0.6, origin="lower", extent=(0, 64, 0, 64))
                spec = c["activities"][k]
                ax.set_title(f"{spec['id']}  |  {LABELS.get(spec['relation'], spec['relation'].replace('_', ' '))}", fontsize=9)
                ax.set_xticks([0, 32, 64])
                ax.set_yticks([0, 32, 64])
                if k % 4 == 0:
                    ax.set_ylabel("y (simulation units)")
                if k >= 12:
                    ax.set_xlabel("x (simulation units)")
            fig.colorbar(image, ax=axes, label=label, shrink=0.65, fraction=0.025)
            save(fig, filename)
            plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.3), layout="constrained")
        ax = axes[0, 0]
        classes = ["specific_core", "pair_shared", "group_shared"]
        order = np.concatenate([part[np.argsort(np.argmax(t["weights"][part], axis=1), kind="stable")]
                                for part in [np.flatnonzero(genes.category.to_numpy() == category) for category in classes]])
        image = ax.imshow(np.log1p(t["weights"][order].T), cmap="Blues", aspect="auto", interpolation="nearest")
        ax.set_title("a  Shared and unequal gene contributions")
        ax.set_yticks(np.arange(16), t["activity_ids"], fontsize=7)
        lengths = np.array([(genes.category == category).sum() for category in classes])
        centers = np.cumsum(lengths) - lengths / 2
        ax.set_xticks(centers, ["Specific", "Pair", "Group"], fontsize=8, rotation=45, ha="right")
        for edge in np.cumsum(lengths)[:-1]:
            ax.axvline(edge - 0.5, color="#555555", lw=0.6)
        fig.colorbar(image, ax=ax, label="log1p contribution", fraction=0.045)
        ax = axes[0, 1]
        signal = pd.read_csv(root / "qc/activity_signal.csv")
        ax.bar(np.arange(16), signal.core_log2fc, color="#648AA8")
        ax.set_xticks(np.arange(16), t["activity_ids"], rotation=90, fontsize=7)
        ax.set_ylabel("Core inside/outside log2FC")
        ax.set_title("b  Activity-wise regional expression")
        ax = axes[0, 2]
        image = ax.imshow(t["depth"].reshape(n, n), origin="lower", extent=(0, 64, 0, 64), cmap="Greys")
        ax.set_title("c  Independent technical depth multiplier")
        ax.set_xlabel("x (simulation units)")
        ax.set_ylabel("y (simulation units)")
        fig.colorbar(image, ax=ax, label="Depth multiplier", fraction=0.045)
        ax = axes[1, 0]
        n_active = (t["instances"] > 0).sum(axis=1)
        active_colors = ["#BBBBBB", "#648AA8", "#C49364", "#65A99A", "#9C83B7", "#A89672", "#C97578"]
        for value in np.unique(n_active):
            color = active_colors[int(value) % len(active_colors)]
            mask = n_active == value
            ax.scatter(t["depth"][mask], counts.sum(axis=1)[mask], s=3, alpha=0.3,
                       color=color, label=f"{value} active")
        ax.set_xlabel("Technical depth multiplier")
        ax.set_ylabel("Total observed counts")
        ax.set_title("d  Depth and activity both change total counts")
        ax.legend(markerscale=3, fontsize=8, frameon=False)
        ax = axes[1, 1]
        background = pd.read_csv(root / "qc/background_moran.csv")
        ax.hist(background.moran_i_depth_corrected, bins=40, color="#8C9CA7", edgecolor="white")
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlabel("Moran I, depth-corrected background")
        ax.set_ylabel("Genes")
        ax.set_title("e  All non-regional genes")
        ax = axes[1, 2]
        instance_signal = pd.read_csv(root / "qc/instance_signal.csv")
        for k, activity_id in enumerate(t["activity_ids"]):
            part = instance_signal[instance_signal.activity_id == activity_id]
            offsets = np.linspace(-0.18, 0.18, len(part)) if len(part) > 1 else [0]
            ax.scatter(k + np.array(offsets), part.core_log2fc, s=22, color="#648AA8")
        ax.set_xticks(np.arange(16), t["activity_ids"], rotation=90, fontsize=7)
        ax.set_ylabel("Instance core / outside log2FC")
        ax.set_title(f"f  All {len(instance_signal)} connected instances")
        save(fig, "generation_overview")
        plt.close(fig)
        if "program_baseline_classes" in c:
            # Same activity, three baseline classes; selection uses truth baseline, never observed appearance.
            fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.5), layout="constrained")
            selection_rows = []
            for row, tier in enumerate(c["program_baseline_classes"]):
                candidates = np.flatnonzero(t["specific"][:, 0] & (genes.baseline_class.to_numpy() == tier["name"]))
                j = candidates[np.argsort(t["baseline"][candidates], kind="stable")[len(candidates) // 2]]
                raw = counts[:, j]
                corrected = raw / t["depth"]
                expected = t["expected_counts"][:, j] / t["depth"]
                inside = t["instances"][:, 0] > 0
                selection_rows.append({"activity_id": "ACT01", "baseline_class": tier["name"],
                                       "gene_id": t["gene_ids"][j], "baseline": float(t["baseline"][j]),
                                       "loading": float(t["weights"][j, 0]),
                                       "outside_detection": float((raw[~inside] > 0).mean()),
                                       "inside_mean_corrected": float(corrected[inside].mean()),
                                       "outside_mean_corrected": float(corrected[~inside].mean())})
                for col, (values, heading) in enumerate([(expected, "Expected before technical depth"),
                                                         (raw, "Observed raw counts"),
                                                         (corrected, "Counts / technical depth")]):
                    ax = axes[row, col]
                    image = ax.imshow(values.reshape(n, n), origin="lower", extent=(0, 64, 0, 64),
                                      cmap="Blues", vmin=0, vmax=float(values.max()) * 1.01, interpolation="nearest")
                    if show_truth_contours:
                        ax.contour(inside.reshape(n, n), levels=[0.5], colors="#333333", linewidths=0.6,
                                   origin="lower", extent=(0, 64, 0, 64))
                    ax.set_title(heading, fontsize=9)
                    ax.set_xticks([0, 32, 64])
                    ax.set_yticks([0, 32, 64])
                    if col == 0:
                        ax.set_ylabel(f"{tier['name'].capitalize()} baseline | {t['gene_ids'][j]}\ny (simulation units)")
                    if row == 2:
                        ax.set_xlabel("x (simulation units)")
                    fig.colorbar(image, ax=ax, fraction=0.046)
            save(fig, "baseline_expression_examples")
            plt.close(fig)
            pd.DataFrame(selection_rows).to_csv(out / "baseline_example_genes.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
            axes[0].bar(np.arange(16), signal.n_contributing_genes, color="#648AA8")
            axes[0].set_xticks(np.arange(16), t["activity_ids"], rotation=90, fontsize=8)
            axes[0].set_ylabel("Contributing genes")
            axes[0].set_title("a  Different program sizes")
            tiers = pd.read_csv(root / "qc/baseline_signal.csv")
            for tier, color in [("low", "#A0BCCF"), ("medium", "#648AA8"), ("high", "#334E68")]:
                part = tiers[tiers.baseline_class == tier]
                axes[1].scatter(part.outside_mean, part.inside_mean, s=28, color=color, label=tier)
            maximum = float(tiers[["inside_mean", "outside_mean"]].to_numpy().max()) * 1.05
            axes[1].plot([0, maximum], [0, maximum], color="#999999", lw=0.8, ls="--")
            axes[1].set_xlim(0, maximum)
            axes[1].set_ylim(0, maximum)
            axes[1].set_xlabel("Outside mean counts / depth")
            axes[1].set_ylabel("Inside mean counts / depth")
            axes[1].set_title("b  All 16 activities, each baseline class")
            axes[1].legend(title="Baseline", frameon=False)
            save(fig, "heterogeneous_membership_and_baseline")
            plt.close(fig)
        if "regional_enhancement" in c:
            effects = pd.read_csv(root / "qc/contributor_gene_signal.csv")
            effects = effects.set_index("gene_id")
            fig, axes = plt.subplots(4, 4, figsize=(10.8, 10.8), layout="constrained")
            worst_rows = []
            for k, ax in enumerate(axes.flat):
                candidates = np.flatnonzero(t["specific"][:, k])
                scores = effects.loc[t["gene_ids"][candidates], "raw_auc"].to_numpy()
                j = candidates[np.argmin(scores)]
                values = counts[:, j]
                image = ax.imshow(values.reshape(n, n), origin="lower", extent=(0, 64, 0, 64),
                                  cmap="Blues", vmin=0, vmax=float(values.max()), interpolation="nearest")
                # No ROI contour, smoothing, clipping or per-region image processing.
                ax.set_title(f"{t['activity_ids'][k]} | {t['gene_ids'][j]}\nRaw-count AUC {scores.min():.3f}", fontsize=8)
                ax.set_xticks([0, 32, 64])
                ax.set_yticks([0, 32, 64])
                if k % 4 == 0:
                    ax.set_ylabel("y (simulation units)")
                if k >= 12:
                    ax.set_xlabel("x (simulation units)")
                fig.colorbar(image, ax=ax, fraction=0.046)
                worst_rows.append({"activity_id": t["activity_ids"][k], "gene_id": t["gene_ids"][j],
                                   **effects.loc[t["gene_ids"][j]].to_dict()})
            save(fig, "weakest_single_genes_unframed")
            plt.close(fig)
            pd.DataFrame(worst_rows).to_csv(out / "weakest_single_genes.csv", index=False)
            fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.3), layout="constrained")
            shared_rows = []
            for row, category in enumerate(["pair_shared", "group_shared"]):
                for group in range(4):
                    ax = axes[row, group]
                    candidates = np.flatnonzero((genes.category.to_numpy() == category) & (t["weights"][:, group * 4] > 0))
                    j = candidates[np.argsort(t["baseline"][candidates], kind="stable")[len(candidates) // 2]]
                    parents = np.flatnonzero(t["weights"][j] > 0)
                    values = counts[:, j]
                    image = ax.imshow(values.reshape(n, n), origin="lower", extent=(0, 64, 0, 64),
                                      cmap="Blues", vmin=0, vmax=float(values.max()), interpolation="nearest")
                    parent_label = ", ".join(t["activity_ids"][parents]) if row == 0 else (
                        f"{t['activity_ids'][parents[0]]}–{t['activity_ids'][parents[-1]]}")
                    ax.set_title(f"{len(parents)} activities | {parent_label}\n{t['gene_ids'][j]}", fontsize=9)
                    ax.set_xticks([0, 32, 64])
                    ax.set_yticks([0, 32, 64])
                    if group == 0:
                        ax.set_ylabel("Pair-shared\ny" if row == 0 else "Four-activity shared\ny")
                    if row == 1:
                        ax.set_xlabel("x (simulation units)")
                    fig.colorbar(image, ax=ax, fraction=0.046)
                    shared_rows.append({"gene_id": t["gene_ids"][j], "category": category,
                                        "activities": ";".join(t["activity_ids"][parents]),
                                        "baseline": float(t["baseline"][j])})
            save(fig, "shared_gene_examples_unframed")
            plt.close(fig)
            pd.DataFrame(shared_rows).to_csv(out / "shared_gene_example_genes.csv", index=False)
        (out / "README.md").write_text(
            "# Generation review figures\n\n"
            "Python/matplotlib review PNGs at 300 dpi; these are not final manuscript exports.\n"
            f"All 16 activities, {sum(len(a['instances']) for a in c['activities'])} instances and non-regional genes are included.\n"
            "No exclusions, fitted components, biological replicates or significance tests.\n"
            "The two spatial grids use one shared linear color scale per figure; no per-panel rescaling.\n"
            "The observed grid uses all specific core genes of each activity. Enhanced baselines have no truth contours.\n"
            "Technical depth is used only for generation QC; it is absent from method inputs.\n"
            "Sources: truth/arrays.npz; truth/genes.csv; qc/activity_signal.csv; "
            "qc/instance_signal.csv; qc/core_profiles.csv; qc/background_moran.csv.\n"
            "Coordinates are arbitrary simulation units, not micrometers.\n"
            "Baseline examples use one median-baseline ACT01 specific member per class; all locations retained.\n"
            "Single-gene example grids use separate scales per panel with explicit colorbars, no clipping.\n"
            "The size/baseline summary includes all 16 activities and all 48 activity-by-baseline groups.\n"
            "weakest_single_genes_unframed selects each activity's lowest raw-count AUC specific member.\n"
            "Enhanced baseline single-gene panels are unsmoothed raw data and never show truth region boundaries.\n"
            "Shared examples select a median-baseline gene for one pair and the four-activity set in each membership group.\n"
        )
        print(out)

def plot_member_heterogeneity(root: Path, *, vector: bool = False) -> None:
    """All activities, fixed member selection, full unsmoothed expression fields.

    Image-plate QC, not a journal-sized figure: six members per activity,
    selected by baseline/category before examining their expression. Every
    panel uses a separate 0-to-peak scale with its peak shown. Observations
    use log1p color, while latent enhancements use linear color. Shared
    genes' observed counts include every owning activity and their baseline.
    """
    simulation = root / "simulation"
    c = json.loads((simulation / "config.json").read_text())
    if not c.get("spatial_heterogeneity", {}).get("enabled", False):
        return
    t = np.load(simulation / "truth/arrays.npz", allow_pickle=False)
    h = np.load(simulation / "truth/gene_activity_fields.npz", allow_pickle=False)
    fields, field_genes, field_activities = h["fields"], h["gene_indices"], h["activity_indices"]
    genes = pd.read_csv(simulation / "truth/genes.csv", keep_default_na=False)
    counts = ad.read_h5ad(input_h5ad_path(root)).X.toarray()
    corrected = counts / t["depth"][:, None]
    out = simulation / "qc/member_heterogeneity"
    out.mkdir(exist_ok=False)
    n = c["nside"]
    selections, summaries, members = [], [], []
    for k, activity_id in enumerate(t["activity_ids"]):
        selected = []
        for baseline in ["low", "medium", "high"]:
            candidates = np.flatnonzero(t["specific"][:, k] & (genes.baseline_class.to_numpy() == baseline))
            selected.append(int(candidates[np.argsort(t["baseline"][candidates], kind="stable")[len(candidates)//2]]))
        for category in ["specific_core", "pair_shared", "group_shared"]:
            candidates = np.flatnonzero((t["weights"][:, k] > 0) & (genes.category.to_numpy() == category))
            candidates = np.array([j for j in candidates if j not in selected])
            selected.append(int(candidates[np.argsort(t["baseline"][candidates], kind="stable")[len(candidates)//2]]))
        columns = np.flatnonzero(field_activities == k)
        column_for_gene = dict(zip(field_genes[columns], columns))
        mask = t["instances"][:, k] > 0
        profiles = fields[mask][:, columns].astype(np.float64)
        correlations = np.corrcoef(profiles.T)
        off_diagonal = correlations[np.triu_indices(len(columns), 1)]
        summaries.append({"activity_id": activity_id, "n_members": len(columns),
                          "median_member_pair_correlation_inside": float(np.median(off_diagonal)),
                          "p10_member_pair_correlation_inside": float(np.quantile(off_diagonal, .1)),
                          "p90_member_pair_correlation_inside": float(np.quantile(off_diagonal, .9))})
        for col in columns:
            j = int(field_genes[col])
            field = fields[:, col]
            members.append({"activity_id": activity_id, "gene_id": t["gene_ids"][j],
                            "outside_enhancement_fraction": float(field[~mask].sum()/field.sum()),
                            "depth_corrected_inside_mean": float(corrected[mask, j].mean()),
                            "depth_corrected_outside_mean": float(corrected[~mask, j].mean())})
        for j in selected:
            selections.append({"activity_id": activity_id, "gene_id": t["gene_ids"][j],
                               "category": genes.category.iloc[j], "baseline_class": genes.baseline_class.iloc[j],
                               "all_owning_activities": genes.activities.iloc[j]})
        # Keep selection order and exact field columns for every exported panel.
        summaries[-1]["selected_genes"] = selected
        summaries[-1]["selected_columns"] = [int(column_for_gene[j]) for j in selected]

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42}):
        def panel(ax, values, title, *, log_color=False):
            peak = float(values.max())
            shown = np.log1p(values) if log_color else values
            ax.imshow(shown.reshape(n, n), origin="lower", extent=(0, 64, 0, 64), cmap="Blues",
                      vmin=0, vmax=max(float(shown.max()), 1e-12), interpolation="nearest")
            scale = "log1p color" if log_color else "linear color"
            ax.set_title(f"{title}\n{scale}; data range: 0–{peak:.1f}", fontsize=9)
            ax.set_xticks([0, 32, 64]); ax.set_yticks([0, 32, 64])

        def save(fig, stem):
            for panel_ax in fig.axes:
                for panel_title in (panel_ax.title, panel_ax._left_title, panel_ax._right_title):
                    label = panel_title.get_text()
                    if len(label) > 2 and label[0].islower() and label[1:3] == "  ":
                        panel_title.set_text(r"$\mathbf{" + label[0] + "}$" + label[1:])
            fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
            if vector:
                fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
                fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
            plt.close(fig)

        for k, row in enumerate(summaries):
            activity_id = row["activity_id"]
            fig, axes = plt.subplots(3, 6, figsize=(17, 8.7), layout="constrained")
            for col, (j, field_col) in enumerate(zip(row["selected_genes"], row["selected_columns"])):
                enhancement = fields[:, field_col] * t["weights"][j, k] * c["signal_scale"]
                label = f"{t['gene_ids'][j]} | {genes.baseline_class.iloc[j]}\n{genes.category.iloc[j]}"
                panel(axes[0, col], enhancement, label)
                panel(axes[1, col], counts[:, j], "Observed raw counts", log_color=True)
                panel(axes[2, col], corrected[:, j], "Counts / technical depth", log_color=True)
            axes[0, 0].set_ylabel("This activity's true enhancement\ny (simulation units)")
            axes[1, 0].set_ylabel("Raw observations\ny (simulation units)")
            axes[2, 0].set_ylabel("Depth-corrected observations\ny (simulation units)")
            for ax in axes[2]: ax.set_xlabel("x (simulation units)")
            for row_index, letter in enumerate("abc"):
                axes[row_index, 0].text(-.22, 1.04, letter, transform=axes[row_index, 0].transAxes,
                                        fontsize=13, fontweight="bold", ha="left", va="bottom", clip_on=False)
            save(fig, f"{activity_id}_members")
        for start in range(0, 16, 4):
            fig, axes = plt.subplots(4, 6, figsize=(17, 11.3), layout="constrained")
            for row_i, k in enumerate(range(start, start+4)):
                for col, j in enumerate(summaries[k]["selected_genes"]):
                    panel(axes[row_i, col], corrected[:, j],
                          f"{t['gene_ids'][j]} | {genes.baseline_class.iloc[j]}", log_color=True)
                axes[row_i, 0].set_ylabel(str(t["activity_ids"][k]))
            save(fig, f"members_ACT{start+1:02d}_ACT{start+4:02d}")
    pd.DataFrame(selections).to_csv(out / "selected_genes.csv", index=False)
    pd.DataFrame(members).to_csv(out / "all_member_statistics.csv", index=False)
    pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("selected_")}
                  for row in summaries]).to_csv(out / "activity_heterogeneity.csv", index=False)
    (out / "README.md").write_text(
        "# Member spatial heterogeneity QC\n\n"
        "All 16 activities, six members each: median-baseline specific members within low/medium/high "
        "classes, then a remaining specific member and a pair/group-shared member (median baseline). "
        "Selection never uses observed expression or recovery. All memberships are described in the CSV tables.\n\n"
        "Detailed plates show isolated true activity enhancement, raw counts, and counts/known technical depth. "
        "Shared genes' observations include other owning activities. The first row excludes baseline and observation noise. "
        "Every panel uses its own zero-to-maximum scale; peaks in original data units are printed. "
        "Latent enhancement uses linear color; observations use log1p color, not normalized log-expression. No clipping, smoothing, "
        "cropping or truth outlines are applied. Correlations describe latent within-region enhancement across "
        "members, not count correlation or model performance. Single simulated sample, no inference tests. "
        "PNG300dpi and optional editable PDF/SVG are large review plates, not final journal figures.\n")


def check(root: Path, *, vector: bool = False) -> None:
    """Write structural QC, descriptive expression statistics and both figure sets."""
    passed = _check_data(root)
    plot_layout(root)
    plot_simulation(root, vector=vector)
    plot_member_heterogeneity(root, vector=vector)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Sample directory containing simulation/.")
    parser.add_argument("--vector", action="store_true", help="Also export expression QC as SVG/PDF; layout always includes them.")
    args = parser.parse_args()
    check(args.root, vector=args.vector)
