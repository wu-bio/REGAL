"""Generate one observed high-resolution sample and count-conserving coarse grids.

Only this study's config and standalone layout module are inputs. Gene profiles,
latent spatial fields, depth and observation noise are generated once at 128x128.
The 64x64 and 32x32 samples are adjacent 2x2 sums, with fractional spatial truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter, label

if __package__:
    from . import generate_region_layout as layout
else:
    import generate_region_layout as layout

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs/resolution"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/simulation.json"
ACTIVITY_IDS = tuple(layout.ACTIVITY_GROUPS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mean_one_lognormal(rng: np.random.Generator, sd: float, size) -> np.ndarray:
    return rng.lognormal(-sd * sd / 2, sd, size=size)


def validate_config(config: dict) -> None:
    if config["high_nside"] != 128 or config["resolutions"] != [128, 64, 32]:
        raise ValueError("This experiment requires one 128 grid followed by 64 and 32 aggregation.")
    if config["n_genes"] != 5000:
        raise ValueError("This design uses 5,000 genes.")
    if len(config["activity_strengths"]) != len(ACTIVITY_IDS):
        raise ValueError("Each activity needs one strength.")
    if len(config["specific_genes_per_activity"]) != len(ACTIVITY_IDS):
        raise ValueError("Each activity needs one specific-gene count.")
    if min(config["activity_strengths"]) <= 0 or min(config["specific_genes_per_activity"]) <= 0:
        raise ValueError("Activity strengths and specific-gene counts must be positive.")
    for group in config["shared_gene_sets"]:
        ids = group["activities"]
        if len(ids) < 2 or len(set(ids)) != len(ids) or not set(ids) <= set(ACTIVITY_IDS):
            raise ValueError(f"Invalid shared-gene activity group: {ids}")
        if not isinstance(group["n_genes"], int) or group["n_genes"] <= 0:
            raise ValueError("Shared-gene counts must be positive integers.")
    if not np.isclose(sum(c["fraction"] for c in config["program_baseline_classes"]), 1):
        raise ValueError("Baseline fractions must sum to one.")
    if not 0 <= config["within_region_amplitude"] < 1 or config["nb_shape"] <= 0:
        raise ValueError("Invalid within-region amplitude or NB shape.")
    if config["target_mean_counts"] <= 0:
        raise ValueError("Expected counts must have a positive target mean.")


def build_spatial_truth(config: dict) -> dict:
    regions, coords = layout.generate_layout(config["high_nside"])
    instance_support, activity_support, overlaps = layout.build_supports(regions, coords)
    n_spots, n_activities = regions.size, len(ACTIVITY_IDS)
    activation = np.zeros((n_spots, n_activities), dtype=np.float32)
    x, y = coords.reshape(-1, 2).T
    for k, strength in enumerate(config["activity_strengths"]):
        texture = np.sin(x * 0.185 + k) * np.cos(y * 0.145 - k)
        mask = activity_support[:, :, k].ravel()
        activation[mask, k] = strength * (1 + config["within_region_amplitude"] * texture[mask])
    instance_activities = np.empty(len(layout.REGION_IDS), dtype="U3")
    for activity, ids in layout.ACTIVITY_GROUPS.items():
        for instance_id in ids:
            instance_activities[instance_id - 1] = activity
    for i in range(len(layout.REGION_IDS)):
        if label(instance_support[:, :, i])[1] != 1:
            raise ValueError("Each planned instance must be connected at the highest resolution.")
    return {
        "regions": regions.ravel(), "coordinates": coords.reshape(-1, 2),
        "activation": activation,
        "activity_fraction": activity_support.reshape(n_spots, n_activities).astype(np.float32),
        "instance_fraction": instance_support.reshape(n_spots, -1).astype(np.float32),
        "base_region_fraction": (regions.ravel()[:, None] == np.array(layout.REGION_IDS)).astype(np.float32),
        "overlap_fraction": overlaps.reshape(n_spots, -1).astype(np.float32),
        "instance_activity_ids": instance_activities,
    }


def build_gene_model(config: dict, spatial_truth: dict) -> dict:
    """Generate global profiles, stratified baselines and nonspatial backgrounds."""
    rng = np.random.default_rng(config["structure_seed"])
    n_genes, n_activities = config["n_genes"], len(ACTIVITY_IDS)
    weights = np.zeros((n_genes, n_activities), dtype=np.float32)
    specific = np.zeros_like(weights, dtype=bool)
    category = np.full(n_genes, "unstructured_background", dtype="U24")
    baseline_class = np.full(n_genes, "background", dtype="U12")
    baseline = config["unstructured_background_mean"] * mean_one_lognormal(
        rng, config["baseline_log_sd"], n_genes,
    )
    blocks = []
    cursor = 0
    groups = [([i], count, "specific_core") for i, count in enumerate(config["specific_genes_per_activity"])]
    groups += [([ACTIVITY_IDS.index(a) for a in g["activities"]], g["n_genes"], "shared")
               for g in config["shared_gene_sets"]]
    for activities, count, kind in groups:
        stop = cursor + count
        if stop >= n_genes:
            raise ValueError("Regional genes exceed the available gene budget.")
        for activity in activities:
            weights[cursor:stop, activity] = config["loading_mean"] * mean_one_lognormal(
                rng, config["loading_log_sd"], count,
            )
        if kind == "specific_core":
            specific[cursor:stop, activities[0]] = True
        category[cursor:stop] = kind
        blocks.append((cursor, stop))
        cursor = stop
    n_regional = cursor
    for start, stop in blocks:
        classes = config["program_baseline_classes"]
        allocation = np.array([c["fraction"] * (stop - start) for c in classes])
        numbers = np.floor(allocation).astype(int)
        numbers[np.argsort(-(allocation - numbers), kind="stable")[:stop - start - numbers.sum()]] += 1
        slots = rng.permutation(np.arange(start, stop))
        offset = 0
        for spec, count in zip(classes, numbers):
            chosen = slots[offset:offset + count]
            baseline[chosen] = spec["mean"] * mean_one_lognormal(rng, config["baseline_log_sd"], count)
            baseline_class[chosen] = spec["name"]
            offset += count
    broad_end = n_regional + config["broad_background_genes"]
    null_end = broad_end + config["nonspatial_factors"] * config["genes_per_nonspatial_factor"]
    if null_end >= n_genes:
        raise ValueError("No room remains for independent background genes.")
    category[n_regional:broad_end] = "broad_background"
    baseline[n_regional:broad_end] = config["broad_background_mean"] * mean_one_lognormal(
        rng, config["baseline_log_sd"], config["broad_background_genes"],
    )
    nonspatial_weights = np.zeros((n_genes, config["nonspatial_factors"]), dtype=np.float32)
    for k in range(config["nonspatial_factors"]):
        start = broad_end + k * config["genes_per_nonspatial_factor"]
        stop = start + config["genes_per_nonspatial_factor"]
        category[start:stop] = "nonspatial_background"
        nonspatial_weights[start:stop, k] = config["nonspatial_loading_mean"] * mean_one_lognormal(
            rng, config["loading_log_sd"], stop - start,
        )
    enhancement = config["regional_enhancement"]
    for k, activity in enumerate(ACTIVITY_IDS):
        members = weights[:, k] > 0
        active = spatial_truth["activation"][:, k]
        minimum = float(active[active > 0].min())
        local_floor = config["local_multiplier_min"] if len(layout.ACTIVITY_GROUPS[activity]) > 1 else 1
        increment = np.maximum(
            enhancement["minimum_expected_increment"],
            (enhancement["minimum_expected_fold"] - 1) * baseline[members],
        )
        floor = increment / (minimum * local_floor)
        spread = 1 + enhancement["weight_spread"] * weights[members, k] / config["loading_mean"]
        weights[members, k] = np.maximum(weights[members, k], floor * spread * 1.000001)
    order = rng.permutation(n_genes)
    return {
        "gene_ids": np.array([f"SIM_G{i + 1:05d}" for i in range(n_genes)]),
        "weights": weights[order], "specific": specific[order], "baseline": baseline[order],
        "nonspatial_weights": nonspatial_weights[order], "category": category[order],
        "baseline_class": baseline_class[order],
    }


def build_regional_expression(config: dict, truth: dict, genes: dict) -> tuple[np.ndarray, dict, list[dict]]:
    """Add gene-specific within-support variation; never extend declared supports."""
    rng = np.random.default_rng(config["heterogeneity_seed"])
    instance_rng = np.random.default_rng(config["instance_seed"])
    nside, n_genes = config["high_nside"], config["n_genes"]
    n_spots = nside * nside
    regional = np.zeros((n_spots, n_genes), dtype=np.float32)
    memberships = np.argwhere(genes["weights"] > 0)
    fields = np.zeros((n_spots, len(memberships)), dtype=np.float32)
    gene_indices = np.empty(len(memberships), dtype=np.int32)
    activity_indices = np.empty(len(memberships), dtype=np.int16)
    spec = config["spatial_heterogeneity"]
    x, y = truth["coordinates"].T
    local_rows, cursor = [], 0
    for k, activity in enumerate(ACTIVITY_IDS):
        print(f"Building latent fields: {activity}", flush=True)
        members = np.flatnonzero(genes["weights"][:, k] > 0)
        instance_ids = layout.ACTIVITY_GROUPS[activity]
        changed = set()
        if len(instance_ids) > 1:
            n_changed = round(len(members) * config["local_deviation_fraction"])
            changed = set(instance_rng.choice(members, n_changed, replace=False))
        mask = truth["activity_fraction"][:, k] > 0
        inside = np.flatnonzero(mask)
        for gene in members:
            base = truth["activation"][:, k].astype(np.float64).copy()
            if gene in changed:
                for instance_id in instance_ids:
                    multiplier = instance_rng.uniform(config["local_multiplier_min"], config["local_multiplier_max"])
                    base[truth["instance_fraction"][:, instance_id - 1] > 0] *= multiplier
                    local_rows.append({"activity_id": activity, "instance_id": f"R{instance_id:02d}",
                                       "gene_id": genes["gene_ids"][gene], "multiplier": float(multiplier)})
            smooth = gaussian_filter(rng.standard_normal((nside, nside)), sigma=spec["correlation_length"])
            smooth = (smooth - smooth.ravel()[mask].mean()) / max(smooth.ravel()[mask].std(), 1e-12)
            preference = np.exp(spec["preference_log_sd"] * np.clip(smooth.ravel(), -2, 2))
            patch = np.ones(n_spots)
            if rng.random() < spec["patch_probability"]:
                center = rng.choice(inside)
                radius = rng.uniform(spec["patch_radius_min"], spec["patch_radius_max"])
                squared_distance = (x - x[center])**2 + (y - y[center])**2
                patch -= spec["patch_attenuation"] * np.exp(-squared_distance / (2 * radius**2))
            field = base * preference * patch
            field *= base.sum() / field.sum()
            fields[:, cursor] = field.astype(np.float32)
            regional[:, gene] += fields[:, cursor] * genes["weights"][gene, k]
            gene_indices[cursor], activity_indices[cursor] = gene, k
            if np.any(fields[~mask, cursor] != 0):
                raise RuntimeError("Gene-specific fields leaked outside declared activity support.")
            cursor += 1
    return regional, {"fields": fields, "gene_indices": gene_indices, "activity_indices": activity_indices}, local_rows


def observe_counts(config: dict, regional: np.ndarray, genes: dict) -> tuple[sparse.csr_matrix, dict]:
    """Sample observations once, consuming the regional matrix in place to limit memory."""
    depth_seed, null_seed, count_seed = np.random.SeedSequence(config["expression_seed"]).spawn(3)
    n_spots, n_genes = regional.shape
    depth = mean_one_lognormal(np.random.default_rng(depth_seed), config["depth_log_sd"], n_spots)
    null = np.random.default_rng(null_seed).binomial(
        1, config["nonspatial_probability"], (n_spots, config["nonspatial_factors"]),
    ).astype(np.float32)
    mean = regional
    mean += genes["baseline"][None, :]
    mean += null @ genes["nonspatial_weights"].T
    mean *= depth[:, None]
    scale = config["target_mean_counts"] / mean.sum(dtype=np.float64) * n_spots
    mean *= scale
    observation_rng = np.random.default_rng(config["observation_seed"])
    count_rng = np.random.default_rng(count_seed)
    matrices = []
    step = config["observation_gene_chunk_size"]
    for start in range(0, n_genes, step):
        stop = min(start + step, n_genes)
        conditional = mean[:, start:stop].astype(np.float64)
        conditional *= mean_one_lognormal(observation_rng, config["observation_log_sd"], conditional.shape)
        mean[:, start:stop] = conditional
        shape = config["nb_shape"]
        counts = count_rng.negative_binomial(shape, shape / (shape + conditional))
        if counts.max() > np.iinfo(np.int32).max:
            raise OverflowError("Counts do not fit int32.")
        matrices.append(sparse.csr_matrix(counts.astype(np.int32)))
    return sparse.hstack(matrices, format="csr"), {
        "expected_counts": mean, "depth_mean": depth, "nonspatial_activation": null,
        "global_expression_scale": np.array(scale),
    }


def coarse_operator(nside: int) -> sparse.csr_matrix:
    row, col = np.divmod(np.arange(nside * nside), nside)
    target = row // 2 * (nside // 2) + col // 2
    return sparse.csr_matrix((np.ones(nside * nside, dtype=np.int64), (target, np.arange(nside * nside))),
                             shape=((nside // 2)**2, nside * nside))


def aggregate_sample(sample: dict) -> dict:
    """Sum observed counts/conditional means; average intensive spatial truth."""
    operator = coarse_operator(sample["nside"])
    average = operator / 4
    summed = {"expected_counts"}
    mean_fields = ("coordinates", "activation", "activity_fraction", "instance_fraction", "base_region_fraction",
                   "overlap_fraction", "depth_mean", "nonspatial_activation")
    result = {"nside": sample["nside"] // 2, "counts": (operator @ sample["counts"]).tocsr()}
    for key in summed:
        result[key] = np.asarray(operator @ sample[key], dtype=np.float32)
    for key in mean_fields:
        result[key] = np.asarray(average @ sample[key], dtype=np.float32)
    result["gene_fields"] = dict(
        sample["gene_fields"], fields=np.asarray(average @ sample["gene_fields"]["fields"], dtype=np.float32),
    )
    result["global_expression_scale"] = sample["global_expression_scale"]
    return result


def count_summary(counts: sparse.csr_matrix) -> dict:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    detected = np.asarray((counts > 0).sum(axis=1)).ravel()
    return {"n_spots": counts.shape[0], "n_genes": counts.shape[1], "total_counts": int(counts.sum()),
            "mean_counts": float(totals.mean()), "counts_q10_q50_q90": np.quantile(totals, [.1, .5, .9]).tolist(),
            "detected_genes_q10_q50_q90": np.quantile(detected, [.1, .5, .9]).tolist(),
            "nonzero_fraction": float(counts.nnz / np.prod(counts.shape)),
            "zero_count_positions": int((totals == 0).sum())}


def write_sample(stage: Path, sample: dict, config: dict, genes: dict, spatial: dict, local_rows: list[dict]) -> dict:
    """Write model inputs without truth labels, and save all truth separately."""
    nside = sample["nside"]
    sample_id = f"{config['sample_prefix']}_{nside}"
    source_id = f"{config['sample_prefix']}_{config['high_nside']}"
    folder = stage / sample_id / "simulation"
    (folder / "truth").mkdir(parents=True)
    spot_ids = np.array([f"SIM_S{i + 1:05d}" for i in range(nside * nside)])
    obs = pd.DataFrame({"sample_id": sample_id}, index=pd.Index(spot_ids, name="spot_id"))
    var = pd.DataFrame({"gene_id": genes["gene_ids"], "gene_name": genes["gene_ids"]},
                       index=pd.Index(genes["gene_ids"], name="gene_id"))
    data = ad.AnnData(sample["counts"], obs=obs, var=var, obsm={"spatial": sample["coordinates"]})
    normalized = data.X.astype(np.float32)
    totals = np.asarray(normalized.sum(axis=1)).ravel()
    normalized = (sparse.diags(config["normalization_target_sum"] / np.maximum(totals, 1)) @ normalized).tocsr()
    normalized.data = np.log1p(normalized.data)
    data.layers["log1p_norm"] = normalized.astype(np.float32)
    data.uns["normalization"] = {"source": "X", "output_layer": "log1p_norm",
                                 "method": "normalize_total_then_log1p",
                                 "target_sum": config["normalization_target_sum"]}
    h5ad = stage / f"{sample_id}.h5ad"
    data.write_h5ad(h5ad, compression="gzip")
    area = (layout.SPATIAL_EXTENT / nside)**2
    truth_keys = ("activation", "activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction",
                  "expected_counts", "depth_mean", "nonspatial_activation", "global_expression_scale")
    np.savez_compressed(
        folder / "truth/arrays.npz", **{key: sample[key] for key in truth_keys},
        activation_sum=sample["activation"] * (config["high_nside"] / nside)**2,
        spot_ids=spot_ids, gene_ids=genes["gene_ids"], activity_ids=np.array(ACTIVITY_IDS),
        instance_ids=np.array([f"R{i:02d}" for i in layout.REGION_IDS]),
        instance_activity_ids=spatial["instance_activity_ids"], weights=genes["weights"], specific=genes["specific"],
        baseline=genes["baseline"], nonspatial_weights=genes["nonspatial_weights"],
    )
    np.savez_compressed(folder / "truth/gene_activity_fields.npz", **sample["gene_fields"])
    memberships = genes["weights"] > 0
    pd.DataFrame({
        "gene_id": genes["gene_ids"], "category": genes["category"], "baseline": genes["baseline"],
        "baseline_class": genes["baseline_class"],
        "activities": [";".join(np.array(ACTIVITY_IDS)[mask]) for mask in memberships],
    }).to_csv(folder / "truth/genes.csv", index=False)
    pd.DataFrame(local_rows).to_csv(folder / "truth/local_deviations.csv", index=False)
    pd.DataFrame([{
        "activity_id": activity, "n_instances": len(layout.ACTIVITY_GROUPS[activity]),
        "n_genes": int(memberships[:, k].sum()), "n_specific_core": int(genes["specific"][:, k].sum()),
        "fractional_occupied_positions": float(sample["activity_fraction"][:, k].sum()),
        "support_area_design_units": float(sample["activity_fraction"][:, k].sum() * area),
    } for k, activity in enumerate(ACTIVITY_IDS)]).to_csv(folder / "truth/activities.csv", index=False)
    pd.DataFrame([{
        "instance_id": f"R{instance_id:02d}", "activity_id": spatial["instance_activity_ids"][i],
        "fractional_occupied_positions": float(sample["instance_fraction"][:, i].sum()),
        "support_area_design_units": float(sample["instance_fraction"][:, i].sum() * area),
    } for i, instance_id in enumerate(layout.REGION_IDS)]).to_csv(folder / "truth/instances.csv", index=False)
    pd.DataFrame(sample["coordinates"], index=obs.index, columns=["x", "y"]).to_csv(folder / "coordinates.csv")
    high_row, high_col = np.divmod(np.arange(config["high_nside"]**2), config["high_nside"])
    factor = config["high_nside"] // nside
    target_index = high_row // factor * nside + high_col // factor
    pd.DataFrame({
        "high_resolution_spot_id": [f"SIM_S{i + 1:05d}" for i in range(config["high_nside"]**2)],
        "spot_id": spot_ids[target_index], "high_resolution_spot_index": np.arange(len(target_index)),
        "spot_index": target_index,
    }).to_csv(folder / "high_resolution_spot_mapping.csv", index=False)
    write_json(folder / "config.json", dict(config, sample_id=sample_id, nside=nside, high_resolution_sample=source_id))
    summary = count_summary(data.X)
    manifest = {
        "schema_version": "regal_resolution_simulation_v1", "sample_id": sample_id, "nside": nside,
        "high_resolution_sample": source_id, "n_activities": len(ACTIVITY_IDS), "n_instances": len(layout.REGION_IDS),
        "counts_mechanism": "one_negative_binomial_draw" if nside == config["high_nside"] else "adjacent_2x2_count_sum",
        "parent_sample": None if nside == config["high_nside"] else f"{config['sample_prefix']}_{nside * 2}",
        "truth_mechanism": "fractions and mean latent fields; conditional expected counts are summed",
        "expected_expression_formula": (
            "global_expression_scale * depth * (baseline + sum(fields * weights) + nonspatial) * observation_multiplier"
        ),
        "observation_noise": "independent mean-one lognormal multipliers at high resolution, before NB sampling",
        "spatial_extent": [0, 128, 0, 128], "gene_profiles_shared_across_resolutions": True,
        "global_expression_scale": float(sample["global_expression_scale"]), "counts_summary": summary,
        "source_hashes": {"generator": sha256(Path(__file__)), "layout": sha256(Path(layout.__file__))},
        "files": {**{str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()},
                  f"../../{sample_id}.h5ad": sha256(h5ad)},
    }
    write_json(folder / "manifest.json", manifest)
    return {"sample_id": sample_id, "nside": nside, **summary, "h5ad_sha256": sha256(h5ad)}


def validate_written_sample(stage: Path, sample: dict, config: dict, genes: dict) -> None:
    """Check full saved counts and compact truth arrays against generated values."""
    sample_id = f"{config['sample_prefix']}_{sample['nside']}"
    data = ad.read_h5ad(stage / f"{sample_id}.h5ad")
    if (data.X != sample["counts"]).nnz or data.X.dtype.kind not in "iu" or np.any(data.X.data < 0):
        raise RuntimeError("Saved integer counts differ from generated counts.")
    if not data.var_names.equals(pd.Index(genes["gene_ids"])) or not data.obs_names.is_unique:
        raise RuntimeError("Saved gene/position identifiers are inconsistent.")
    if not np.array_equal(data.obsm["spatial"], sample["coordinates"]):
        raise RuntimeError("Saved coordinates differ from generated coordinates.")
    if set(data.obs.columns) != {"sample_id"} or set(data.obsm) != {"spatial"}:
        raise RuntimeError("Spatial/activity truth leaked into model inputs.")
    if not np.isfinite(data.layers["log1p_norm"].data).all():
        raise RuntimeError("Normalized expression is nonfinite.")
    with np.load(stage / sample_id / "simulation/truth/arrays.npz") as truth:
        for key in ("activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction", "activation"):
            if not np.array_equal(truth[key], sample[key]):
                raise RuntimeError(f"Saved truth differs: {key}")
        if not np.array_equal(truth["weights"], genes["weights"]):
            raise RuntimeError("Gene profiles differ across resolutions.")
    for key in ("activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction"):
        if not np.all((sample[key] >= 0) & (sample[key] <= 1)):
            raise RuntimeError(f"Invalid coverage fractions: {key}")
    if not np.array_equal(sample["base_region_fraction"].sum(axis=1), np.ones(data.n_obs)):
        raise RuntimeError("Base-region coverage does not fill the whole spatial grid.")
    if np.any(sample["counts"].sum(axis=1) == 0):
        raise RuntimeError("Empty expression positions in generated data.")


def validate_aggregation(high: dict, parent: dict, coarse: dict) -> None:
    operator = coarse_operator(parent["nside"])
    if (operator @ parent["counts"] != coarse["counts"]).nnz:
        raise RuntimeError("Coarse counts do not equal neighboring raw-count sums.")
    if not np.array_equal(np.asarray(high["counts"].sum(axis=0)), np.asarray(coarse["counts"].sum(axis=0))):
        raise RuntimeError("Per-gene counts are not conserved.")
    ratio = (high["nside"] / coarse["nside"])**2
    for key in ("activity_fraction", "instance_fraction", "base_region_fraction", "overlap_fraction"):
        if not np.array_equal(high[key].sum(axis=0), coarse[key].sum(axis=0) * ratio):
            raise RuntimeError(f"Physical truth coverage is not conserved: {key}")


def run(config: dict, data_root: Path) -> dict:
    """Stage and verify all three samples before publishing new paths."""
    validate_config(config)
    targets = [data_root / f"{config['sample_prefix']}_{n}{suffix}"
               for n in config["resolutions"] for suffix in ("", ".h5ad")]
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    spatial = build_spatial_truth(config)
    genes = build_gene_model(config, spatial)
    regional, gene_fields, local_rows = build_regional_expression(config, spatial, genes)
    print("Drawing observed 128x128 counts once.", flush=True)
    counts, observation = observe_counts(config, regional, genes)
    high = {"nside": config["high_nside"], "counts": counts, "gene_fields": gene_fields, **spatial, **observation}
    middle = aggregate_sample(high)
    low = aggregate_sample(middle)
    validate_aggregation(high, high, middle)
    validate_aggregation(high, middle, low)
    reports = []
    with tempfile.TemporaryDirectory(prefix=".resolution-generation-", dir=data_root) as directory:
        stage = Path(directory)
        for sample in (high, middle, low):
            print(f"Saving and validating {sample['nside']}x{sample['nside']} sample.", flush=True)
            reports.append(write_sample(stage, sample, config, genes, spatial, local_rows))
            validate_written_sample(stage, sample, config, genes)
        for target in targets:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Target appeared while generating: {target}")
        for target in targets:
            (stage / target.name).rename(target)
    return {
        "samples": reports, "n_activities": len(ACTIVITY_IDS), "n_instances": len(layout.REGION_IDS),
        "n_unique_regional_genes": int((genes["weights"] > 0).any(axis=1).sum()),
        "n_gene_activity_memberships": int((genes["weights"] > 0).sum()),
        "high_expected_mean_counts_before_observation_noise": config["target_mean_counts"],
        "global_expression_scale": float(observation["global_expression_scale"]),
        "raw_count_aggregation_exact": True, "per_gene_counts_conserved": True,
        "physical_activity_and_instance_coverage_conserved": True, "saved_inputs_and_truth_validated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, help="Override the root for new sample directories and h5ads.")
    parser.add_argument(
        "--preflight", action="store_true", help="Validate config, layout and genes without sampling or writing.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    data_root = (args.data_root or Path(config["data_root"])).resolve()
    config["data_root"] = str(data_root)
    if args.preflight:
        spatial = build_spatial_truth(config)
        genes = build_gene_model(config, spatial)
        print(json.dumps({"preflight": "PASS", "n_spots_high": len(spatial["coordinates"]),
                          "n_activities": len(ACTIVITY_IDS), "n_instances": len(layout.REGION_IDS),
                          "n_genes": config["n_genes"],
                          "n_regional_genes": int((genes["weights"] > 0).any(axis=1).sum()),
                          "n_memberships": int((genes["weights"] > 0).sum()),
                          "targets": [str(data_root / f"{config['sample_prefix']}_{n}.h5ad")
                                      for n in config["resolutions"]]}, indent=2))
        return
    report = run(config, data_root)
    report_path = OUTPUT_ROOT / "generation_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
