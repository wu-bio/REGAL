"""Generate the REGAL 16-activity benchmark without fitting any analysis method."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import ndimage

if __package__:
    from .landscape_layout import ellipse_mask, primary_components, child_components
else:
    from landscape_layout import ellipse_mask, primary_components, child_components

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs/heterogeneous.json"


def gene_counts(c: dict, key: str, length: int) -> np.ndarray:
    values = np.asarray(c[key])
    if values.ndim == 0:
        values = np.repeat(values, length)
    if values.shape != (length,) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 1):
        raise ValueError(f"{key} requires {length} positive integer counts (or one shared count).")
    return values


def input_h5ad_path(root: Path) -> Path:
    """Resolve the h5ad beside the sample directory."""
    return root.parent / f"{root.name}.h5ad"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lognormal(rng: np.random.Generator, mean: float, sd: float, size: object) -> np.ndarray:
    return rng.lognormal(np.log(mean) - sd**2 / 2, sd, size=size)


def membership_groups(c: dict, key: str, size: int) -> list[list[int]]:
    ids = [a['id'] for a in c['activities']]
    groups = c.get(key, [ids[k:k+size] for k in range(0,16,size)])
    if len(groups) != 16//size or any(len(v) != size for v in groups) or sorted(sum(groups, [])) != sorted(ids):
        raise ValueError(f'{key} must partition all 16 activities into groups of {size}.')
    return [[ids.index(activity) for activity in group] for group in groups]


def validate_config(c: dict) -> None:
    if len(c["activities"]) != 16 or len(c["activity_strengths"]) != 16:
        raise ValueError("This design requires exactly 16 activities and strengths.")
    if [a["id"] for a in c["activities"]] != [f"ACT{k:02d}" for k in range(1, 17)]:
        raise ValueError("Activity IDs must be ACT01 through ACT16, in order.")
    partitions = c.get("heterogeneous_partitions", [])
    if (len(partitions) != 1 or partitions[0].get("parent") != "ACT06"
            or partitions[0].get("children") != ["ACT10"]
            or set(partitions[0]) != {"parent", "children", "axis", "boundary"}):
        raise ValueError("The fixed layout requires only the ACT06–ACT10 nested pair.")
    if c["landscape_layout"]["primary_activity_ids"] != [
            "ACT01", "ACT02", "ACT03", "ACT04", "ACT05", "ACT06", "ACT07", "ACT09"]:
        raise ValueError("The fixed layout requires eight backbone activities, including ACT09.")
    for activity in c["activities"]:
        expression = activity["expression"]
        if (not np.isfinite(expression["regional_scale"]) or expression["regional_scale"] <= 0
                or not np.isfinite(expression["baseline_scale"]) or expression["baseline_scale"] < 1):
            raise ValueError("Activity expression requires regional_scale>0 and baseline_scale>=1.")
    if not isinstance(c["nside"], int) or c["nside"] < 32:
        raise ValueError("nside must be an integer >=32 on the canonical 64-unit layout.")
    for key, length in [("specific_genes_per_activity", 16), ("shared_genes_per_pair", 8),
                        ("shared_genes_per_group", 4)]:
        gene_counts(c, key, length)
    regional_budget = sum(gene_counts(c, key, length).sum() for key, length in [
        ("specific_genes_per_activity", 16), ("shared_genes_per_pair", 8), ("shared_genes_per_group", 4)])
    if "expected_regional_genes" in c and regional_budget != c["expected_regional_genes"]:
        raise ValueError("The unique regional gene budget does not match expected_regional_genes.")
    positive = ["n_genes", "loading_mean", "broad_background_mean",
                "unstructured_background_mean", "nb_shape", "nonspatial_loading_mean", "signal_scale"]
    if any(c[key] <= 0 for key in positive) or min(c["activity_strengths"]) <= 0:
        raise ValueError("Means, sizes, signal strengths and NB shape must be positive.")
    if not 0 <= c["within_region_amplitude"] < 1:
        raise ValueError("within_region_amplitude must be in [0,1).")
    if not 0 <= c["local_deviation_fraction"] < 1:
        raise ValueError("local_deviation_fraction must be in [0,1).")
    if not 0 < c["local_multiplier_min"] <= c["local_multiplier_max"]:
        raise ValueError("Local multipliers must be positive and ordered.")
    if not 0 <= c["nonspatial_probability"] <= 1:
        raise ValueError("nonspatial_probability must be in [0,1].")
    for key in ["loading_log_sd", "baseline_log_sd", "depth_log_sd"]:
        if c[key] < 0:
            raise ValueError(f"{key} must be nonnegative.")
    if "program_baseline_classes" in c:
        classes = c["program_baseline_classes"]
        if (len({s["name"] for s in classes}) != len(classes)
                or not np.isclose(sum(s["fraction"] for s in classes), 1)
                or any(s["mean"] <= 0 or s["fraction"] <= 0 for s in classes)):
            raise ValueError("Baseline classes need unique names, positive means/fractions and fractions summing to 1.")
    elif c["program_baseline_mean"] <= 0:
        raise ValueError("program_baseline_mean must be positive.")
    if "regional_enhancement" in c:
        spec = c["regional_enhancement"]
        if (spec["minimum_expected_fold"] <= 1 or spec["minimum_expected_increment"] <= 0
                or spec["weight_spread"] < 0):
            raise ValueError("Regional enhancement requires fold>1, increment>0 and spread>=0.")
    noise = c.get("expression_noise", {})
    if not isinstance(noise, dict) or not isinstance(noise.get("enabled", False), bool):
        raise ValueError("expression_noise must be an object with a boolean enabled flag.")
    if noise.get("enabled", False):
        if (not isinstance(noise.get("seed"), int) or noise["seed"] < 0
                or not np.isfinite(noise.get("log_sd", np.nan)) or noise["log_sd"] < 0
                or not np.isfinite(noise.get("nb_shape", np.nan)) or noise["nb_shape"] <= 0):
            raise ValueError("Enabled expression noise requires seed>=0, log_sd>=0 and nb_shape>0.")
    heterogeneity = c.get("spatial_heterogeneity", {})
    if not isinstance(heterogeneity, dict) or not isinstance(heterogeneity.get("enabled", False), bool):
        raise ValueError("spatial_heterogeneity requires a boolean enabled flag.")
    if heterogeneity.get("enabled", False):
        if (not isinstance(heterogeneity.get("seed"), int) or heterogeneity["seed"] < 0
                or any(not np.isfinite(heterogeneity.get(key, np.nan)) or heterogeneity[key] <= 0
                       for key in ["correlation_length", "preference_log_sd", "boundary_width",
                                   "boundary_shift", "patch_radius_min", "patch_radius_max"])
                or not 0 <= heterogeneity.get("patch_probability", -1) <= 1
                or not 0 < heterogeneity.get("patch_attenuation", -1) < 1
                or heterogeneity["patch_radius_min"] > heterogeneity["patch_radius_max"]):
            raise ValueError("Invalid spatial heterogeneity seed, spatial scales or patch settings.")


def build_structure(c: dict) -> dict:
    """All biological truth depends only on structure_seed, never expression_seed."""
    validate_config(c)
    rng = np.random.default_rng(c["structure_seed"])
    n = c["nside"]
    axis = (np.arange(n) + 0.5) * 64 / n
    x, y = np.meshgrid(axis, axis)
    coords = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    activation = np.zeros((n * n, 16), dtype=np.float32)
    reference_activation = np.zeros_like(activation)
    instances = np.zeros((n * n, 16), dtype=np.int16)
    primary = primary_components(c, coords)
    children = child_components(c, coords, primary)
    for k, spec in enumerate(c["activities"]):
        for m, shape in enumerate(spec["instances"], 1):
            if "rectangle" in shape:
                x0, x1, y0, y1 = shape["rectangle"]
                mask = (coords[:, 0] >= x0) & (coords[:, 0] < x1)
                mask &= (coords[:, 1] >= y0) & (coords[:, 1] < y1)
            elif "disk" in shape:
                cx, cy, radius = shape["disk"]
                mask = (coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2 < radius**2
            elif "ellipse" in shape:
                mask = ellipse_mask(coords, shape)
            elif "child_component" in shape:
                mask = children[spec["id"]][int(shape["child_component"]) - 1]
            elif "landscape_component" in shape:
                mask = primary[spec["id"]][int(shape["landscape_component"]) - 1]
            else:
                raise ValueError(f"Unknown shape: {shape}")
            if "allowed_primary_ids" in shape:
                allowed = np.logical_or.reduce([
                    np.logical_or.reduce(primary[activity_id]) for activity_id in shape["allowed_primary_ids"]])
                mask &= allowed
            if not mask.any() or np.any(instances[mask, k]):
                raise ValueError(f"Empty or overlapping instances within {spec['id']}")
            instances[mask, k] = m
        mask = instances[:, k] > 0
        # Deterministic, bounded variation; it does not blur the support boundary.
        texture = np.sin(coords[:, 0] * 0.37 + k) * np.cos(coords[:, 1] * 0.29 - k)
        # ACT04 loadings use the common ACT04/ACT09 reference extent; its
        # actual activation remains restricted to ACT04's own support.
        reference_mask = (mask | np.logical_or.reduce(primary["ACT09"])) if spec["id"] == "ACT04" else mask
        reference_activation[reference_mask, k] = c["activity_strengths"][k] * (
            1 + c["within_region_amplitude"] * texture[reference_mask])
        activation[mask, k] = reference_activation[mask, k]
        activation[:, k] *= spec["expression"]["regional_scale"]

    g = c["n_genes"]
    weights = np.zeros((g, 16), dtype=np.float32)
    kind = np.full(g, "unstructured_background", dtype=object)
    owner = np.full(g, "", dtype=object)
    specific = np.zeros((g, 16), dtype=bool)
    cursor = 0

    def add_genes(ks: list[int], count: int, category: str) -> None:
        nonlocal cursor
        end = cursor + count
        if end > g:
            raise ValueError("Requested gene sets exceed n_genes.")
        for k in ks:
            weights[cursor:end, k] = lognormal(rng, c["loading_mean"], c["loading_log_sd"], count)
        kind[cursor:end] = category
        owner[cursor:end] = ";".join(c["activities"][k]["id"] for k in ks)
        if category == "specific_core":
            specific[cursor:end, ks[0]] = True
        cursor = end

    for k in range(16):
        add_genes([k], int(gene_counts(c, "specific_genes_per_activity", 16)[k]), "specific_core")
    for j, ks in enumerate(membership_groups(c, "shared_pair_groups", 2)):
        add_genes(ks, int(gene_counts(c, "shared_genes_per_pair", 8)[j]), "pair_shared")
    for j, ks in enumerate(membership_groups(c, "shared_activity_groups", 4)):
        add_genes(ks, int(gene_counts(c, "shared_genes_per_group", 4)[j]), "group_shared")
    regional_end = cursor
    broad_end = cursor + c["broad_background_genes"]
    null_factor_end = broad_end + c["nonspatial_factors"] * c["genes_per_nonspatial_factor"]
    if null_factor_end >= g:
        raise ValueError("Need room for broad, nonspatial-factor and independent background genes.")
    kind[cursor:broad_end] = "broad_background"
    baseline = lognormal(rng, c["unstructured_background_mean"], c["baseline_log_sd"], g)
    baseline_class = np.full(g, "background", dtype=object)
    if "program_baseline_classes" in c:
        # Stratify every membership block, so each activity contains all baseline levels.
        start = 0
        block_sizes = list(gene_counts(c, "specific_genes_per_activity", 16))
        block_sizes += list(gene_counts(c, "shared_genes_per_pair", 8))
        block_sizes += list(gene_counts(c, "shared_genes_per_group", 4))
        for size in block_sizes:
            allocation = np.array([s["fraction"] * size for s in c["program_baseline_classes"]])
            ns = np.floor(allocation).astype(int)
            remainder = int(size - ns.sum())
            ns[np.argsort(-(allocation - ns), kind="stable")[:remainder]] += 1
            slots = rng.permutation(np.arange(start, start + size))
            offset = 0
            for spec, count in zip(c["program_baseline_classes"], ns):
                chosen = slots[offset:offset + count]
                baseline[chosen] = lognormal(rng, spec["mean"], c["baseline_log_sd"], count)
                baseline_class[chosen] = spec["name"]
                offset += count
            start += size
    else:
        baseline[:regional_end] = lognormal(rng, c["program_baseline_mean"], c["baseline_log_sd"], regional_end)
        baseline_class[:regional_end] = "low"
    baseline[regional_end:broad_end] = lognormal(
        rng, c["broad_background_mean"], c["baseline_log_sd"], c["broad_background_genes"])
    # Calibrate gene loadings on a common reference expression scale. Activity
    # expression settings define the final baseline and activation, not the
    # reference used for loading floors; this preserves the fixed benchmark.
    reference_baseline = baseline.copy()
    gene_baseline_scale = np.ones(g, dtype=np.float32)
    for k, activity in enumerate(c["activities"]):
        members = weights[:, k] > 0
        gene_baseline_scale[members] = np.maximum(
            gene_baseline_scale[members], activity["expression"]["baseline_scale"])
    # A shared gene has one baseline: use the largest requested scale once.
    baseline *= gene_baseline_scale
    nonspatial_weights = np.zeros((g, c["nonspatial_factors"]), dtype=np.float32)
    for l in range(c["nonspatial_factors"]):
        start = broad_end + l * c["genes_per_nonspatial_factor"]
        end = start + c["genes_per_nonspatial_factor"]
        nonspatial_weights[start:end, l] = lognormal(
            rng, c["nonspatial_loading_mean"], c["loading_log_sd"], end - start)

    if "regional_enhancement" in c:
        enhancement = c["regional_enhancement"]
        for k, activity in enumerate(c["activities"]):
            members = weights[:, k] > 0
            minimum_activation = float(reference_activation[reference_activation[:, k] > 0, k].min())
            # Reserve the full possible local attenuation, including recurrent instances.
            local_floor = min(1.0, c["local_multiplier_min"]) if len(activity["instances"]) > 1 else 1.0
            increment = np.maximum(enhancement["minimum_expected_increment"],
                                   (enhancement["minimum_expected_fold"] - 1) * reference_baseline[members])
            floor = increment / (c["signal_scale"] * minimum_activation * local_floor)
            spread = 1 + enhancement["weight_spread"] * weights[members, k] / c["loading_mean"]
            weights[members, k] = np.maximum(weights[members, k], floor * spread * 1.000001)

    local_rows = []
    regional_mean = activation @ weights.T
    for k, spec in enumerate(c["activities"]):
        if len(spec["instances"]) == 1:
            continue
        genes = np.flatnonzero(weights[:, k] > 0)
        changed = rng.choice(genes, round(len(genes) * c["local_deviation_fraction"]), replace=False)
        for m in range(1, len(spec["instances"]) + 1):
            multipliers = rng.uniform(c["local_multiplier_min"], c["local_multiplier_max"], len(changed))
            mask = instances[:, k] == m
            delta = np.outer(activation[mask, k], weights[changed, k] * (multipliers - 1))
            regional_mean[np.ix_(mask, changed)] += delta
            for gene, multiplier in zip(changed, multipliers):
                local_rows.append({"activity_id": spec["id"], "instance_id": m,
                                   "gene_index": int(gene), "multiplier": float(multiplier)})

    # Randomize gene order to avoid blocks leaking program identity to methods.
    order = rng.permutation(g)
    inverse = np.argsort(order)
    gene_ids = np.array([f"SIM_G{j + 1:05d}" for j in range(g)])
    for row in local_rows:
        row["gene_id"] = gene_ids[inverse[row.pop("gene_index")]]
    return {"coords": coords, "activation": activation, "instances": instances,
            "weights": weights[order], "specific": specific[order], "baseline": baseline[order],
            "nonspatial_weights": nonspatial_weights[order], "regional_mean": regional_mean[:, order],
            "baseline_class": baseline_class[order], "gene_ids": gene_ids, "kind": kind[order],
            "owner": owner[order], "local_rows": local_rows}




def regional_expression(c: dict, s: dict) -> tuple[np.ndarray, dict | None]:
    """Optional gene-specific spatial variation; leave original streams untouched.

    Distances use the layout's 64-unit coordinate system. Each membership has
    its own smooth preference, optional attenuated patch and soft shifted edge.
    The total enhancement per gene/activity is conserved, including existing
    recurrent-instance deviations. Baselines and global loadings never change.
    Saved fields are activity contributions per unit gene loading, before noise.
    """
    spec = c.get("spatial_heterogeneity", {})
    if not spec.get("enabled", False):
        return s["regional_mean"], None
    rng = np.random.default_rng(spec["seed"])
    n, spacing = c["nside"], 64 / c["nside"]
    coords = s["coords"]
    gene_index = {g: j for j, g in enumerate(s["gene_ids"])}
    local = {(row["activity_id"], row["instance_id"], gene_index[row["gene_id"]]): row["multiplier"]
             for row in s["local_rows"]}
    regional = np.zeros_like(s["regional_mean"], dtype=np.float64)
    fields, gene_indices, activity_indices, rows = [], [], [], []
    for k, activity in enumerate(c["activities"]):
        mask = (s["instances"][:, k] > 0).reshape(n, n)
        inside = np.flatnonzero(mask.ravel())
        distance_in = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1]
        distance_out, nearest = ndimage.distance_transform_edt(~mask, sampling=spacing, return_indices=True)
        signed = np.where(mask, distance_in - spacing / 2, -(distance_out - spacing / 2))
        for j in np.flatnonzero(s["weights"][:, k] > 0):
            base = s["activation"][:, k].astype(np.float64).copy()
            for m in range(1, len(activity["instances"]) + 1):
                base[s["instances"][:, k] == m] *= local.get((activity["id"], m, j), 1.0)
            extended = base.reshape(n, n)[tuple(nearest)]
            shift = rng.uniform(-spec["boundary_shift"], spec["boundary_shift"])
            edge = np.clip(0.5 + (signed + shift) / (2 * spec["boundary_width"]), 0, 1)
            smooth = ndimage.gaussian_filter(rng.standard_normal((n, n)),
                                             sigma=spec["correlation_length"] / spacing)
            smooth = (smooth - smooth[mask].mean()) / max(smooth[mask].std(), 1e-12)
            preference = np.exp(spec["preference_log_sd"] * np.clip(smooth, -2, 2))
            patch = np.ones((n, n))
            has_patch = rng.random() < spec["patch_probability"]
            cx = cy = radius = 0.0
            if has_patch:
                cx, cy = coords[rng.choice(inside)]
                radius = rng.uniform(spec["patch_radius_min"], spec["patch_radius_max"])
                squared = ((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2).reshape(n, n)
                patch -= spec["patch_attenuation"] * np.exp(-squared / (2 * radius**2))
            field = (extended * edge * preference * patch).ravel()
            field *= base.sum() / field.sum()
            field = field.astype(np.float32)
            regional[:, j] += field * float(s["weights"][j, k])
            fields.append(field)
            gene_indices.append(j)
            activity_indices.append(k)
            rows.append({"activity_id": activity["id"], "gene_id": s["gene_ids"][j],
                         "boundary_shift": float(shift), "has_attenuated_patch": bool(has_patch),
                         "patch_x": float(cx), "patch_y": float(cy), "patch_radius": float(radius),
                         "outside_enhancement_fraction": float(field[~mask.ravel()].sum() / field.sum()),
                         "relative_total_change": float(field.sum(dtype=np.float64) / base.sum() - 1)})
    truth = {"fields": np.column_stack(fields), "gene_indices": np.array(gene_indices),
             "activity_indices": np.array(activity_indices), "rows": rows}
    return regional, truth


def observation_mean(c: dict, mean: np.ndarray) -> tuple[np.ndarray, float]:
    """Perturb observations only; independent draws do not consume the original streams.

    All spot/gene entries receive independent mean-one lognormal multipliers.
    No support mask or activity membership is consulted. The unperturbed mean
    is preserved in expectation over perturbation draws, not exactly per draw.
    """
    noise = c.get("expression_noise", {})
    if not noise.get("enabled", False):
        return mean, c["nb_shape"]
    multiplier = lognormal(np.random.default_rng(noise["seed"]), 1.0, noise["log_sd"], mean.shape)
    return mean * multiplier, noise["nb_shape"]


def generate(c: dict, root: Path) -> None:
    h5ad_path = root.parent / f"{root.name}.h5ad"
    if root.exists() or h5ad_path.exists():
        raise FileExistsError(f"Refusing to overwrite sample directory or h5ad: {root}")
    s = build_structure(c)
    regional_mean, heterogeneity = regional_expression(c, s)
    # Separate streams keep the depth/background draws fixed when only NB shape is changed.
    depth_ss, null_ss, count_ss = np.random.SeedSequence(c["expression_seed"]).spawn(3)
    depth = lognormal(np.random.default_rng(depth_ss), 1.0, c["depth_log_sd"], len(s["coords"]))
    u = np.random.default_rng(null_ss).binomial(
        1, c["nonspatial_probability"], (len(depth), c["nonspatial_factors"])).astype(np.float32)
    mean = c["signal_scale"] * regional_mean + s["baseline"][None, :] + u @ s["nonspatial_weights"].T
    mean = mean.astype(np.float64) * depth[:, None]
    mean, r = observation_mean(c, mean)
    counts = np.random.default_rng(count_ss).negative_binomial(r, r / (r + mean)).astype(np.int32)
    sample_root = root
    root = sample_root / "simulation"
    root.mkdir(parents=True, exist_ok=False)
    (root / "truth").mkdir()
    if heterogeneity is not None:
        np.savez_compressed(root / "truth/gene_activity_fields.npz",
                            **{key: heterogeneity[key] for key in ["fields", "gene_indices", "activity_indices"]})
        pd.DataFrame(heterogeneity["rows"]).to_csv(root / "truth/spatial_heterogeneity.csv", index=False)
    obs = pd.DataFrame(index=pd.Index([f"SIM_S{i + 1:05d}" for i in range(len(depth))], name="spot_id"))
    obs["sample_id"] = c["sample_id"]
    var = pd.DataFrame(index=pd.Index(s["gene_ids"], name="gene_id"))
    var["gene_id"] = s["gene_ids"]
    var["gene_name"] = s["gene_ids"]
    # No truth categories, labels, loadings, technical depth or seeds in the method input.
    data = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=var, obsm={"spatial": s["coords"]})
    target_sum = c.get("normalization_target_sum", 10000.0)
    normalized = data.X.astype(np.float32)
    totals = np.asarray(normalized.sum(axis=1)).ravel()
    normalized = sparse.diags(target_sum / np.maximum(totals, 1)) @ normalized
    normalized.data = np.log1p(normalized.data)
    data.layers["log1p_norm"] = normalized.tocsr().astype(np.float32)
    data.uns["normalization"] = {"source": "X", "output_layer": "log1p_norm",
                                 "method": "normalize_total_then_log1p", "target_sum": target_sum}
    data.write_h5ad(h5ad_path, compression="gzip")
    pd.DataFrame(s["coords"], index=obs.index, columns=["x", "y"]).to_csv(root / "coordinates.csv")
    np.savez_compressed(root / "truth/arrays.npz", spot_ids=obs.index.to_numpy(dtype=str),
                        gene_ids=s["gene_ids"], activity_ids=np.array([a["id"] for a in c["activities"]]),
                        activation=s["activation"], instances=s["instances"], weights=s["weights"],
                        specific=s["specific"], baseline=s["baseline"], depth=depth,
                        nonspatial_activation=u, nonspatial_weights=s["nonspatial_weights"],
                        expected_counts=mean.astype(np.float32))
    pd.DataFrame({"gene_id": s["gene_ids"], "category": s["kind"], "activities": s["owner"],
                  "baseline": s["baseline"], "baseline_class": s["baseline_class"]}).to_csv(
                      root / "truth/genes.csv", index=False)
    pd.DataFrame(s["local_rows"], columns=["activity_id", "instance_id", "multiplier", "gene_id"]).to_csv(
        root / "truth/local_deviations.csv", index=False)
    pd.DataFrame([{"activity_id": a["id"], "relation": a["relation"],
                   "n_instances": len(a["instances"]), "n_spots": int((s["instances"][:, k] > 0).sum()),
                   "n_genes": int((s["weights"][:, k] > 0).sum()),
                   "n_specific_core": int(s["specific"][:, k].sum())}
                  for k, a in enumerate(c["activities"])]).to_csv(root / "truth/activities.csv", index=False)
    write_json(root / "config.json", c)
    write_json(root / "manifest.json", {
        "schema_version": "regal_simulation_v2", "status": "generated",
        "h5ad_path": str(h5ad_path),
        "sample_id": c["sample_id"], "n_spots": len(depth), "n_genes": len(s["gene_ids"]), "n_activities": 16,
        "n_instances": sum(len(a["instances"]) for a in c["activities"]),
        "python": platform.python_version(), "numpy": np.__version__, "anndata": ad.__version__,
        "source_reference": "https://github.com/willtownes/nsf-paper/blob/main/simulations/sim.py",
        "generator_sha256": sha256(Path(__file__)),
        "files": {**{str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()},
                  f"../../{h5ad_path.name}": sha256(h5ad_path)}})
    print(json.dumps({"output_root": str(sample_root), "shape": list(counts.shape), "n_activities": 16,
                      "n_instances": sum(len(a["instances"]) for a in c["activities"])}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expression-seed", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if args.expression_seed is not None:
        c["expression_seed"] = args.expression_seed
        c["sample_id"] = c["sample_id"].rsplit("_seed", 1)[0] + f"_seed{args.expression_seed}"
    root = (args.output_root or Path(c["output_root"])).expanduser().resolve()
    c["output_root"] = str(root)
    validate_config(c)
    if root.exists() or (root.parent / f"{root.name}.h5ad").exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {root}")
    if args.preflight:
        s = build_structure(c)
        print(json.dumps({"output_root": str(root), "shape": [c["nside"]**2, c["n_genes"]],
                          "n_activities": 16, "n_instances": int(s["instances"].max(axis=0).sum()),
                          "minimum_instance_spots": min(int((s["instances"][:, k] == m).sum())
                              for k in range(16) for m in range(1, s["instances"][:, k].max() + 1)),
                          "status": "preflight_only_no_files_written"}))
        return
    generate(c, root)


if __name__ == "__main__":
    main()
