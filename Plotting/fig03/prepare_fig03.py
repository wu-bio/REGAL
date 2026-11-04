"""Compute figure source tables from configured inputs and saved model results."""
from __future__ import annotations
import argparse
from string import Template
from copy import deepcopy
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_CONFIG: dict
_CONFIG_DIR: Path
_ROOTS: dict

from plot_fig03 import FIGURE_SELECTION

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig03'
TABLE_NAMES = ('domain_rsp_association.csv', 'positive_loading_profiles.csv', 'program_structure_metrics.csv', 'rsp_family_coherence.csv', 'rsp_family_members.csv', 'rsp_family_spatial_fields.csv', 'rsp_family_support.csv', 'rss_ablation_metrics.csv', 'spatial_gene_evidence.csv')

def _resolve(value):
    result = Path(Template(value).substitute(_ROOTS)).expanduser()
    return result if result.is_absolute() else _CONFIG_DIR / result

def path(name):
    return _resolve(_CONFIG['paths'][name])

def option(name):
    return _CONFIG['options'][name]

def write_table(frame, destination):
    destination = Path(destination)
    if destination.name in TABLE_NAMES:
        frame.to_csv(destination, index=False)

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Any
import yaml
import math
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

@dataclass(frozen=True)
class SpatialContext:
    """Minimal spatial and histology context shared by the R1 panels."""
    spot_ids: pd.Index
    obs: pd.DataFrame
    coordinates: np.ndarray
    image: np.ndarray
    image_coordinates: np.ndarray

def _as_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype.kind in {'u', 'i'}:
        array = array.astype(float) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(float)
        if np.nanmax(array) > 1.0:
            array = array / 255.0
    return np.clip(array, 0.0, 1.0)

def load_spatial_context(h5ad_path: str | Path) -> SpatialContext:
    """Load only the fields required to keep all R1 spatial maps aligned."""
    h5ad_path = Path(h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        if 'spatial' not in adata.obsm:
            raise KeyError(f"Missing adata.obsm['spatial'] in {h5ad_path}")
        spatial_entries = adata.uns.get('spatial', {})
        if not spatial_entries:
            raise KeyError(f"Missing adata.uns['spatial'] in {h5ad_path}")
        library_id = next(iter(spatial_entries))
        entry = spatial_entries[library_id]
        image = _as_float_image(np.asarray(entry['images']['hires']))
        image_scale = float(entry['scalefactors']['tissue_hires_scalef'])
        coordinates = np.asarray(adata.obsm['spatial'], dtype=float)[:, :2]
        spot_ids = pd.Index(adata.obs_names.astype(str), name='spot_id')
        obs = adata.obs.copy()
        obs.index = spot_ids
    finally:
        if adata.file is not None:
            adata.file.close()
    return SpatialContext(spot_ids=spot_ids, obs=obs, coordinates=coordinates, image=image, image_coordinates=coordinates * image_scale)

def load_log_normalized_expression(h5ad_path: str | Path, genes: Iterable[str]) -> pd.DataFrame:
    """Return library-normalized log1p expression for a requested gene subset."""
    requested = list(dict.fromkeys((str(gene) for gene in genes)))
    adata = ad.read_h5ad(h5ad_path)
    available = [gene for gene in requested if gene in adata.var_names]
    if not available:
        return pd.DataFrame(index=adata.obs_names.astype(str))
    library_size = np.asarray(adata.X.sum(axis=1)).reshape(-1).astype(float)
    matrix = adata[:, available].X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    normalized = np.log1p(matrix / np.maximum(library_size[:, None], 1.0) * 10000.0)
    return pd.DataFrame(normalized, index=pd.Index(adata.obs_names.astype(str), name='spot_id'), columns=available)

def load_raw_expression(h5ad_path: str | Path, genes: Iterable[str]) -> pd.DataFrame:
    """Return raw count columns for a requested gene subset."""
    requested = list(dict.fromkeys((str(gene) for gene in genes)))
    adata = ad.read_h5ad(h5ad_path)
    available = [gene for gene in requested if gene in adata.var_names]
    matrix = adata[:, available].X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(np.asarray(matrix, dtype=float), index=pd.Index(adata.obs_names.astype(str), name='spot_id'), columns=available)

@dataclass(frozen=True)
class ArtifactPaths:
    h5ad: Path
    sparkx: Path
    stagate: Path
    nmf: Path
    nnmf: Path
    stamp: Path
    regal: Path
    regal_rss_ablation: Path

def load_selection():
    selection = deepcopy(FIGURE_SELECTION)
    return selection

def artifact_paths(selection: dict[str, Any]) -> ArtifactPaths:
    return ArtifactPaths(**{name: path(name) for name in ('h5ad', 'sparkx', 'stagate', 'nmf', 'nnmf', 'stamp', 'regal', 'regal_rss_ablation')})

def _matrix_csv(path: Path, index_name: str) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = pd.Index(matrix.index.astype(str), name=index_name)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors='coerce').fillna(0.0)

def load_program_outputs(paths: ArtifactPaths, spot_ids: pd.Index) -> dict[str, dict[str, pd.DataFrame]]:
    """Load native loading matrices and continuous fields without rescaling them."""
    regal_loadings_long = pd.read_parquet(paths.regal / 'rsp_gene_loadings.parquet')
    regal_loadings = regal_loadings_long.pivot(index='gene', columns='rsp_id', values='loading').fillna(0.0)
    regal_fields_long = pd.read_parquet(paths.regal / 'rsp_activation.parquet')
    regal_fields = regal_fields_long.pivot(index='spot_id', columns='rsp_id', values='activation')
    outputs = {'NMF': {'loadings': _matrix_csv(paths.nmf / 'nmf_gene_loadings.csv', 'gene'), 'fields': _matrix_csv(paths.nmf / 'nmf_spot_weights.csv', 'spot_id')}, 'NNMF': {'loadings': _matrix_csv(paths.nnmf / 'nnmf_signatures.csv', 'gene'), 'fields': _matrix_csv(paths.nnmf / 'nnmf_weights.csv', 'spot_id')}, 'STAMP': {'loadings': _matrix_csv(paths.stamp / 'stamp_gene_modules.csv', 'gene'), 'fields': _matrix_csv(paths.stamp / 'stamp_spot_topics.csv', 'spot_id')}, 'REGAL': {'loadings': regal_loadings.apply(pd.to_numeric, errors='coerce').fillna(0.0), 'fields': regal_fields.apply(pd.to_numeric, errors='coerce').fillna(0.0)}}
    spot_ids = pd.Index(spot_ids.astype(str), name='spot_id')
    for (model, objects) in outputs.items():
        loadings = objects['loadings']
        fields = objects['fields']
        fields.index = pd.Index(fields.index.astype(str), name='spot_id')
        fields.columns = fields.columns.astype(str)
        missing_spots = spot_ids.difference(fields.index)
        if len(missing_spots):
            raise ValueError(f'{model} is missing {len(missing_spots)} shared spots')
        if set(loadings.columns) != set(fields.columns):
            raise ValueError(f'{model} loading and field component axes do not match')
        objects['fields'] = fields.loc[spot_ids, loadings.columns]
    return outputs

def load_sparkx(paths: ArtifactPaths) -> pd.DataFrame:
    table = pd.read_csv(paths.sparkx)
    required = {'gene', 'combined_pvalue', 'adjusted_pvalue', 'detected_spots'}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f'SPARK-X output is missing columns: {sorted(missing)}')
    table['gene'] = table['gene'].astype(str)
    return table

def load_stagate_domains(paths: ArtifactPaths, spot_ids: pd.Index) -> pd.Series:
    adata = ad.read_h5ad(paths.stagate, backed='r')
    try:
        if 'STAGATE_domain' not in adata.obs:
            raise KeyError("STAGATE output lacks obs['STAGATE_domain']")
        domains = adata.obs['STAGATE_domain'].astype(str).copy()
        domains.index = domains.index.astype(str)
    finally:
        if adata.file is not None:
            adata.file.close()
    missing = pd.Index(spot_ids.astype(str)).difference(domains.index)
    if len(missing):
        raise ValueError(f'STAGATE is missing {len(missing)} shared spots')
    return domains.loc[pd.Index(spot_ids.astype(str))].rename('STAGATE_domain')

def load_rsp_family_tables(paths: ArtifactPaths) -> dict[str, pd.DataFrame]:
    files = {'rsp': 'rsp.parquet', 'members': 'rsp_family_members.parquet', 'member_activation': 'rsp_family_member_activation.parquet', 'activation': 'rsp_activation.parquet', 'loadings': 'rsp_gene_loadings.parquet'}
    tables: dict[str, pd.DataFrame] = {}
    for (key, filename) in files.items():
        path = paths.regal / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        tables[key] = pd.read_parquet(path)
    return tables

def load_rss_ablation_metrics(paths: ArtifactPaths) -> pd.DataFrame:
    table = pd.read_csv(paths.regal_rss_ablation / 'component_metrics.csv')
    expected = {'model', 'component', 'top50_loading_fraction', 'local_contrast_z', 'program_score_field_spearman'}
    missing = expected.difference(table.columns)
    if missing:
        raise ValueError(f'RSS ablation metrics are missing columns: {sorted(missing)}')
    return table

def validate_source_table(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f'{path.name} is missing columns: {sorted(missing)}')
    return table
MODEL_ORDER = ('NMF', 'NNMF', 'STAMP', 'REGAL')

def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.nanstd(first) == 0 or np.nanstd(second) == 0:
        return np.nan
    return float(spearmanr(first, second).statistic)

def spatial_graph(coordinates: np.ndarray) -> sp.csr_matrix:
    n_neighbors = 6
    model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    indices = model.kneighbors(coordinates, return_distance=False)
    rows = np.repeat(np.arange(coordinates.shape[0]), n_neighbors)
    columns = indices[:, 1:].reshape(-1)
    graph = sp.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(coordinates.shape[0], coordinates.shape[0]))
    return graph.maximum(graph.T).tocsr()

def moran_i(values: np.ndarray, graph: sp.csr_matrix) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - float(values.mean())
    denominator = float(centered @ centered)
    weight_sum = float(graph.sum())
    if denominator <= 0 or weight_sum <= 0:
        return np.nan
    numerator = float(centered @ (graph @ centered))
    return float(values.size / weight_sum * numerator / denominator)

def _top_field_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    positive = np.flatnonzero(np.asarray(values, dtype=float) > 0)
    target = min(int(math.ceil(len(values) * fraction)), positive.size)
    if target == 0:
        return np.array([], dtype=int)
    ordered = positive[np.argsort(values[positive], kind='stable')]
    return ordered[-target:]

def _top_field_jaccard(first: np.ndarray, second: np.ndarray, *, fraction: float) -> float:
    first_indices = _top_field_indices(np.asarray(first, dtype=float), fraction)
    second_indices = _top_field_indices(np.asarray(second, dtype=float), fraction)
    if first_indices.size == 0 or second_indices.size == 0:
        return np.nan
    first_mask = np.zeros(len(first), dtype=bool)
    second_mask = np.zeros(len(second), dtype=bool)
    first_mask[first_indices] = True
    second_mask[second_indices] = True
    union = np.logical_or(first_mask, second_mask).sum()
    return float(np.logical_and(first_mask, second_mask).sum() / union)

def local_regional_contrast(field: np.ndarray, program_score: np.ndarray, graph: sp.csr_matrix) -> float:
    high = _top_field_indices(field, 0.1)
    if high.size < 2:
        return np.nan
    high_mask = np.zeros(len(field), dtype=bool)
    high_mask[high] = True
    ring = np.unique(graph[high].indices)
    ring = ring[~high_mask[ring]]
    score_sd = float(np.std(program_score))
    return float(np.mean(program_score[high]) - np.mean(program_score[ring])) / score_sd if ring.size and score_sd > 0 else np.nan

def top_gene_lists(loadings: pd.DataFrame, top_n: int, valid_genes: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    valid = loadings.index.astype(str).isin(valid_genes)
    for component in loadings.columns.astype(str):
        ranked = loadings.loc[valid, component].sort_values(ascending=False)
        positive = ranked.loc[ranked > 0]
        result[component] = positive.head(top_n).index.astype(str).tolist()
    return result

def loading_profile_tables(outputs: dict[str, dict[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves: dict[str, list[np.ndarray]] = {}
    component_records: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        curves[model] = []
        loadings = outputs[model]['loadings'].clip(lower=0)
        for component in loadings.columns.astype(str):
            values = np.sort(loadings[component].to_numpy(dtype=float))[::-1]
            values = values[values > 0]
            if not values.size or float(values.sum()) <= 0:
                continue
            cumulative = np.cumsum(values) / float(values.sum())
            curves[model].append(cumulative)
            proportions = values / float(values.sum())
            record: dict[str, Any] = {'model': model, 'component': component, 'positive_gene_count': int(values.size), 'effective_gene_number': 1.0 / float(np.square(proportions).sum()), 'top50_loading_fraction': float(values[:50].sum() / values.sum())}
            for target in (0.5, 0.7, 0.9):
                record[f'genes_to_{int(target * 100)}'] = int(np.searchsorted(cumulative, target) + 1)
            component_records.append(record)
    curve_records: list[dict[str, Any]] = []
    for (model, model_curves) in curves.items():
        max_rank = max((len(curve) for curve in model_curves))
        padded = np.ones((len(model_curves), max_rank), dtype=float)
        for (index, curve) in enumerate(model_curves):
            padded[index, :len(curve)] = curve
        (q25, median, q75) = np.quantile(padded, [0.25, 0.5, 0.75], axis=0)
        keep = np.unique(np.concatenate((np.arange(min(200, max_rank)), np.geomspace(1, max_rank, 350).astype(int) - 1, np.array([max_rank - 1]))))
        for index in keep[(keep >= 0) & (keep < max_rank)]:
            curve_records.append({'model': model, 'gene_rank': int(index + 1), 'curve_q25': float(q25[index]), 'curve_median': float(median[index]), 'curve_q75': float(q75[index])})
    return (pd.DataFrame(component_records), pd.DataFrame(curve_records))

def component_metric_table(outputs: dict[str, dict[str, pd.DataFrame]], h5ad_path: Path, coordinates: np.ndarray) -> pd.DataFrame:
    top_n = 20
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        valid_genes = set(adata.var_names.astype(str))
    finally:
        if adata.file is not None:
            adata.file.close()
    top_lists = {model: top_gene_lists(objects['loadings'], top_n, valid_genes) for (model, objects) in outputs.items()}
    genes = sorted({gene for model_lists in top_lists.values() for gene_list in model_lists.values() for gene in gene_list})
    expression = load_raw_expression(h5ad_path, genes)
    graph = spatial_graph(coordinates)
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        fields = outputs[model]['fields']
        for (component, component_genes) in top_lists[model].items():
            available = [gene for gene in component_genes if gene in expression.columns]
            if len(available) < 2:
                continue
            log_expression = np.log1p(expression[available].to_numpy(dtype=float))
            ranks = np.column_stack([rankdata(log_expression[:, index]) / len(log_expression) for index in range(len(available))])
            program_score = ranks.mean(axis=1)
            field = fields[component].to_numpy(dtype=float)
            record: dict[str, Any] = {'model': model, 'component': component, 'top_n': top_n, 'top_genes': ','.join(available), 'program_score_field_spearman': _safe_spearman(program_score, field), 'moran_i': moran_i(field, graph), 'local_contrast_z': local_regional_contrast(field, program_score, graph)}
            rows.append(record)
    return pd.DataFrame(rows)

def domain_rsp_mean_activation(domains: pd.Series, activation: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    annotated = activation.copy()
    annotated['domain'] = domains.to_numpy()
    matrix = annotated.groupby('domain', observed=True).mean()
    domain_order = sorted(matrix.index.astype(str), key=int)
    matrix = matrix.reindex(domain_order)
    peak_domain_positions = matrix.to_numpy().argmax(axis=0)
    peak_strengths = matrix.max(axis=0).to_numpy()
    column_order = np.lexsort((-peak_strengths, peak_domain_positions))
    matrix = matrix.iloc[:, column_order]
    rows: list[dict[str, Any]] = []
    for (row_order, domain) in enumerate(matrix.index.astype(str)):
        for (column_order_value, rsp_id) in enumerate(matrix.columns.astype(str)):
            rows.append({'domain': domain, 'rsp_id': rsp_id, 'mean_activation': float(matrix.loc[domain, rsp_id]), 'domain_order': row_order, 'rsp_order': column_order_value, 'above_high_mean_threshold': bool(matrix.loc[domain, rsp_id] >= threshold), 'domain_high_rsp_count': int((matrix.loc[domain] >= threshold).sum()), 'rsp_high_domain_count': int((matrix[rsp_id] >= threshold).sum())})
    return pd.DataFrame(rows)

def program_match_audit(outputs: dict[str, dict[str, pd.DataFrame]], selection: dict[str, Any]) -> pd.DataFrame:
    regal_loadings = outputs['REGAL']['loadings']
    regal_fields = outputs['REGAL']['fields']
    rows: list[dict[str, Any]] = []
    for activity in selection['program_fields']['activities']:
        rsp_id = str(activity['regal'])
        activity_rows: list[dict[str, Any]] = []
        for model in ('NNMF', 'STAMP'):
            selected_component = str(activity[model.lower()])
            loadings = outputs[model]['loadings']
            fields = outputs[model]['fields']
            common_genes = regal_loadings.index.intersection(loadings.index)
            model_rows: list[dict[str, Any]] = []
            for component in loadings.columns.astype(str):
                gene_rho = _safe_spearman(regal_loadings.loc[common_genes, rsp_id].to_numpy(dtype=float), loadings.loc[common_genes, component].to_numpy(dtype=float))
                spatial_rho = _safe_spearman(regal_fields[rsp_id].to_numpy(dtype=float), fields[component].to_numpy(dtype=float))
                combined = 0.7 * max(spatial_rho, 0.0) + 0.3 * max(gene_rho, 0.0)
                model_rows.append({'activity_order': int(activity['order']), 'activity_family': activity['family'], 'activity_display_label': activity['display_label'], 'activity_role': activity['role'], 'activity_reason': activity['reason'], 'regal_rsp': rsp_id, 'comparator': model, 'component': component, 'common_gene_count': len(common_genes), 'gene_loading_spearman': gene_rho, 'spatial_field_spearman': spatial_rho, 'combined_match_score': combined, 'is_selected': component == selected_component})
            ranked = sorted(model_rows, key=lambda row: row['combined_match_score'], reverse=True)
            for (rank, record) in enumerate(ranked, start=1):
                record['combined_match_rank'] = rank
            selected = [record for record in ranked if record['is_selected']]
            if len(selected) != 1 or selected[0]['combined_match_rank'] != 1:
                raise ValueError(f'Frozen {model} match for {rsp_id} is no longer the top joint match')
            activity_rows.extend(ranked)
        selected_nnmf = str(activity['nnmf'])
        selected_stamp = str(activity['stamp'])
        triplet_fields = (outputs['NNMF']['fields'][selected_nnmf].to_numpy(dtype=float), outputs['STAMP']['fields'][selected_stamp].to_numpy(dtype=float), regal_fields[rsp_id].to_numpy(dtype=float))
        field_pairs = ((0, 1), (0, 2), (1, 2))
        triplet_spatial = np.asarray([_safe_spearman(triplet_fields[first], triplet_fields[second]) for (first, second) in field_pairs], dtype=float)
        shared_genes = regal_loadings.index.intersection(outputs['NNMF']['loadings'].index).intersection(outputs['STAMP']['loadings'].index)
        triplet_loadings = (outputs['NNMF']['loadings'].loc[shared_genes, selected_nnmf].to_numpy(dtype=float), outputs['STAMP']['loadings'].loc[shared_genes, selected_stamp].to_numpy(dtype=float), regal_loadings.loc[shared_genes, rsp_id].to_numpy(dtype=float))
        triplet_gene = np.asarray([_safe_spearman(triplet_loadings[first], triplet_loadings[second]) for (first, second) in field_pairs], dtype=float)
        triplet_metrics = {'triplet_common_gene_count': len(shared_genes), 'triplet_spatial_spearman_mean': float(np.nanmean(triplet_spatial)), 'triplet_gene_spearman_mean': float(np.nanmean(triplet_gene)), 'comparator_spatial_spearman': float(triplet_spatial[0])}
        for (fraction, label) in ((0.05, '05'), (0.1, '10'), (0.2, '20')):
            overlaps = [_top_field_jaccard(triplet_fields[first], triplet_fields[second], fraction=fraction) for (first, second) in field_pairs]
            triplet_metrics[f'triplet_top{label}_jaccard_mean'] = float(np.nanmean(overlaps))
        for record in activity_rows:
            record.update(triplet_metrics)
        rows.extend(activity_rows)
    return pd.DataFrame(rows)

def rss_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    model_labels = {'REGAL-RSS': 'RSS input', 'REGAL-expression': 'HVG expression'}
    specifications = {'top50_loading_fraction': 'Loading concentration', 'local_contrast_z': 'Local contrast', 'program_score_field_spearman': 'Field concordance'}
    rows: list[dict[str, Any]] = []
    for (metric, label) in specifications.items():
        reference_values = metrics.loc[metrics['model'].eq('REGAL-RSS'), metric].dropna()
        reference = float(reference_values.median())
        if not np.isfinite(reference) or np.isclose(reference, 0):
            raise ValueError(f'Cannot normalize RSS ablation metric {metric}')
        for (model, display) in model_labels.items():
            subset = metrics.loc[metrics['model'].eq(model), ['component', metric]].dropna()
            relative = subset[metric].to_numpy(dtype=float) / reference
            for ((component, raw), normalized) in zip(subset.to_numpy(), relative):
                rows.append({'record_type': 'program', 'metric': metric, 'metric_label': label, 'model': display, 'component': str(component), 'raw_value': float(raw), 'rss_median_reference': reference, 'relative_value': float(normalized), 'n_programs': len(subset)})
            (q25, median, q75) = np.quantile(relative, [0.25, 0.5, 0.75])
            rows.append({'record_type': 'summary', 'metric': metric, 'metric_label': label, 'model': display, 'rss_median_reference': reference, 'relative_q25': float(q25), 'relative_median': float(median), 'relative_q75': float(q75), 'n_programs': len(subset)})
    return pd.DataFrame(rows)

def _family_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    members = tables['members']
    rsp = tables['rsp']
    summary = members.groupby(['family_id', 'primary_rsp_id'], observed=True).agg(family_size=('candidate_id', 'size'), k_min=('source_k', 'min'), k_max=('source_k', 'max'), k_center=('source_k', 'mean'), k_count=('source_k', 'nunique'), seed_count=('source_seed_offset', 'nunique'), spatial_field_correlation=('family_mean_map_corr', 'first'), active_field_jaccard=('family_mean_active_jaccard', 'first'), gene_jaccard_70=('family_mean_gene_jaccard_70', 'first'), gene_loading_correlation=('family_mean_loading_corr', 'first')).reset_index().merge(rsp[['rsp_id', 'top_genes', 'activation_active_fraction']], left_on='primary_rsp_id', right_on='rsp_id', how='left', validate='one_to_one')
    if summary['rsp_id'].isna().any():
        raise ValueError('Retained RSP families and final RSP table do not align')
    return summary.sort_values(['k_center', 'primary_rsp_id']).reset_index(drop=True)

def family_support_table(members: pd.DataFrame, summary: pd.DataFrame, k_grid: list[int], seeds_per_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (row_order, family_row) in enumerate(summary.itertuples(index=False)):
        family = members.loc[members['family_id'].eq(family_row.family_id)]
        for (k_order, k) in enumerate(k_grid):
            seed_count = int(family.loc[family['source_k'].eq(k), 'source_seed_offset'].nunique())
            rows.append({'family_id': family_row.family_id, 'rsp_id': family_row.primary_rsp_id, 'k': int(k), 'seed_count': seed_count, 'seed_fraction': seed_count / seeds_per_k, 'k_center': float(family_row.k_center), 'rsp_order': row_order, 'k_order': k_order})
    return pd.DataFrame(rows)

def _gene_set(value: Any) -> set[str]:
    return {gene for gene in str(value).split(',') if gene}

def example_family_tables(tables: dict[str, pd.DataFrame], selection: dict[str, Any], spot_ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    member_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    all_members = tables['members']
    all_member_fields = tables['member_activation']
    all_final_fields = tables['activation']
    all_loadings = tables['loadings']
    for example in selection['rsp_stability']['examples']:
        rsp_id = str(example['rsp_id'])
        selected_ids = [str(value) for value in example['candidate_ids']]
        family = all_members.loc[all_members['primary_rsp_id'].eq(rsp_id)].copy()
        if not set(selected_ids).issubset(set(family['candidate_id'].astype(str))):
            raise ValueError(f'Frozen candidate members for {rsp_id} are missing')
        final = all_final_fields.loc[all_final_fields['rsp_id'].eq(rsp_id)].set_index('spot_id')['activation'].reindex(spot_ids)
        if final.isna().any():
            raise ValueError(f'Final field for {rsp_id} is incomplete')
        final_genes = set(all_loadings.loc[all_loadings['rsp_id'].eq(rsp_id) & all_loadings['in_contribution_70'].astype(bool), 'gene'].astype(str))
        family_fields = all_member_fields.loc[all_member_fields['candidate_id'].isin(family['candidate_id'])]
        field_lookup = {candidate_id: group.set_index('spot_id')['activation'].reindex(spot_ids) for (candidate_id, group) in family_fields.groupby('candidate_id', observed=True)}
        for member in family.itertuples(index=False):
            candidate_id = str(member.candidate_id)
            field = field_lookup[candidate_id]
            genes = _gene_set(member.contribution_70_genes)
            union = genes | final_genes
            record = dict(member._asdict())
            record.update({'example_role': example['role'], 'example_color': example['color'], 'selected_for_panel': candidate_id in selected_ids, 'candidate_to_medoid_spatial_spearman': _safe_spearman(field.to_numpy(dtype=float), final.to_numpy(dtype=float)), 'candidate_to_medoid_gene_jaccard_70': len(genes & final_genes) / len(union) if union else np.nan})
            member_rows.append(record)
            if candidate_id in selected_ids:
                map_order = selected_ids.index(candidate_id)
                field_rows.extend(({'example_rsp': rsp_id, 'example_role': example['role'], 'map_kind': 'candidate', 'map_id': candidate_id, 'map_order': map_order, 'spot_id': spot_id, 'activation': float(value)} for (spot_id, value) in zip(spot_ids, field.to_numpy(dtype=float))))
        member_rows.append({'primary_rsp_id': rsp_id, 'rsp_id': rsp_id, 'candidate_id': rsp_id, 'member_status': 'retained_family_representative', 'is_primary': True, 'source_k': int(tables['rsp'].loc[tables['rsp']['rsp_id'].eq(rsp_id), 'source_k'].iloc[0]), 'source_seed': int(tables['rsp'].loc[tables['rsp']['rsp_id'].eq(rsp_id), 'source_seed'].iloc[0]), 'example_role': example['role'], 'example_color': example['color'], 'selected_for_panel': True, 'candidate_to_medoid_spatial_spearman': 1.0, 'candidate_to_medoid_gene_jaccard_70': 1.0})
        field_rows.extend(({'example_rsp': rsp_id, 'example_role': example['role'], 'map_kind': 'final', 'map_id': rsp_id, 'map_order': 3, 'spot_id': spot_id, 'activation': float(value)} for (spot_id, value) in zip(spot_ids, final.to_numpy(dtype=float))))
    return (pd.DataFrame(member_rows), pd.DataFrame(field_rows))

def build_source_tables() -> dict[str, pd.DataFrame]:
    """Build every auditable source table required by the formal R1 exporter."""
    selection = load_selection()
    paths = artifact_paths(selection)
    spatial = load_spatial_context(paths.h5ad)
    outputs = load_program_outputs(paths, spatial.spot_ids)
    sparkx = load_sparkx(paths)
    domains = load_stagate_domains(paths, spatial.spot_ids)
    threshold = float(selection['object_views']['high_mean_activation_threshold'])
    domain_rsp = domain_rsp_mean_activation(domains, outputs['REGAL']['fields'], threshold=threshold)
    selected_gene = str(selection['object_views']['sparkx_gene'])
    gene_row = sparkx.loc[sparkx['gene'].eq(selected_gene)]
    if len(gene_row) != 1:
        raise ValueError(f'Frozen SPARK-X gene {selected_gene} is unavailable')
    significant_count = int((sparkx['adjusted_pvalue'] <= 0.05).sum())
    spatial_gene = pd.DataFrame([{'gene': selected_gene, 'adjusted_pvalue': float(gene_row['adjusted_pvalue'].iloc[0]), 'combined_pvalue': float(gene_row['combined_pvalue'].iloc[0]), 'detected_spots': int(gene_row['detected_spots'].iloc[0]), 'significant_gene_count': significant_count, 'selection_reason': selection['object_views']['sparkx_reason']}])
    highlighted_domain = str(selection['object_views']['highlighted_domain'])
    highlighted_rsp = str(selection['object_views']['highlighted_rsp'])
    domain_counts = domain_rsp.groupby('domain')['domain_high_rsp_count'].first()
    rsp_counts = domain_rsp.groupby('rsp_id')['rsp_high_domain_count'].first()
    if highlighted_domain not in domain_counts.index:
        raise ValueError(f'Frozen highlighted domain {highlighted_domain} is unavailable')
    if highlighted_rsp not in rsp_counts.index:
        raise ValueError(f'Frozen highlighted RSP {highlighted_rsp} is unavailable')
    if int(domain_counts.loc[highlighted_domain]) < 2:
        raise ValueError('Frozen highlighted domain no longer has multi-program support')
    if int(rsp_counts.loc[highlighted_rsp]) < 2:
        raise ValueError('Frozen highlighted RSP no longer has cross-domain support')
    domain_rsp['is_highlighted_domain'] = domain_rsp['domain'].astype(str).eq(highlighted_domain)
    domain_rsp['is_highlighted_rsp'] = domain_rsp['rsp_id'].astype(str).eq(highlighted_rsp)
    program_matches = program_match_audit(outputs, selection)
    (concentration, curves) = loading_profile_tables(outputs)
    metrics = component_metric_table(outputs, paths.h5ad, spatial.coordinates)
    component = concentration.merge(metrics, on=['model', 'component'], how='left', validate='one_to_one')
    program_structure = component
    positive_loading_profiles = curves
    rss_ablation = rss_ablation_table(load_rss_ablation_metrics(paths))
    family_tables = load_rsp_family_tables(paths)
    family_coherence = _family_summary(family_tables)
    example_ids = {str(item['rsp_id']): item for item in selection['rsp_stability']['examples']}
    family_coherence['example_role'] = family_coherence['primary_rsp_id'].map({key: value['role'] for (key, value) in example_ids.items()})
    family_coherence['example_color'] = family_coherence['primary_rsp_id'].map({key: value['color'] for (key, value) in example_ids.items()})
    family_support = family_support_table(family_tables['members'], family_coherence, [int(value) for value in selection['rsp_stability']['k_grid']], int(selection['rsp_stability']['seeds_per_k']))
    (example_members, example_fields) = example_family_tables(family_tables, selection, spatial.spot_ids)
    tables = {'spatial_gene_evidence.csv': spatial_gene, 'domain_rsp_association.csv': domain_rsp, 'cross_method_program_matching.csv': program_matches, 'program_structure_metrics.csv': program_structure, 'positive_loading_profiles.csv': positive_loading_profiles, 'rss_ablation_metrics.csv': rss_ablation, 'rsp_family_support.csv': family_support, 'rsp_family_coherence.csv': family_coherence, 'rsp_family_members.csv': example_members, 'rsp_family_spatial_fields.csv': example_fields}
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for (filename, table) in tables.items():
        write_table(table, TABLE_DIR / filename)
    return tables

def prepare():
    global TABLE_DIR
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    build_source_tables()

def main():
    global _CONFIG, _CONFIG_DIR, _ROOTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    filename = args.config.expanduser().resolve()
    _CONFIG_DIR = filename.parent
    document = yaml.safe_load(filename.read_text())
    _ROOTS = {'HOME': str(Path.home())}
    for name, value in document['roots'].items():
        _ROOTS[name] = Template(value).substitute(_ROOTS)
    _CONFIG = document['fig03']
    prepare()
    missing = [name for name in TABLE_NAMES if not (TABLE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f'Missing prepared tables: {missing}')
    print(TABLE_DIR)

if __name__ == '__main__':
    main()
