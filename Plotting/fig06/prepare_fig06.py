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

from plot_fig06 import RESOURCE_PATHS

from plot_fig06 import FIGURE_SELECTION

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig06'
TABLE_NAMES = ('dcis1_microstructure_associations.csv', 'dcis1_microstructure_readback.csv', 'dcis2_entity_state_de_readback.csv', 'dcis2_entity_state_pathway_readback.csv', 'dcis2_microstructure_readback.csv', 'dcis2_rsd_morphology.csv')

def _resolve(value):
    result = Path(Template(value).substitute(_ROOTS)).expanduser()
    return result if result.is_absolute() else _CONFIG_DIR / result

def load_selection():
    selection = deepcopy(FIGURE_SELECTION)
    selection['sample_id'] = option('sample_id')
    selection['analysis_parameters'] = option('analysis_parameters')
    return selection

def path(name):
    return _resolve(_CONFIG['paths'][name])

def resource(name):
    return RESOURCE_PATHS[name]

def option(name):
    return _CONFIG['options'][name]

def write_table(frame, destination):
    destination = Path(destination)
    if destination.name in TABLE_NAMES:
        frame.to_csv(destination, index=False)

from pathlib import Path
from typing import Iterable
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Any, Iterable
import yaml
from scipy import stats
from collections import deque
import re
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
import os
import warnings
import matplotlib.pyplot as plt
from patsy import dmatrix
DEFAULT_OBS_COLUMNS = ('in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres', 'janesick_cluster', 'janesick_annotation', 'sample_id', 'library_id', 'total_counts')

def load_spot_context(h5ad_path: str | Path, *, sample_id: str, obs_columns: Iterable[str]=DEFAULT_OBS_COLUMNS, spatial_key: str='spatial') -> pd.DataFrame:
    """Read Visium coordinates and selected observation columns."""
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        out = pd.DataFrame({'spot_id': adata.obs_names.astype(str), 'sample_id': sample_id})
        if spatial_key not in adata.obsm:
            raise ValueError(f'Missing {spatial_key!r} coordinates in {h5ad_path}')
        spatial = np.asarray(adata.obsm[spatial_key])[:, :2].astype(float)
        out['x_fullres'] = spatial[:, 0]
        out['y_fullres'] = spatial[:, 1]
        scale = _hires_scale(adata)
        out['x_hires'] = out['x_fullres'] * scale
        out['y_hires'] = out['y_fullres'] * scale
        for col in obs_columns:
            if col not in adata.obs or col in out:
                continue
            values = adata.obs[col]
            if pd.api.types.is_numeric_dtype(values):
                out[col] = pd.to_numeric(values, errors='coerce').to_numpy()
            else:
                out[col] = values.astype(str).to_numpy()
    finally:
        adata.file.close()
    return out

def present_genes(h5ad_path: str | Path) -> list[str]:
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        return adata.var_names.astype(str).tolist()
    finally:
        adata.file.close()

def load_normalized_expression(h5ad_path: str | Path, genes: Iterable[str], *, target_sum: float=10000.0, layer: str | None=None) -> tuple[pd.DataFrame, list[str]]:
    """Return log1p library-normalized expression for selected genes."""
    requested = list(dict.fromkeys((str(g) for g in genes if str(g))))
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        selected = [gene for gene in requested if gene in adata.var_names]
        sub = adata[:, selected].to_memory()
        matrix = sub.layers[layer] if layer and layer in sub.layers else sub.X
        values = _dense(matrix).astype(float)
        totals = _row_totals(adata)
        totals = np.maximum(totals.astype(float), 1.0)
        values = np.log1p(values / totals[:, None] * float(target_sum))
        expr = pd.DataFrame(values, index=sub.obs_names.astype(str), columns=selected)
    finally:
        adata.file.close()
    return (expr, selected)

def load_expression_for_spots(h5ad_path: str | Path, spot_ids: Iterable[str], genes: Iterable[str], *, target_sum: float=10000.0, layer: str | None=None) -> pd.DataFrame:
    (expr, present) = load_normalized_expression(h5ad_path, genes, target_sum=target_sum, layer=layer)
    ordered = [str(spot) for spot in spot_ids if str(spot) in expr.index]
    return expr.loc[ordered, present].copy()

def load_raw_expression_for_spots(h5ad_path: str | Path, spot_ids: Iterable[str], genes: Iterable[str] | None=None, *, layer: str | None=None) -> pd.DataFrame:
    """Read raw expression for a small spot subset without densifying the full h5ad."""
    requested_spots = list(dict.fromkeys((str(spot) for spot in spot_ids if str(spot))))
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        spot_to_pos = {str(spot): idx for (idx, spot) in enumerate(adata.obs_names.astype(str))}
        ordered_spots = [spot for spot in requested_spots if spot in spot_to_pos]
        positions = np.array(sorted((spot_to_pos[spot] for spot in ordered_spots)), dtype=int)
        if genes is None:
            selected_genes = adata.var_names.astype(str).tolist()
        else:
            requested_genes = list(dict.fromkeys((str(gene) for gene in genes if str(gene))))
            selected_genes = [gene for gene in requested_genes if gene in adata.var_names]
        sub = adata[positions, selected_genes].to_memory()
        matrix = sub.layers[layer] if layer and layer in sub.layers else sub.X
        frame = pd.DataFrame(_dense(matrix).astype(float), index=sub.obs_names.astype(str), columns=selected_genes)
    finally:
        adata.file.close()
    return frame.reindex(ordered_spots)

def _hires_scale(adata: ad.AnnData) -> float:
    registration = adata.uns.get('janesick_registration', {})
    scalefactors = registration.get('visium_scalefactors', {})
    if 'tissue_hires_scalef' in scalefactors:
        return float(scalefactors['tissue_hires_scalef'])
    spatial = adata.uns.get('spatial', {})
    if spatial:
        library = next(iter(spatial.values()))
        return float(library.get('scalefactors', {}).get('tissue_hires_scalef', 1.0))
    return 1.0

def _dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, 'to_memory'):
        matrix = matrix.to_memory()
        if sparse.issparse(matrix):
            return matrix.toarray()
    return np.asarray(matrix)

def _row_totals(adata: ad.AnnData) -> np.ndarray:
    if 'total_counts' in adata.obs:
        return pd.to_numeric(adata.obs['total_counts'], errors='coerce').fillna(0).to_numpy(float)
    matrix = adata.X
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if hasattr(matrix, 'to_memory'):
        matrix = matrix.to_memory()
        if sparse.issparse(matrix):
            return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return np.asarray(matrix).sum(axis=1).astype(float)
REQUIRED_RSP_FILES = {'rsp': 'rsp.parquet', 'activation': 'rsp_activation.parquet', 'gene_loadings': 'rsp_gene_loadings.parquet', 'core_gene_loadings': 'rsp_core_gene_loadings.parquet'}
REQUIRED_RSMD_FILES = {'domains': 'domains.parquet', 'membership': 'domain_spot_membership.parquet', 'domain_rsp_map': 'domain_rsp_map.parquet', 'metadomains': 'metadomains.parquet'}
SCFFPE_TO_XENIUM = {'B Cells': 'B_Cells', 'CD4+ T Cells': 'CD4+_T_Cells', 'CD8+ T Cells': 'CD8+_T_Cells', 'DCIS 1': 'DCIS_1', 'DCIS 2': 'DCIS_2', 'Endothelial': 'Endothelial', 'IRF7+ DCs': 'IRF7+_DCs', 'Invasive Tumor': 'Invasive_Tumor', 'LAMP3+ DCs': 'LAMP3+_DCs', 'Macrophages 1': 'Macrophages_1', 'Macrophages 2': 'Macrophages_2', 'Mast Cells': 'Mast_Cells', 'Myoepi ACTA2+': 'Myoepi_ACTA2+', 'Myoepi KRT15+': 'Myoepi_KRT15+', 'Perivascular-Like': 'Perivascular-Like', 'Prolif Invasive Tumor': 'Prolif_Invasive_Tumor', 'Stromal': 'Stromal'}

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)

def load_rsp_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(sample_dir) / 'rsp_out'
    return _load_required_tables(root, REQUIRED_RSP_FILES)

def load_rsmd_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(sample_dir) / 'rsmd_out'
    return _load_required_tables(root, REQUIRED_RSMD_FILES)

def load_rss_scores(sample_dir: str | Path, genes: Iterable[str], spot_ids: Iterable[str] | None=None) -> pd.DataFrame:
    """Read current sparse RSS scores for selected genes as a spot-by-gene matrix."""
    requested_genes = list(dict.fromkeys((str(gene) for gene in genes if str(gene))))
    path = Path(sample_dir) / 'rss_out' / 'rss' / 'rss_sparse.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing current RSS artifact: {path}')
    long = pd.read_parquet(path, columns=['spot_id', 'gene', 'rss_score'], filters=[('gene', 'in', requested_genes)])
    long['spot_id'] = long['spot_id'].astype(str)
    long['gene'] = long['gene'].astype(str)
    requested_spots = None if spot_ids is None else list(dict.fromkeys(map(str, spot_ids)))
    if requested_spots is not None:
        long = long.loc[long['spot_id'].isin(requested_spots)]
    matrix = long.pivot(index='spot_id', columns='gene', values='rss_score')
    if requested_spots is not None:
        matrix = matrix.reindex(requested_spots)
    return matrix.reindex(columns=requested_genes).fillna(0.0).astype(float)

def load_xenium_mapping(path: str | Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    required = {'visium_barcode', 'label', 'in_visium_capture_area'}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f'Xenium mapping is missing columns: {sorted(missing)}')
    keep = mapping['in_visium_capture_area'].astype(bool) & mapping['visium_barcode'].notna()
    return mapping.loc[keep].copy()

def xenium_composition_by_spot(mapping: pd.DataFrame) -> pd.DataFrame:
    counts = mapping.groupby(['visium_barcode', 'label']).size().unstack(fill_value=0).rename_axis(None, axis=1)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    total = counts.sum(axis=1).astype(float)
    out = pd.DataFrame(index=counts.index)
    out.index.name = 'spot_id'
    out['xenium_total_cells'] = total
    for label in sorted(counts.columns):
        out[f'count_{label}'] = counts[label].astype(float)
        out[f'frac_{label}'] = counts[label].div(total.replace(0, np.nan)).fillna(0.0)
    return out.reset_index()

def activation_wide(activation: pd.DataFrame, rsp_ids: Iterable[str]) -> pd.DataFrame:
    rsp_ids = list(dict.fromkeys((str(rsp) for rsp in rsp_ids)))
    sub = activation.loc[activation['rsp_id'].astype(str).isin(rsp_ids)].copy()
    out = pd.DataFrame(index=sorted(sub['spot_id'].astype(str).unique()))
    out.index.name = 'spot_id'
    value_map = {'activation': 'activation', 'activation_raw': 'activation_raw', 'activation_active': 'active', 'rank_in_spot': 'rank'}
    for (value_col, prefix) in value_map.items():
        wide = sub.pivot(index='spot_id', columns='rsp_id', values=value_col)
        wide.index = wide.index.astype(str)
        for rsp_id in rsp_ids:
            values = wide[rsp_id] if rsp_id in wide else 0.0
            out[f'{prefix}_{rsp_id}'] = pd.Series(values, index=wide.index).reindex(out.index)
    activation_cols = [column for column in out if column.startswith(('activation_', 'activation_raw_'))]
    active_cols = [column for column in out if column.startswith('active_')]
    out[activation_cols] = out[activation_cols].fillna(0.0).astype(float)
    out[active_cols] = out[active_cols].fillna(False).astype(bool)
    return out.reset_index()

def selected_rsd_membership(selection: dict[str, Any], domains: pd.DataFrame, membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domain_index = domains.set_index(['rsp_seed_id', 'domain_id'], drop=False)
    long_rows: list[pd.DataFrame] = []
    registry_rows: list[dict[str, Any]] = []
    for item in selection['selected_rsp_objects']:
        rsp_id = str(item['rsp_id'])
        for domain_id in map(str, item.get('selected_domains', [])):
            key = (rsp_id, domain_id)
            if key not in domain_index.index:
                raise ValueError(f'Selected domain does not match current artifacts: {rsp_id}:{domain_id}')
            domain = domain_index.loc[key]
            if isinstance(domain, pd.DataFrame):
                raise ValueError(f'Duplicate domain rows for {rsp_id}:{domain_id}')
            spots = membership.loc[membership['domain_key'].astype(str).eq(str(domain['domain_key'])), 'spot_id'].astype(str)
            long_rows.append(pd.DataFrame({'spot_id': spots, 'rsp_id': rsp_id, 'domain_id': domain_id, 'domain_key': str(domain['domain_key']), 'block': str(item['block']), 'role': str(item['role'])}))
            registry_rows.append({'rsp_id': rsp_id, 'domain_id': domain_id, 'domain_key': str(domain['domain_key']), 'rsmd_id': str(domain['rsmd_id']), 'spot_count': int(len(spots)), 'qc_pass': bool(domain.get('qc_pass', True)), 'geo_centroid_x': _number(domain.get('geo_centroid_x')), 'geo_centroid_y': _number(domain.get('geo_centroid_y')), 'geo_area_est': _number(domain.get('geo_area_est')), 'geo_compactness': _number(domain.get('geo_compactness')), 'geo_boundary_ratio': _number(domain.get('geo_boundary_ratio')), 'geo_elongation': _number(domain.get('geo_elongation')), 'components_count': _number(domain.get('components_count')), 'domain_reliability': _number(domain.get('domain_reliability')), 'block': str(item['block']), 'role': str(item['role']), 'description': str(item.get('description', ''))})
    long = pd.concat(long_rows, ignore_index=True)
    registry = pd.DataFrame(registry_rows)
    wide = long.assign(value=True).pivot_table(index='spot_id', columns='rsp_id', values='value', aggfunc='any', fill_value=False).astype(bool)
    wide.columns = [f'selected_rsd_{column}' for column in wide.columns]
    return (long, wide.reset_index(), registry)

def validate_inputs(selection: dict[str, Any], rsp_tables: dict[str, pd.DataFrame], rsmd_tables: dict[str, pd.DataFrame], spot_context: pd.DataFrame, composition: pd.DataFrame) -> pd.DataFrame:
    selected_rsp = [str(item['rsp_id']) for item in selection['selected_rsp_objects']]
    available_rsp = set(rsp_tables['rsp']['rsp_id'].astype(str))
    missing_rsp = sorted(set(selected_rsp).difference(available_rsp))
    (_, _, registry) = selected_rsd_membership(selection, rsmd_tables['domains'], rsmd_tables['membership'])
    spot_ids = set(spot_context['spot_id'].astype(str))
    activation_spots = set(rsp_tables['activation']['spot_id'].astype(str))
    composition_spots = set(composition['spot_id'].astype(str))
    rows = [{'check': 'selected_rsp_present', 'passed': not missing_rsp, 'value': ';'.join(missing_rsp)}, {'check': 'selected_domains_present', 'passed': len(registry) > 0, 'value': len(registry)}, {'check': 'spot_context_unique', 'passed': spot_context['spot_id'].is_unique, 'value': len(spot_context)}, {'check': 'activation_spot_alignment', 'passed': spot_ids == activation_spots, 'value': len(activation_spots)}, {'check': 'composition_subset_of_visium', 'passed': composition_spots.issubset(spot_ids), 'value': len(composition_spots)}]
    result = pd.DataFrame(rows)
    if not result['passed'].all():
        failed = result.loc[~result['passed'], 'check'].tolist()
        raise ValueError(f'R4 input validation failed: {failed}')
    return result

def build_shared_spot_table(spot_context: pd.DataFrame, activation: pd.DataFrame, composition: pd.DataFrame, rsd_wide: pd.DataFrame, targeted_expression: pd.DataFrame | None=None) -> pd.DataFrame:
    out = spot_context.copy()
    out['spot_id'] = out['spot_id'].astype(str)
    for table in (activation, composition, rsd_wide):
        local = table.copy()
        local['spot_id'] = local['spot_id'].astype(str)
        out = out.merge(local, on='spot_id', how='left', validate='one_to_one')
    if targeted_expression is not None:
        expr = targeted_expression.copy()
        expr.index = expr.index.astype(str)
        expr.columns = [f'expr_{column}' for column in expr.columns]
        out = out.merge(expr.rename_axis('spot_id').reset_index(), on='spot_id', how='left', validate='one_to_one')
    count_cols = [column for column in out if column.startswith('count_')]
    frac_cols = [column for column in out if column.startswith('frac_')]
    rsd_cols = [column for column in out if column.startswith('selected_rsd_')]
    out[count_cols + frac_cols + ['xenium_total_cells']] = out[count_cols + frac_cols + ['xenium_total_cells']].fillna(0.0)
    out[rsd_cols] = out[rsd_cols].fillna(False).astype(bool)
    return out

def load_scffpe_markers(path: str | Path, *, practical_only: bool=True) -> pd.DataFrame:
    markers = pd.read_csv(path)
    if practical_only and 'passes_practical_marker_filter' in markers:
        markers = markers.loc[markers['passes_practical_marker_filter'].astype(bool)].copy()
    markers['gene'] = markers['gene'].astype(str)
    markers['cell_label'] = markers['cell_label'].astype(str)
    markers['xenium_label'] = markers['cell_label'].map(SCFFPE_TO_XENIUM).fillna('')
    return markers

def loading_basis_pair(loadings: pd.DataFrame, rsp_a: str, rsp_b: str) -> pd.DataFrame:
    columns = ['gene', 'loading', 'loading_rank', 'loading_fraction', 'in_contribution_70', 'is_core_gene']
    a = loadings.loc[loadings['rsp_id'].astype(str).eq(rsp_a), columns].rename(columns={column: f'{rsp_a}_{column}' for column in columns if column != 'gene'})
    b = loadings.loc[loadings['rsp_id'].astype(str).eq(rsp_b), columns].rename(columns={column: f'{rsp_b}_{column}' for column in columns if column != 'gene'})
    out = a.merge(b, on='gene', how='outer')
    for rsp_id in (rsp_a, rsp_b):
        for column in ('loading', 'loading_fraction'):
            out[f'{rsp_id}_{column}'] = out[f'{rsp_id}_{column}'].fillna(0.0)
        for column in ('in_contribution_70', 'is_core_gene'):
            out[f'{rsp_id}_{column}'] = out[f'{rsp_id}_{column}'].fillna(False).astype(bool)
    out['loading_difference'] = out[f'{rsp_a}_loading'] - out[f'{rsp_b}_loading']
    out['shared_core'] = out[f'{rsp_a}_is_core_gene'] & out[f'{rsp_b}_is_core_gene']
    out['shared_contribution70'] = out[f'{rsp_a}_in_contribution_70'] & out[f'{rsp_b}_in_contribution_70']
    out['loading_class'] = np.select([out['shared_core'], out['loading_difference'] > 0, out['loading_difference'] < 0], ['shared_core', f'{rsp_a}_weighted', f'{rsp_b}_weighted'], default='balanced_or_zero')
    return out.sort_values('loading_difference', ascending=False).reset_index(drop=True)

def multi_rsp_loading_basis(loadings: pd.DataFrame, rsp_ids: Iterable[str]) -> pd.DataFrame:
    rsp_ids = list(map(str, rsp_ids))
    merged: pd.DataFrame | None = None
    for rsp_id in rsp_ids:
        sub = loadings.loc[loadings['rsp_id'].astype(str).eq(rsp_id), ['gene', 'loading', 'loading_rank', 'in_contribution_70', 'is_core_gene']].rename(columns={'loading': f'loading_{rsp_id}', 'loading_rank': f'rank_{rsp_id}', 'in_contribution_70': f'contribution70_{rsp_id}', 'is_core_gene': f'core_{rsp_id}'})
        merged = sub if merged is None else merged.merge(sub, on='gene', how='outer')
    assert merged is not None
    loading_cols = [f'loading_{rsp}' for rsp in rsp_ids]
    core_cols = [f'core_{rsp}' for rsp in rsp_ids]
    contribution_cols = [f'contribution70_{rsp}' for rsp in rsp_ids]
    merged[loading_cols] = merged[loading_cols].fillna(0.0)
    merged[core_cols + contribution_cols] = merged[core_cols + contribution_cols].fillna(False).astype(bool)
    merged['positive_loading_count'] = (merged[loading_cols] > 0).sum(axis=1)
    merged['core_or_contribution_count'] = pd.concat([merged[core_cols].set_axis(rsp_ids, axis=1), merged[contribution_cols].set_axis(rsp_ids, axis=1)], axis=0).groupby(level=0).max().sum(axis=1)
    merged['shared_backbone'] = merged['positive_loading_count'].eq(len(rsp_ids)) & merged['core_or_contribution_count'].ge(2)
    merged['dominant_rsp'] = merged[loading_cols].idxmax(axis=1).str.replace('loading_', '', regex=False)
    merged['loading_range'] = merged[loading_cols].max(axis=1) - merged[loading_cols].min(axis=1)
    return merged.sort_values(['shared_backbone', 'loading_range'], ascending=[False, False]).reset_index(drop=True)

def loading_gene_sets(gene_loadings: pd.DataFrame, rsp_ids: Iterable[str]) -> pd.DataFrame:
    """Build gene sets from artifact-defined core and contribution-70 support."""
    rows: list[dict[str, Any]] = []
    for rsp_id in map(str, rsp_ids):
        rsp = gene_loadings.loc[gene_loadings['rsp_id'].astype(str).eq(rsp_id)].copy()
        rsp = rsp.sort_values('loading_rank')
        contribution = rsp['in_contribution_70'].fillna(False).astype(bool)
        core = rsp['is_core_gene'].fillna(False).astype(bool)
        scopes = {'core': rsp.loc[core], 'contribution70': rsp.loc[contribution], 'core_or_contribution70': rsp.loc[core | contribution]}
        for (scope, subset) in scopes.items():
            for (rank, row) in enumerate(subset.itertuples(index=False), start=1):
                rows.append({'gene_set_id': f'{rsp_id}_{scope}', 'gene_set_type': 'single_rsp_loading', 'rsp_id': rsp_id, 'scope': scope, 'gene': str(row.gene), 'gene_set_rank': rank, 'loading_statistic': float(row.loading), 'loading_rank': float(row.loading_rank)})
    return pd.DataFrame(rows)

def pairwise_loading_gene_sets(gene_loadings: pd.DataFrame, rsp_a: str, rsp_b: str) -> pd.DataFrame:
    """Build pairwise sets inside each RSP's core/contribution-70 support."""
    pair = loading_basis_pair(gene_loadings, rsp_a, rsp_b)
    supported_a = pair[f'{rsp_a}_is_core_gene'] | pair[f'{rsp_a}_in_contribution_70']
    supported_b = pair[f'{rsp_b}_is_core_gene'] | pair[f'{rsp_b}_in_contribution_70']
    shared = pair.loc[supported_a & supported_b].copy()
    shared['set_statistic'] = pair.loc[shared.index, [f'{rsp_a}_loading', f'{rsp_b}_loading']].min(axis=1)
    a_weighted = pair.loc[supported_a & pair['loading_difference'].gt(0)].copy()
    a_weighted = a_weighted.sort_values('loading_difference', ascending=False)
    a_weighted['set_statistic'] = a_weighted['loading_difference']
    b_weighted = pair.loc[supported_b & pair['loading_difference'].lt(0)].copy()
    b_weighted = b_weighted.sort_values('loading_difference')
    b_weighted['set_statistic'] = -b_weighted['loading_difference']
    definitions = [(f'{rsp_a}_{rsp_b}_shared_supported', 'pair_shared', 'shared', shared.sort_values('set_statistic', ascending=False)), (f'{rsp_a}_weighted_vs_{rsp_b}_supported', 'pair_weighted', rsp_a, a_weighted), (f'{rsp_b}_weighted_vs_{rsp_a}_supported', 'pair_weighted', rsp_b, b_weighted)]
    rows: list[dict[str, Any]] = []
    for (gene_set_id, gene_set_type, direction, subset) in definitions:
        for (rank, row) in enumerate(subset.itertuples(index=False), start=1):
            rows.append({'gene_set_id': gene_set_id, 'gene_set_type': gene_set_type, 'rsp_id': direction, 'scope': 'core_or_contribution70', 'gene': str(row.gene), 'gene_set_rank': rank, 'loading_statistic': float(row.set_statistic), 'loading_rank': np.nan})
    return pd.DataFrame(rows)

def relative_loading_gene_set(gene_loadings: pd.DataFrame, reference_rsp: str, target_rsp: str) -> pd.DataFrame:
    """Return one complete target-relative loading set.

    Artifact-defined core/contribution membership is used only as an inclusion
    rule.  The returned set is not partitioned into hand-labelled biological
    categories; relative loading is retained as the continuous ranking
    statistic for downstream enrichment and spatial readback.
    """
    pair = loading_basis_pair(gene_loadings, reference_rsp, target_rsp).copy()
    reference = str(reference_rsp)
    target = str(target_rsp)
    target_supported = pair[f'{target}_is_core_gene'].fillna(False).astype(bool) | pair[f'{target}_in_contribution_70'].fillna(False).astype(bool)
    pair['comparison_id'] = f'{reference}_vs_{target}'
    pair['reference_rsp'] = reference
    pair['target_rsp'] = target
    pair['relative_loading'] = pair[f'{target}_loading'] - pair[f'{reference}_loading']
    pair = pair.loc[target_supported & pair['relative_loading'].gt(0)].copy()
    pair = pair.sort_values('relative_loading', ascending=False).reset_index(drop=True)
    pair['gene_set_id'] = f'{target}_weighted_vs_{reference}_supported'
    pair['gene_set_type'] = 'complete_relative_loading'
    pair['gene_set_rank'] = np.arange(1, len(pair) + 1)
    total = pair['relative_loading'].sum()
    pair['relative_loading_weight'] = pair['relative_loading'] / total if total else np.nan
    return pair

def signed_loading_contrast_score(gene_loadings: pd.DataFrame, expression: pd.DataFrame, reference_rsp: str, target_rsp: str) -> tuple[pd.Series, pd.DataFrame]:
    """Score a complete target-versus-reference loading contrast.

    Target-enhanced artifact-supported genes receive positive weights and
    reference-enhanced genes receive negative weights.  This distinguishes
    relative RSP realization from the expression magnitude of their shared
    carrier program.
    """
    target = relative_loading_gene_set(gene_loadings, reference_rsp, target_rsp)
    reference = relative_loading_gene_set(gene_loadings, target_rsp, reference_rsp)
    weights = pd.concat([target[['gene', 'relative_loading']].assign(signed_loading=lambda frame: frame['relative_loading']), reference[['gene', 'relative_loading']].assign(signed_loading=lambda frame: -frame['relative_loading'])], ignore_index=True)
    weights = weights.groupby('gene', as_index=False)['signed_loading'].sum()
    weights = weights.loc[weights['signed_loading'].ne(0)].copy()
    present = [gene for gene in weights['gene'].astype(str) if gene in expression.columns]
    weights = weights.loc[weights['gene'].isin(present)].copy()
    denominator = weights['signed_loading'].abs().sum()
    weights['signed_loading_weight'] = weights['signed_loading'] / denominator
    numeric = expression[present].apply(pd.to_numeric, errors='coerce')
    z_scores = (numeric - numeric.mean()).div(numeric.std(ddof=0).replace(0, np.nan), axis=1)
    indexed_weights = weights.set_index('gene').loc[present, 'signed_loading_weight']
    score = z_scores.mul(indexed_weights, axis=1).sum(axis=1, min_count=1)
    score.name = f'signed_contrast_{target_rsp}_vs_{reference_rsp}'
    return (score, weights.sort_values('signed_loading', ascending=False).reset_index(drop=True))

def multi_rsp_specific_gene_sets(gene_loadings: pd.DataFrame, rsp_ids: Iterable[str]) -> pd.DataFrame:
    """Build multi-RSP sets inside artifact-defined core/contribution support."""
    rsp_ids = list(map(str, rsp_ids))
    basis = multi_rsp_loading_basis(gene_loadings, rsp_ids)
    loading_cols = [f'loading_{rsp_id}' for rsp_id in rsp_ids]
    rows: list[dict[str, Any]] = []
    shared = basis.loc[basis['shared_backbone']].copy()
    shared['set_statistic'] = shared[loading_cols].min(axis=1)
    shared = shared.sort_values('set_statistic', ascending=False)
    for (rank, row) in enumerate(shared.itertuples(index=False), start=1):
        rows.append({'gene_set_id': f"{'_'.join(rsp_ids)}_shared_supported", 'gene_set_type': 'multi_shared', 'rsp_id': 'shared', 'scope': 'core_or_contribution70_in_at_least_two_rsps', 'gene': str(row.gene), 'gene_set_rank': rank, 'loading_statistic': float(row.set_statistic), 'loading_rank': np.nan})
    for rsp_id in rsp_ids:
        own = f'loading_{rsp_id}'
        others = [column for column in loading_cols if column != own]
        subset = basis.copy()
        subset['set_statistic'] = subset[own] - subset[others].max(axis=1)
        own_supported = subset[f'core_{rsp_id}'] | subset[f'contribution70_{rsp_id}']
        subset = subset.loc[own_supported & subset['set_statistic'].gt(0)].sort_values('set_statistic', ascending=False)
        for (rank, row) in enumerate(subset.itertuples(index=False), start=1):
            rows.append({'gene_set_id': f"{rsp_id}_weighted_within_{'_'.join(rsp_ids)}_supported", 'gene_set_type': 'multi_weighted', 'rsp_id': rsp_id, 'scope': 'core_or_contribution70', 'gene': str(row.gene), 'gene_set_rank': rank, 'loading_statistic': float(row.set_statistic), 'loading_rank': np.nan})
    return pd.DataFrame(rows)

def load_gmt(path: str | Path, database: str) -> dict[str, set[str]]:
    """Read a GMT collection with database-qualified term names."""
    gene_sets: dict[str, set[str]] = {}
    with Path(path).open() as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                genes = {gene for gene in parts[2:] if gene}
                if genes:
                    gene_sets[f'{database}:{parts[0]}'] = genes
    return gene_sets

def loading_gene_set_ora(query_sets: pd.DataFrame, pathway_sets: dict[str, set[str]], universe: Iterable[str], *, minimum_overlap: int=3, minimum_term_size: int=10, maximum_term_size: int=1000) -> pd.DataFrame:
    """Run one-sided ORA for loading-derived gene sets with complete provenance."""
    universe = set(map(str, universe))
    rows: list[dict[str, Any]] = []
    metadata_cols = ['gene_set_id', 'gene_set_type', 'rsp_id', 'scope']
    pathway_records = [(term.split(':', 1)[0], term, set(genes)) for (term, genes) in pathway_sets.items()]
    database_coverage: dict[str, set[str]] = {}
    for (database, _, genes) in pathway_records:
        database_coverage.setdefault(database, set()).update(genes)
    for (keys, group) in query_sets.groupby(metadata_cols, dropna=False, sort=True):
        metadata = dict(zip(metadata_cols, keys))
        raw_query = set(group['gene'].astype(str)).intersection(universe)
        for (database, term, genes) in pathway_records:
            database_universe = universe.intersection(database_coverage[database])
            query = raw_query.intersection(database_universe)
            if not query:
                continue
            term_genes = genes.intersection(database_universe)
            if not minimum_term_size <= len(term_genes) <= maximum_term_size:
                continue
            overlap = query.intersection(term_genes)
            p_value = stats.hypergeom.sf(len(overlap) - 1, len(database_universe), len(term_genes), len(query))
            rows.append({**metadata, 'term': term, 'database': database, 'query_n': len(query), 'term_n': len(term_genes), 'universe_n': len(database_universe), 'overlap_n': len(overlap), 'overlap_genes': ';'.join(sorted(overlap)), 'p_value': float(p_value)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['q_value'] = out.groupby('gene_set_id')['p_value'].transform(benjamini_hochberg)
        out['q_value_within_database'] = out.groupby(['gene_set_id', 'database'])['p_value'].transform(benjamini_hochberg)
        out['fold_enrichment'] = out['overlap_n'] / out['query_n'] / (out['term_n'] / out['universe_n'])
        out = out.loc[out['overlap_n'].ge(minimum_overlap)].copy()
    return out.sort_values(['gene_set_id', 'q_value', 'p_value']).reset_index(drop=True)

def loading_gene_set_marker_attribution(query_sets: pd.DataFrame, markers: pd.DataFrame, universe: Iterable[str]) -> pd.DataFrame:
    """Attribute shared or differential loading gene sets to scFFPE cell states."""
    universe = set(map(str, universe))
    marker = markers.copy()
    marker['gene'] = marker['gene'].astype(str)
    marker['cell_label'] = marker['cell_label'].astype(str)
    marker['marker_rank'] = pd.to_numeric(marker.get('rank', np.nan), errors='coerce')
    marker_frequency = marker.groupby('gene')['cell_label'].nunique().clip(lower=1)
    marker['marker_specificity_weight'] = marker['gene'].map(1.0 / marker_frequency)
    marker['marker_rank_weight'] = (101.0 - marker['marker_rank']).clip(lower=1.0, upper=100.0) / 100.0
    marker_sets = {cell_label: set(group['gene']).intersection(universe) for (cell_label, group) in marker.groupby('cell_label', sort=True)}
    metadata_cols = ['gene_set_id', 'gene_set_type', 'rsp_id', 'scope']
    rows: list[dict[str, Any]] = []
    for (keys, group) in query_sets.groupby(metadata_cols, dropna=False, sort=True):
        metadata = dict(zip(metadata_cols, keys))
        query = set(group['gene'].astype(str)).intersection(universe)
        ranked = group.loc[group['gene'].astype(str).isin(query), ['gene', 'gene_set_rank']].copy()
        ranked['gene_set_rank_weight'] = ((len(ranked) + 1 - pd.to_numeric(ranked['gene_set_rank'], errors='coerce')) / max(len(ranked), 1)).clip(lower=0.0)
        for (cell_label, marker_genes) in marker_sets.items():
            overlap = query.intersection(marker_genes)
            a = len(overlap)
            b = len(query - marker_genes)
            c = len(marker_genes - query)
            d = max(0, len(universe) - a - b - c)
            (odds_ratio, p_value) = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
            overlap_detail = ranked.loc[ranked['gene'].isin(overlap)].merge(marker.loc[marker['cell_label'].eq(cell_label), ['gene', 'marker_rank_weight', 'marker_specificity_weight']], on='gene', how='left')
            overlap_detail['evidence_weight'] = overlap_detail['gene_set_rank_weight'] * overlap_detail['marker_rank_weight'] * overlap_detail['marker_specificity_weight']
            leading = overlap_detail.sort_values('evidence_weight', ascending=False)
            rows.append({**metadata, 'cell_label': cell_label, 'xenium_label': SCFFPE_TO_XENIUM.get(cell_label, ''), 'query_n': len(query), 'marker_n': len(marker_genes), 'overlap_n': a, 'overlap_genes': ';'.join(leading['gene'].astype(str)), 'weighted_marker_support': float(overlap_detail['evidence_weight'].sum()), 'odds_ratio': float(odds_ratio), 'p_value': float(p_value)})
    out = pd.DataFrame(rows)
    out['q_value'] = out.groupby('gene_set_id')['p_value'].transform(benjamini_hochberg)
    out['source_evidence'] = np.where(out['q_value'].le(0.05) & out['odds_ratio'].gt(1) & out['overlap_n'].ge(2), 'marker_enriched_source', 'not_marker_enriched')
    return out.sort_values(['gene_set_id', 'source_evidence', 'q_value', 'weighted_marker_support'], ascending=[True, True, True, False]).reset_index(drop=True)

def expression_gene_set_scores(expression: pd.DataFrame, gene_sets: pd.DataFrame, *, minimum_genes: int=3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score loading-derived gene sets using mean gene-wise standardized expression."""
    numeric = expression.apply(pd.to_numeric, errors='coerce')
    z_scores = (numeric - numeric.mean(axis=0)).div(numeric.std(axis=0, ddof=0).replace(0, np.nan), axis=1)
    scores = pd.DataFrame(index=expression.index.astype(str))
    coverage_rows: list[dict[str, Any]] = []
    for (gene_set_id, group) in gene_sets.groupby('gene_set_id', sort=True):
        requested = list(dict.fromkeys(group.sort_values('gene_set_rank')['gene'].astype(str)))
        present = [gene for gene in requested if gene in z_scores.columns and z_scores[gene].notna().any()]
        coverage_rows.append({'gene_set_id': gene_set_id, 'gene_set_type': group['gene_set_type'].iloc[0], 'rsp_id': group['rsp_id'].iloc[0], 'scope': group['scope'].iloc[0], 'requested_gene_count': len(requested), 'present_gene_count': len(present), 'present_genes': ';'.join(present), 'score_available': len(present) >= minimum_genes})
        if len(present) >= minimum_genes:
            scores[f'geneset_{gene_set_id}'] = z_scores[present].mean(axis=1, skipna=True)
    scores.index.name = 'spot_id'
    return (scores.reset_index(), pd.DataFrame(coverage_rows))

def gene_set_association_table(table: pd.DataFrame, outcome_cols: Iterable[str], score_cols: Iterable[str], *, scope_mask: pd.Series | np.ndarray | None=None, control_cols: Iterable[str]=()) -> pd.DataFrame:
    """Relate loading-derived gene-set scores to continuous phenotypes."""
    local = table if scope_mask is None else table.loc[np.asarray(scope_mask, dtype=bool)]
    controls = [column for column in control_cols if column in local]
    rows: list[dict[str, Any]] = []
    for outcome in outcome_cols:
        if outcome not in local:
            continue
        for score in score_cols:
            if score not in local:
                continue
            (rho, p_value, n) = safe_spearman(local[outcome], local[score])
            rows.append({'outcome': outcome, 'gene_set_score': score.removeprefix('geneset_'), 'association_measure': 'spearman', 'controls': '', 'n_spots': n, 'rho': rho, 'p_value': p_value})
            if controls:
                (partial_rho, partial_p, partial_n) = partial_spearman(local[outcome], local[score], local[controls])
                rows.append({'outcome': outcome, 'gene_set_score': score.removeprefix('geneset_'), 'association_measure': 'partial_spearman', 'controls': ';'.join(controls), 'n_spots': partial_n, 'rho': partial_rho, 'p_value': partial_p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['q_value'] = out.groupby(['outcome', 'association_measure'])['p_value'].transform(benjamini_hochberg)
    return out.sort_values(['outcome', 'association_measure', 'q_value']).reset_index(drop=True)

def weighted_cell_marker_attribution(gene_loadings: pd.DataFrame, markers: pd.DataFrame, *, marker_rank_cap: int=100) -> pd.DataFrame:
    """Attribute artifact-supported loading genes to scFFPE states."""
    marker = markers.copy()
    marker['gene'] = marker['gene'].astype(str)
    marker['cell_label'] = marker['cell_label'].astype(str)
    marker['marker_rank'] = pd.to_numeric(marker.get('rank', np.nan), errors='coerce')
    marker['marker_rank_weight'] = (marker_rank_cap + 1 - marker['marker_rank']).clip(lower=1, upper=marker_rank_cap) / marker_rank_cap
    gene_frequency = marker.groupby('gene')['cell_label'].nunique().rename('marker_cell_frequency')
    marker = marker.merge(gene_frequency, on='gene', how='left')
    marker['marker_specificity_weight'] = 1.0 / marker['marker_cell_frequency'].clip(lower=1)
    loading = gene_loadings.loc[gene_loadings['is_core_gene'].fillna(False).astype(bool) | gene_loadings['in_contribution_70'].fillna(False).astype(bool)].copy()
    loading['gene'] = loading['gene'].astype(str)
    loading['loading_rank_weight'] = 1.0 / np.sqrt(pd.to_numeric(loading['loading_rank'], errors='coerce').clip(lower=1))
    hits = loading.merge(marker, on='gene', how='inner')
    hits['evidence_weight'] = hits['loading_rank_weight'] * hits['marker_rank_weight'] * hits['marker_specificity_weight']
    rows: list[dict[str, Any]] = []
    for ((rsp_id, cell_label), group) in hits.groupby(['rsp_id', 'cell_label'], sort=True):
        leading = group.sort_values('evidence_weight', ascending=False)
        rows.append({'rsp_id': str(rsp_id), 'cell_label': str(cell_label), 'xenium_label': SCFFPE_TO_XENIUM.get(str(cell_label), ''), 'n_supported_marker_hits': group['gene'].nunique(), 'weighted_marker_support': float(group['evidence_weight'].sum()), 'best_loading_rank': float(group['loading_rank'].min()), 'leading_marker_genes': ';'.join((f'{row.gene}(r{int(row.loading_rank)})' for row in leading.head(12).itertuples(index=False)))})
    out = pd.DataFrame(rows)
    out['support_share_within_rsp'] = out['weighted_marker_support'] / out.groupby('rsp_id')['weighted_marker_support'].transform('sum')
    out['support_rank_within_rsp'] = out.groupby('rsp_id')['weighted_marker_support'].rank(method='min', ascending=False)
    out['support_percentile_across_rsps'] = out.groupby('cell_label')['weighted_marker_support'].rank(method='average', pct=True)
    return out.sort_values(['rsp_id', 'support_rank_within_rsp', 'cell_label']).reset_index(drop=True)

def cell_carrier_evidence_table(rsp_id: str, outcome_col: str, attribution: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    """Separate likely signal carriers from cells that only define local context."""
    molecular = attribution.loc[attribution['rsp_id'].astype(str).eq(rsp_id)].copy()
    assoc = associations.loc[associations['outcome'].eq(outcome_col) & associations['association_measure'].eq('fraction_spearman')].copy()
    out = molecular.merge(assoc[['cell_label', 'n_spots', 'rho', 'p_value', 'q_value']], left_on='xenium_label', right_on='cell_label', how='outer', suffixes=('_scffpe', '_xenium'))
    if 'cell_label_xenium' in out:
        out['xenium_label'] = out['xenium_label'].fillna(out['cell_label_xenium'])
    out['evidence_cell_label'] = out['cell_label_scffpe'].fillna(out['xenium_label'])
    out['rsp_id'] = rsp_id
    out['outcome'] = outcome_col
    molecular_pass = out['support_rank_within_rsp'].le(3) & out['support_percentile_across_rsps'].ge(0.75) & out['n_supported_marker_hits'].ge(2)
    association_pass = out['q_value'].le(0.05) & out['rho'].gt(0)
    out['evidence_class'] = np.select([molecular_pass & association_pass, molecular_pass, ~molecular_pass & association_pass], ['candidate_signal_carrier', 'molecular_carrier_support', 'contextual_association'], default='unsupported')
    out['interpretation_rule'] = 'carrier requires top-three specificity-weighted scFFPE support, across-RSP percentile >=0.75, at least two core/contribution-70 marker hits, and positive mapped-cell association'
    return out.sort_values(['evidence_class', 'support_rank_within_rsp', 'q_value'], na_position='last').reset_index(drop=True)

def continuous_cell_associations(table: pd.DataFrame, outcome_cols: Iterable[str], cell_labels: Iterable[str], *, scope_mask: pd.Series | np.ndarray | None=None) -> pd.DataFrame:
    local = table if scope_mask is None else table.loc[np.asarray(scope_mask, dtype=bool)]
    rows: list[dict[str, Any]] = []
    for outcome_col in outcome_cols:
        for label in cell_labels:
            count_col = f'count_{label}'
            frac_col = f'frac_{label}'
            if count_col not in local or frac_col not in local:
                continue
            (rho_frac, p_frac, n_frac) = safe_spearman(local[outcome_col], local[frac_col])
            (rho_count, p_count, n_count) = partial_spearman(local[outcome_col], local[count_col], local[['xenium_total_cells']])
            rows.extend([{'outcome': outcome_col, 'cell_label': label, 'association_measure': 'fraction_spearman', 'n_spots': n_frac, 'rho': rho_frac, 'p_value': p_frac}, {'outcome': outcome_col, 'cell_label': label, 'association_measure': 'count_partial_spearman_total_cells', 'n_spots': n_count, 'rho': rho_count, 'p_value': p_count}])
    out = pd.DataFrame(rows)
    out['q_value'] = out.groupby(['outcome', 'association_measure'])['p_value'].transform(benjamini_hochberg)
    return out.sort_values(['outcome', 'association_measure', 'q_value']).reset_index(drop=True)

def producer_evidence_table(rsp_id: str, outcome_col: str, enrichment: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    enrich = enrichment.loc[enrichment['rsp_id'].astype(str).eq(rsp_id) & enrichment['program_scope'].isin(['core', 'contribution70'])].copy()
    enrich['enrichment_pass'] = enrich['q_value'].le(0.05) & enrich['odds_ratio'].gt(1) & enrich['overlap_n'].ge(2)
    best_enrich = enrich.sort_values(['enrichment_pass', 'q_value', 'overlap_n'], ascending=[False, True, False])
    best_enrich = best_enrich.drop_duplicates('xenium_label')
    assoc = associations.loc[associations['outcome'].eq(outcome_col)].copy()
    assoc['association_pass'] = assoc['q_value'].le(0.05) & assoc['rho'].gt(0)
    best_assoc = assoc.sort_values(['association_pass', 'q_value', 'rho'], ascending=[False, True, False])
    best_assoc = best_assoc.drop_duplicates('cell_label')
    out = best_enrich.merge(best_assoc, left_on='xenium_label', right_on='cell_label', how='outer', suffixes=('_enrichment', '_association'))
    out['rsp_id'] = rsp_id
    out['outcome'] = outcome_col
    out['enrichment_pass'] = out['enrichment_pass'].fillna(False).astype(bool)
    out['association_pass'] = out['association_pass'].fillna(False).astype(bool)
    out['support_class'] = np.select([out['enrichment_pass'] & out['association_pass'], ~out['enrichment_pass'] & out['association_pass'], out['enrichment_pass'] & ~out['association_pass']], ['candidate_producer', 'contextual_support', 'loading_supported_only'], default='unsupported')
    return out.sort_values(['support_class', 'q_value_enrichment', 'q_value_association'], na_position='last')

def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': pd.to_numeric(y, errors='coerce')}).dropna()
    if len(frame) < 3 or frame['x'].nunique() < 2 or frame['y'].nunique() < 2:
        return (float('nan'), float('nan'), len(frame))
    result = stats.spearmanr(frame['x'], frame['y'])
    return (float(result.statistic), float(result.pvalue), len(frame))

def partial_spearman(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> tuple[float, float, int]:
    frame = pd.concat([pd.to_numeric(x, errors='coerce').rename('x'), pd.to_numeric(y, errors='coerce').rename('y'), controls.apply(pd.to_numeric, errors='coerce')], axis=1).dropna()
    if len(frame) < 4 or frame['x'].nunique() < 2 or frame['y'].nunique() < 2:
        return (float('nan'), float('nan'), len(frame))
    ranked = frame.rank(method='average')
    design = np.column_stack([np.ones(len(ranked)), ranked[controls.columns].to_numpy(float)])
    x_resid = ranked['x'].to_numpy(float) - design @ np.linalg.lstsq(design, ranked['x'], rcond=None)[0]
    y_resid = ranked['y'].to_numpy(float) - design @ np.linalg.lstsq(design, ranked['y'], rcond=None)[0]
    result = stats.pearsonr(x_resid, y_resid)
    return (float(result.statistic), float(result.pvalue), len(frame))

def benjamini_hochberg(values: pd.Series | np.ndarray) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    valid = np.isfinite(p_values)
    result = np.full(len(p_values), np.nan)
    if not valid.any():
        return result
    valid_values = p_values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order] * len(valid_values) / np.arange(1, len(valid_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.clip(ranked, 0, 1)
    result[np.where(valid)[0]] = restored
    return result

def _load_required_tables(root: Path, files: dict[str, str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for (key, filename) in files.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing required table: {path}')
        tables[key] = pd.read_parquet(path)
    return tables

def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')

def build_spatial_edges(spots: pd.DataFrame, *, x_col: str='x_fullres', y_col: str='y_fullres', radius_multiplier: float=1.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the Visium six-neighbor graph from spot-center distances."""
    local = spots[['spot_id', x_col, y_col]].dropna().copy().reset_index(drop=True)
    coordinates = local[[x_col, y_col]].to_numpy(float)
    tree = cKDTree(coordinates)
    (nearest, _) = tree.query(coordinates, k=2)
    step = float(np.median(nearest[:, 1]))
    radius = step * float(radius_multiplier)
    pairs = sorted(tree.query_pairs(radius))
    edges = pd.DataFrame({'source': [str(local.iloc[i]['spot_id']) for (i, _) in pairs], 'target': [str(local.iloc[j]['spot_id']) for (_, j) in pairs], 'distance': [float(np.linalg.norm(coordinates[i] - coordinates[j])) for (i, j) in pairs]})
    degree = pd.concat([edges['source'], edges['target']]).value_counts().reindex(local['spot_id'], fill_value=0)
    qc = pd.DataFrame([{'n_spots': len(local), 'n_edges': len(edges), 'median_nearest_neighbor_distance': step, 'adjacency_radius': radius, 'median_degree': float(degree.median()), 'mean_degree': float(degree.mean()), 'fraction_degree_six': float(degree.eq(6).mean())}])
    return (edges, qc)

def xenium_cell_microstructures(cells: pd.DataFrame, *, target_label: str, x_col: str='x_um', y_col: str='y_um', eps_um: float | None=None, min_samples: int=5, minimum_cells: int=20, prefix: str='X') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Define connected cell-level microstructures in Xenium coordinates.

    When ``eps_um`` is omitted, the radius is the 75th percentile of the fifth
    nearest-neighbor distance. Small/noise clusters remain explicitly
    unassigned instead of being forced into a nearby structure.
    """
    target = cells.loc[cells['label'].astype(str).eq(target_label)].copy().reset_index(drop=True)
    if target.empty:
        raise ValueError(f'No Xenium cells found for label={target_label!r}')
    coordinates = target[[x_col, y_col]].to_numpy(float)
    neighbor_k = min(6, len(target))
    (distances, _) = cKDTree(coordinates).query(coordinates, k=neighbor_k)
    fifth_index = min(5, neighbor_k - 1)
    inferred_eps = float(np.quantile(distances[:, fifth_index], 0.75))
    radius = float(eps_um) if eps_um is not None else inferred_eps
    labels = DBSCAN(eps=radius, min_samples=int(min_samples), n_jobs=-1).fit_predict(coordinates)
    raw_counts = pd.Series(labels[labels >= 0]).value_counts()
    retained = raw_counts.loc[raw_counts.ge(int(minimum_cells))]
    ordered = retained.sort_values(ascending=False).index.tolist()
    width = max(3, len(str(max(1, len(ordered)))))
    id_map = {label: f'{prefix}{rank:0{width}d}' for (rank, label) in enumerate(ordered, start=1)}
    target['microstructure_id'] = pd.Series(labels).map(id_map).fillna('unassigned')
    target['microstructure_assigned'] = target['microstructure_id'].ne('unassigned')
    rows: list[dict[str, float | int | str]] = []
    for (microstructure_id, group) in target.loc[target['microstructure_assigned']].groupby('microstructure_id'):
        xy = group[[x_col, y_col]].to_numpy(float)
        centroid = xy.mean(axis=0)
        compactness = _expanded_hull_compactness(xy, radius=float(np.median(distances[:, 1])) / 2.0)
        rows.append({'microstructure_id': microstructure_id, 'dcis1_cell_count': len(group), 'centroid_x_um': float(centroid[0]), 'centroid_y_um': float(centroid[1]), 'x_span_um': float(np.ptp(xy[:, 0])), 'y_span_um': float(np.ptp(xy[:, 1])), 'compactness': compactness, 'mapped_visium_spot_count': group['visium_barcode'].dropna().astype(str).nunique()})
    summary = pd.DataFrame(rows).sort_values('dcis1_cell_count', ascending=False).reset_index(drop=True)
    parameters = pd.DataFrame([{'target_label': target_label, 'n_target_cells': len(target), 'median_nearest_neighbor_um': float(np.median(distances[:, 1])), 'fifth_neighbor_q75_um': inferred_eps, 'dbscan_eps_um': radius, 'dbscan_min_samples': int(min_samples), 'minimum_cells': int(minimum_cells), 'n_retained_microstructures': len(summary), 'assigned_cell_fraction': float(target['microstructure_assigned'].mean())}])
    return (target, summary, parameters)

def xenium_microstructure_context(assignments: pd.DataFrame, all_cells: pd.DataFrame, *, context_labels: Iterable[str], contact_radius_um: float, x_col: str='x_um', y_col: str='y_um') -> pd.DataFrame:
    """Measure cell-level boundary contact around each Xenium microstructure."""
    context_labels = list(dict.fromkeys(map(str, context_labels)))
    context = all_cells.loc[all_cells['label'].astype(str).isin(context_labels)].copy()
    context_groups = {label: group.reset_index(drop=True) for (label, group) in context.groupby(context['label'].astype(str), sort=False)}
    rows: list[dict[str, float | int | str]] = []
    assigned = assignments.loc[assignments['microstructure_assigned'].astype(bool)]
    for (microstructure_id, structure) in assigned.groupby('microstructure_id'):
        structure_xy = structure[[x_col, y_col]].to_numpy(float)
        for label in context_labels:
            group = context_groups.get(label)
            if group is None or group.empty:
                rows.append({'microstructure_id': microstructure_id, 'context_label': label, 'dcis1_cell_count': len(structure), 'context_cell_count_within_radius': 0, 'context_cells_per_100_dcis1': 0.0, 'contacted_dcis1_fraction': 0.0, 'median_nearest_context_distance_um': np.nan, 'contact_radius_um': float(contact_radius_um)})
                continue
            context_xy = group[[x_col, y_col]].to_numpy(float)
            context_tree = cKDTree(context_xy)
            (nearest, _) = context_tree.query(structure_xy, k=1)
            contacted = nearest <= float(contact_radius_um)
            nearby_lists = context_tree.query_ball_point(structure_xy, r=float(contact_radius_um))
            unique_nearby = set((index for indices in nearby_lists for index in indices))
            rows.append({'microstructure_id': microstructure_id, 'context_label': label, 'dcis1_cell_count': len(structure), 'context_cell_count_within_radius': len(unique_nearby), 'context_cells_per_100_dcis1': len(unique_nearby) / len(structure) * 100.0, 'contacted_dcis1_fraction': float(contacted.mean()), 'median_nearest_context_distance_um': float(np.median(nearest)), 'contact_radius_um': float(contact_radius_um)})
    return pd.DataFrame(rows).sort_values(['microstructure_id', 'context_label']).reset_index(drop=True)

def xenium_target_cell_neighborhoods(cells: pd.DataFrame, *, target_labels: Iterable[str], context_labels: Iterable[str], radius_um: float, x_col: str='x_um', y_col: str='y_um') -> pd.DataFrame:
    """Measure the actual cell-level organization around selected Xenium cells."""
    target_labels = list(dict.fromkeys(map(str, target_labels)))
    context_labels = list(dict.fromkeys(map(str, context_labels)))
    local = cells.dropna(subset=[x_col, y_col]).copy().reset_index(drop=True)
    target = local.loc[local['label'].astype(str).isin(target_labels)].copy().reset_index(drop=True)
    if target.empty:
        raise ValueError(f'No Xenium cells found for target labels={target_labels!r}')
    target_xy = target[[x_col, y_col]].to_numpy(float)
    all_xy = local[[x_col, y_col]].to_numpy(float)
    all_tree = cKDTree(all_xy)
    target['local_all_cell_count'] = all_tree.query_ball_point(target_xy, r=float(radius_um), return_length=True)
    label_counts: dict[str, np.ndarray] = {}
    for label in list(dict.fromkeys([*target_labels, *context_labels])):
        group = local.loc[local['label'].astype(str).eq(label)]
        safe_label = _safe_label(label)
        if group.empty:
            counts = np.zeros(len(target), dtype=int)
            nearest = np.full(len(target), np.nan)
        else:
            tree = cKDTree(group[[x_col, y_col]].to_numpy(float))
            counts = tree.query_ball_point(target_xy, r=float(radius_um), return_length=True)
            (nearest, _) = tree.query(target_xy, k=1)
        label_counts[label] = np.asarray(counts, dtype=float)
        target[f'local_count_{safe_label}'] = counts
        target[f'nearest_distance_um_{safe_label}'] = nearest
        target[f'contact_{safe_label}'] = np.asarray(counts) > 0
    target_count = np.sum([label_counts[label] for label in target_labels], axis=0)
    target['local_invasive_cell_count'] = target_count
    target['local_invasive_fraction_all_cells'] = target_count / np.maximum(target['local_all_cell_count'].to_numpy(float), 1.0)
    proliferative = label_counts.get('Prolif_Invasive_Tumor', np.zeros(len(target)))
    target['local_proliferative_fraction_invasive'] = proliferative / np.maximum(target_count, 1.0)
    target['cell_neighborhood_radius_um'] = float(radius_um)
    return target

def aggregate_target_cell_neighborhoods(neighborhoods: pd.DataFrame, *, spot_col: str='visium_barcode') -> pd.DataFrame:
    """Aggregate cell-coordinate neighborhood measurements to registered Visium spots."""
    local = neighborhoods.loc[neighborhoods[spot_col].notna()].copy()
    local[spot_col] = local[spot_col].astype(str)
    rows: list[dict[str, float | int | str]] = []
    for (spot_id, group) in local.groupby(spot_col, sort=True):
        row: dict[str, float | int | str] = {'spot_id': spot_id, 'mapped_invasive_cell_count': len(group), 'mapped_proliferative_invasive_fraction': float(group['label'].astype(str).eq('Prolif_Invasive_Tumor').mean()), 'mean_local_all_cell_count': float(group['local_all_cell_count'].mean()), 'mean_local_invasive_cell_count': float(group['local_invasive_cell_count'].mean()), 'mean_local_invasive_fraction_all_cells': float(group['local_invasive_fraction_all_cells'].mean()), 'mean_local_proliferative_fraction_invasive': float(group['local_proliferative_fraction_invasive'].mean())}
        for column in group.columns:
            if column.startswith('contact_'):
                row[f'fraction_cells_{column}'] = float(group[column].mean())
            elif column.startswith('local_count_'):
                row[f'mean_{column}'] = float(group[column].mean())
            elif column.startswith('nearest_distance_um_'):
                row[f'median_{column}'] = float(group[column].median())
        rows.append(row)
    return pd.DataFrame(rows)

def map_xenium_microstructures_to_visium(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map Xenium-defined structures to Visium without forcing exclusive membership."""
    assigned = assignments.loc[assignments['microstructure_assigned'].astype(bool) & assignments['visium_barcode'].notna()].copy()
    long = assigned.groupby(['visium_barcode', 'microstructure_id']).size().rename('dcis1_cell_count').reset_index().rename(columns={'visium_barcode': 'spot_id'})
    total = long.groupby('spot_id')['dcis1_cell_count'].transform('sum')
    long['microstructure_cell_fraction_within_spot'] = long['dcis1_cell_count'] / total
    long['microstructure_rank_within_spot'] = long.groupby('spot_id')['dcis1_cell_count'].rank(method='min', ascending=False)
    rows: list[dict[str, float | int | str]] = []
    for (spot_id, group) in long.groupby('spot_id'):
        ordered = group.sort_values(['dcis1_cell_count', 'microstructure_id'], ascending=[False, True])
        fractions = ordered['microstructure_cell_fraction_within_spot'].to_numpy(float)
        entropy = float(-(fractions * np.log(fractions)).sum()) if len(fractions) else np.nan
        normalized_entropy = entropy / np.log(len(fractions)) if len(fractions) > 1 else 0.0
        rows.append({'spot_id': str(spot_id), 'dominant_microstructure_id': str(ordered.iloc[0]['microstructure_id']), 'dominant_microstructure_fraction': float(fractions[0]), 'mapped_dcis1_structure_cell_count': int(ordered['dcis1_cell_count'].sum()), 'n_microstructures_in_spot': len(ordered), 'microstructure_mixing_entropy': entropy, 'microstructure_mixing_entropy_normalized': normalized_entropy, 'is_mixed_microstructure_spot': bool(len(ordered) > 1 and fractions[0] < 0.8)})
    return (long.sort_values(['spot_id', 'microstructure_rank_within_spot']), pd.DataFrame(rows))

def component_membership(spot_ids: Iterable[str], edges: pd.DataFrame, selected_spots: Iterable[str], *, prefix: str='C') -> pd.DataFrame:
    all_spots = list(dict.fromkeys(map(str, spot_ids)))
    selected = set(map(str, selected_spots)).intersection(all_spots)
    index = {spot: idx for (idx, spot) in enumerate(all_spots)}
    chosen = [index[spot] for spot in all_spots if spot in selected]
    if not chosen:
        return pd.DataFrame(columns=['spot_id', 'component_id'])
    adjacency = _adjacency_matrix(all_spots, edges)
    (n_components, labels) = connected_components(adjacency[chosen][:, chosen], directed=False)
    width = max(3, len(str(max(0, n_components - 1))))
    return pd.DataFrame({'spot_id': [all_spots[idx] for idx in chosen], 'component_id': [f'{prefix}{label:0{width}d}' for label in labels]})

def component_geometry(spots: pd.DataFrame, edges: pd.DataFrame, membership: pd.DataFrame, *, x_col: str='x_fullres', y_col: str='y_fullres') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize component geometry and assign boundary order to member spots."""
    spot_index = spots.set_index('spot_id', drop=False)
    neighbors = neighbor_map(edges)
    step = float(edges['distance'].median()) if len(edges) else 1.0
    component_rows: list[dict[str, float | int | str]] = []
    spot_rows: list[dict[str, float | int | str | bool]] = []
    for (component_id, group) in membership.groupby('component_id', sort=True):
        member = set(group['spot_id'].astype(str))
        boundary = {spot for spot in member if any((neighbor not in member for neighbor in neighbors.get(spot, set())))}
        orders = _boundary_orders(member, boundary, neighbors)
        coordinates = spot_index.loc[sorted(member), [x_col, y_col]].to_numpy(float)
        compactness = _expanded_hull_compactness(coordinates, radius=step / 2.0)
        centroid = coordinates.mean(axis=0)
        centered = coordinates - centroid
        covariance = np.cov(centered.T) if len(coordinates) > 1 else np.zeros((2, 2))
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        elongation = float(np.sqrt(eigenvalues[0] / eigenvalues[1])) if eigenvalues[1] > 0 else np.nan
        component_rows.append({'component_id': component_id, 'spot_count': len(member), 'centroid_x': float(centroid[0]), 'centroid_y': float(centroid[1]), 'boundary_spot_count': len(boundary), 'boundary_fraction': len(boundary) / len(member), 'max_boundary_order': max(orders.values()) if orders else 0, 'compactness': compactness, 'elongation': elongation})
        for spot in sorted(member):
            spot_rows.append({'spot_id': spot, 'component_id': component_id, 'is_boundary': spot in boundary, 'boundary_order': int(orders.get(spot, 0))})
    return (pd.DataFrame(component_rows), pd.DataFrame(spot_rows))

def neighborhood_means(table: pd.DataFrame, edges: pd.DataFrame, value_cols: Iterable[str], *, prefix: str='neighbor_mean_') -> pd.DataFrame:
    indexed = table.set_index('spot_id')
    neighbors = neighbor_map(edges)
    rows: list[dict[str, float | str]] = []
    for spot in indexed.index.astype(str):
        row: dict[str, float | str] = {'spot_id': spot}
        available = [neighbor for neighbor in neighbors.get(spot, set()) if neighbor in indexed.index]
        for column in value_cols:
            row[f'{prefix}{column}'] = pd.to_numeric(indexed.loc[available, column], errors='coerce').mean() if available else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def territory_boundary_depth(all_spots: Iterable[str], edges: pd.DataFrame, territory_spots: Iterable[str]) -> pd.DataFrame:
    """Measure graph depth from the boundary of each connected territory component.

    Boundary spots have depth zero. Increasing ``boundary_order`` moves inward
    by one Visium-neighbor step; ``relative_depth`` scales this order to the
    deepest spot in the same connected component.
    """
    all_spots = list(dict.fromkeys(map(str, all_spots)))
    territory = set(map(str, territory_spots)).intersection(all_spots)
    if not territory:
        return pd.DataFrame(columns=['spot_id', 'territory_component_id', 'is_territory_boundary', 'boundary_order', 'component_max_boundary_order', 'relative_depth'])
    membership = component_membership(all_spots, edges, territory, prefix='T')
    neighbors = neighbor_map(edges)
    rows: list[dict[str, float | int | str | bool]] = []
    for (component_id, group) in membership.groupby('component_id', sort=True):
        members = set(group['spot_id'].astype(str))
        boundary = {spot for spot in members if any((neighbor not in members for neighbor in neighbors.get(spot, set())))}
        if not boundary:
            boundary = set(members)
        orders = _boundary_orders(members, boundary, neighbors)
        max_order = max(orders.values()) if orders else 0
        for spot in sorted(members):
            order = int(orders.get(spot, 0))
            rows.append({'spot_id': spot, 'territory_component_id': component_id, 'is_territory_boundary': spot in boundary, 'boundary_order': order, 'component_max_boundary_order': int(max_order), 'relative_depth': order / max_order if max_order > 0 else 0.0})
    return pd.DataFrame(rows)

def activation_pair_table(table: pd.DataFrame, rsp_a: str, rsp_b: str, *, scope_mask: pd.Series | np.ndarray | None=None, epsilon: float=1e-09) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = table if scope_mask is None else table.loc[np.asarray(scope_mask, dtype=bool)]
    col_a = f'activation_{rsp_a}'
    col_b = f'activation_{rsp_b}'
    out = local[['spot_id', 'x_fullres', 'y_fullres', col_a, col_b]].copy()
    out['coactivation_geometric_mean'] = np.sqrt(np.clip(out[col_a], 0, None) * np.clip(out[col_b], 0, None))
    out['activation_balance'] = (out[col_a] - out[col_b]) / (out[col_a] + out[col_b] + epsilon)
    pearson = out[[col_a, col_b]].corr(method='pearson').iloc[0, 1]
    spearman = out[[col_a, col_b]].corr(method='spearman').iloc[0, 1]
    summary = pd.DataFrame([{'rsp_a': rsp_a, 'rsp_b': rsp_b, 'n_spots': len(out), 'pearson': float(pearson), 'spearman': float(spearman), 'mean_coactivation': float(out['coactivation_geometric_mean'].mean()), 'mean_balance': float(out['activation_balance'].mean())}])
    return (out, summary)

def rsd_relation_table(rsd_sets: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    keys = list(rsd_sets)
    for (i, left) in enumerate(keys):
        for right in keys[i + 1:]:
            (a, b) = (rsd_sets[left], rsd_sets[right])
            overlap = a & b
            union = a | b
            rows.append({'left': left, 'right': right, 'left_spots': len(a), 'right_spots': len(b), 'overlap_spots': len(overlap), 'jaccard': len(overlap) / len(union) if union else np.nan, 'left_contained_by_right': len(overlap) / len(a) if a else np.nan, 'right_contained_by_left': len(overlap) / len(b) if b else np.nan})
    return pd.DataFrame(rows)

def graph_distance_groups(all_spots: Iterable[str], edges: pd.DataFrame, core_spots: Iterable[str], carrier_spots: Iterable[str], *, max_ring_order: int=2) -> pd.DataFrame:
    all_spots = list(dict.fromkeys(map(str, all_spots)))
    core = set(map(str, core_spots))
    carrier = set(map(str, carrier_spots))
    distance = _multi_source_distance(all_spots, edges, core)
    rows = []
    for spot in all_spots:
        if spot in core:
            group = 'core'
        elif spot in carrier and distance.get(spot) == 1:
            group = 'ring1'
        elif spot in carrier and distance.get(spot) == 2 and (max_ring_order >= 2):
            group = 'ring2'
        elif spot in carrier:
            group = 'distant_carrier'
        else:
            continue
        rows.append({'spot_id': spot, 'carrier_group': group, 'graph_distance_to_core': distance.get(spot, np.nan)})
    return pd.DataFrame(rows)

def nearest_carrier_background(all_spots: Iterable[str], edges: pd.DataFrame, core_spots: Iterable[str], carrier_spots: Iterable[str], *, n: int=100) -> pd.DataFrame:
    all_spots = list(dict.fromkeys(map(str, all_spots)))
    core = set(map(str, core_spots))
    carrier = set(map(str, carrier_spots)).difference(core)
    distance = _multi_source_distance(all_spots, edges, core)
    ordered = sorted(carrier, key=lambda spot: (distance.get(spot, np.inf), spot))[:n]
    return pd.DataFrame({'spot_id': ordered, 'background_rank': np.arange(1, len(ordered) + 1), 'graph_distance_to_core': [distance.get(spot, np.nan) for spot in ordered]})

def neighbor_map(edges: pd.DataFrame) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {}
    for row in edges.itertuples(index=False):
        (source, target) = (str(row.source), str(row.target))
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    return neighbors

def _safe_label(label: str) -> str:
    return re.sub('[^0-9A-Za-z]+', '_', str(label)).strip('_')

def _adjacency_matrix(spots: list[str], edges: pd.DataFrame) -> csr_matrix:
    index = {spot: idx for (idx, spot) in enumerate(spots)}
    rows: list[int] = []
    cols: list[int] = []
    for row in edges.itertuples(index=False):
        (source, target) = (str(row.source), str(row.target))
        if source not in index or target not in index:
            continue
        (i, j) = (index[source], index[target])
        rows.extend([i, j])
        cols.extend([j, i])
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(spots), len(spots)))

def _boundary_orders(members: set[str], boundary: set[str], neighbors: dict[str, set[str]]) -> dict[str, int]:
    distance = {spot: 0 for spot in boundary}
    queue = deque(boundary)
    while queue:
        spot = queue.popleft()
        for neighbor in neighbors.get(spot, set()):
            if neighbor in members and neighbor not in distance:
                distance[neighbor] = distance[spot] + 1
                queue.append(neighbor)
    return distance

def _multi_source_distance(all_spots: list[str], edges: pd.DataFrame, sources: set[str]) -> dict[str, int]:
    neighbors = neighbor_map(edges)
    distance = {spot: 0 for spot in sources if spot in all_spots}
    queue = deque(distance)
    while queue:
        spot = queue.popleft()
        for neighbor in neighbors.get(spot, set()):
            if neighbor not in distance:
                distance[neighbor] = distance[spot] + 1
                queue.append(neighbor)
    return distance

def _expanded_hull_compactness(coordinates: np.ndarray, radius: float) -> float:
    if len(coordinates) < 3:
        return float('nan')
    try:
        hull = ConvexHull(coordinates)
    except Exception:
        return float('nan')
    area = float(hull.volume) + float(hull.area) * radius + np.pi * radius ** 2
    perimeter = float(hull.area) + 2 * np.pi * radius
    return float(4 * np.pi * area / perimeter ** 2) if perimeter > 0 else float('nan')

def prepare():
    SAMPLE_DIR = path('sample_dir')
    VISIUM_H5AD = path('h5ad')
    XENIUM_MAPPING = path('xenium_mapping')
    SCFFPE_MARKERS = resource('scffpe_markers')
    LITERATURE_ANCHORS = resource('literature_anchors.yaml')
    REACTOME_GMT = resource('c2.cp.reactome.v2026.1.Hs.symbols.gmt')
    BIOPLANET_GMT = resource('BioPlanet_2019.Enrichr.gmt')
    HALLMARK_GMT = resource('h.all.v2026.1.Hs.symbols.gmt')
    SAMPLE_ID = option('sample_id')
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-r4')
    warnings.filterwarnings('ignore', category=FutureWarning)
    selection = load_selection()
    literature = load_yaml(LITERATURE_ANCHORS)
    rsp_tables = load_rsp_tables(SAMPLE_DIR)
    rsmd_tables = load_rsmd_tables(SAMPLE_DIR)
    selected_objects = selection['selected_rsp_objects']
    selected_rsp_ids = [str(item['rsp_id']) for item in selected_objects]

    def object_by_role(role: str) -> dict:
        matches = [item for item in selected_objects if item['role'] == role]
        if len(matches) != 1:
            raise ValueError(f'Expected one selected object for role={role!r}, found {len(matches)}')
        return matches[0]

    def selected_set(rsd_long: pd.DataFrame, rsp_id: str) -> set[str]:
        return set(rsd_long.loc[rsd_long['rsp_id'].eq(rsp_id), 'spot_id'].astype(str))

    def marker_genes_for_case(case_key: str) -> list[str]:
        return list(literature.get(case_key, {}).get('marker_index', {}).keys())

    def combine_readbacks(*parts: tuple[str, str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for (readback_type, readback_unit, table) in parts:
            frame = table.copy()
            frame.insert(0, 'readback_unit', readback_unit)
            frame.insert(0, 'readback_type', readback_type)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False)

    def significant_pathway_readback(table: pd.DataFrame, n_per_gene_set: int=10) -> pd.DataFrame:
        return table.loc[table['q_value'].le(0.05)].sort_values(['gene_set_id', 'q_value', 'p_value']).groupby('gene_set_id', sort=False).head(n_per_gene_set).reset_index(drop=True)

    def export_table(name: str, table: pd.DataFrame) -> None:
        write_table(table, TABLE_DIR / name)
    spot_context = load_spot_context(VISIUM_H5AD, sample_id=SAMPLE_ID)
    xenium_mapping = load_xenium_mapping(XENIUM_MAPPING)
    xenium_composition = xenium_composition_by_spot(xenium_mapping)
    (rsd_long, rsd_wide, selected_domain_registry) = selected_rsd_membership(selection, rsmd_tables['domains'], rsmd_tables['membership'])
    activation_table = activation_wide(rsp_tables['activation'], selected_rsp_ids)
    scffpe_markers = load_scffpe_markers(SCFFPE_MARKERS)
    dcis1_loading_sets = pairwise_loading_gene_sets(rsp_tables['gene_loadings'], 'RSP0005', 'RSP0019')
    dcis2_loading_sets = pairwise_loading_gene_sets(rsp_tables['gene_loadings'], 'RSP0012', 'RSP0018')
    invasive_loading_sets = pd.concat([pairwise_loading_gene_sets(rsp_tables['gene_loadings'], 'RSP0001', 'RSP0015'), pairwise_loading_gene_sets(rsp_tables['gene_loadings'], 'RSP0001', 'RSP0004')], ignore_index=True)
    single_loading_sets = loading_gene_sets(rsp_tables['gene_loadings'], selected_rsp_ids)
    loading_sets_all = pd.concat([single_loading_sets, dcis1_loading_sets, dcis2_loading_sets, invasive_loading_sets], ignore_index=True)
    pathway_sets = {**load_gmt(HALLMARK_GMT, 'Hallmark'), **load_gmt(REACTOME_GMT, 'Reactome'), **load_gmt(BIOPLANET_GMT, 'BioPlanet')}
    loading_ora_all = loading_gene_set_ora(loading_sets_all, pathway_sets, set(rsp_tables['gene_loadings']['gene'].astype(str)))
    score_set_ids = {'RSP0005_RSP0019_shared_supported', 'RSP0005_weighted_vs_RSP0019_supported', 'RSP0019_weighted_vs_RSP0005_supported', 'RSP0012_RSP0018_shared_supported', 'RSP0012_weighted_vs_RSP0018_supported', 'RSP0018_weighted_vs_RSP0012_supported', 'RSP0001_RSP0015_shared_supported', 'RSP0001_weighted_vs_RSP0015_supported', 'RSP0015_weighted_vs_RSP0001_supported', 'RSP0001_RSP0004_shared_supported', 'RSP0001_weighted_vs_RSP0004_supported', 'RSP0004_weighted_vs_RSP0001_supported', 'RSP0000_core_or_contribution70'}
    score_gene_sets = loading_sets_all.loc[loading_sets_all['gene_set_id'].isin(score_set_ids)].copy()
    targeted_genes = set(score_gene_sets['gene'].astype(str))
    for case_key in ('dcis_dual_identity', 'invasive_tumor_front', 'triple_positive_island'):
        targeted_genes.update(marker_genes_for_case(case_key))
    targeted_genes.update(['ADIPOQ', 'FABP4', 'PLIN1', 'PLIN4', 'LPL', 'CFD', 'G0S2', 'GPD1', 'ADH1B'])
    (targeted_expression, present_targeted_genes) = load_normalized_expression(VISIUM_H5AD, sorted(targeted_genes))
    input_integrity = validate_inputs(selection, rsp_tables, rsmd_tables, spot_context, xenium_composition)
    shared = build_shared_spot_table(spot_context, activation_table, xenium_composition, rsd_wide, targeted_expression)
    (gene_set_scores, gene_set_coverage) = expression_gene_set_scores(targeted_expression, score_gene_sets)
    shared = shared.merge(gene_set_scores, on='spot_id', how='left', validate='one_to_one')
    for required_label in ['DCIS_1', 'DCIS_2', 'Invasive_Tumor', 'Prolif_Invasive_Tumor', 'Myoepi_ACTA2+', 'Myoepi_KRT15+', 'Stromal']:
        for prefix in ('count', 'frac'):
            column = f'{prefix}_{required_label}'
            if column not in shared:
                shared[column] = 0.0
    shared['frac_Invasive_combined'] = shared['frac_Invasive_Tumor'] + shared['frac_Prolif_Invasive_Tumor']
    anchor_threshold = float(selection['analysis_parameters']['dcis2_primary_fraction'])
    tumor_fractions = shared[['frac_DCIS_1', 'frac_DCIS_2', 'frac_Invasive_combined']]
    dominant = tumor_fractions.idxmax(axis=1)
    shared['anchor_DCIS_1'] = dominant.eq('frac_DCIS_1') & shared['frac_DCIS_1'].ge(anchor_threshold)
    shared['anchor_DCIS_2'] = dominant.eq('frac_DCIS_2') & shared['frac_DCIS_2'].ge(anchor_threshold)
    shared['anchor_Invasive'] = dominant.eq('frac_Invasive_combined') & shared['frac_Invasive_combined'].ge(anchor_threshold)
    (edges, graph_qc) = build_spatial_edges(shared)
    input_integrity = pd.concat([input_integrity, pd.DataFrame([{'check': 'shared_spot_count', 'passed': len(shared) == 4992 and shared['spot_id'].is_unique, 'value': len(shared)}, {'check': 'graph_median_degree_near_six', 'passed': graph_qc.iloc[0]['median_degree'] >= 5, 'value': graph_qc.iloc[0]['median_degree']}])], ignore_index=True)
    assert input_integrity['passed'].all()
    assert len(shared) == 4992 and shared['spot_id'].is_unique
    selected_spot_counts = rsd_long.groupby('rsp_id')['spot_id'].nunique().rename('selected_rsd_spot_count')
    object_registry = pd.DataFrame(selected_objects).merge(rsp_tables['rsp'], on='rsp_id', how='left')
    object_registry['selected_domains'] = object_registry['selected_domains'].apply(lambda values: ';'.join(map(str, values)))
    object_registry = object_registry.merge(selected_spot_counts, on='rsp_id', how='left')
    object_registry['selected_rsd_spot_count'] = object_registry['selected_rsd_spot_count'].astype(int)
    current_section = 'DCIS#1'
    dcis1_a = object_by_role('dcis1_layer_a')['rsp_id']
    dcis1_b = object_by_role('dcis1_layer_b')['rsp_id']
    dcis1_scope_set = selected_set(rsd_long, dcis1_a) | selected_set(rsd_long, dcis1_b)
    dcis1_scope = shared['spot_id'].isin(dcis1_scope_set)
    (dcis1_balance, dcis1_joint_summary) = activation_pair_table(shared, dcis1_a, dcis1_b, scope_mask=dcis1_scope)
    dcis1_balance = dcis1_balance.merge(shared[['spot_id', 'anchor_DCIS_1', 'frac_DCIS_1', 'frac_Myoepi_ACTA2+', 'frac_Myoepi_KRT15+', 'frac_Stromal', f'selected_rsd_{dcis1_a}', f'selected_rsd_{dcis1_b}']], on='spot_id', how='left')
    for (column, value) in dcis1_joint_summary.iloc[0].items():
        if column not in {'rsp_a', 'rsp_b'}:
            dcis1_balance[f'summary_{column}'] = value
    dcis1_loading_basis = loading_basis_pair(rsp_tables['gene_loadings'], dcis1_a, dcis1_b)
    dcis1_loading_source = loading_gene_set_marker_attribution(dcis1_loading_sets, scffpe_markers, set(rsp_tables['gene_loadings']['gene'].astype(str)))
    shared = shared.merge(dcis1_balance[['spot_id', 'activation_balance']].rename(columns={'activation_balance': 'dcis1_activation_balance'}), on='spot_id', how='left')
    dcis1_set_ids = set(dcis1_loading_sets['gene_set_id'])
    dcis1_loading_enrichment = loading_ora_all.loc[loading_ora_all['gene_set_id'].isin(dcis1_set_ids)].copy()
    (dcis1_cell_assignments, dcis1_microstructure_geometry, dcis1_clustering_qc) = xenium_cell_microstructures(xenium_mapping, target_label='DCIS_1', eps_um=float(selection['analysis_parameters']['dcis1_xenium_eps_um']), min_samples=int(selection['analysis_parameters']['dcis1_xenium_min_samples']), minimum_cells=int(selection['analysis_parameters']['dcis1_xenium_min_cells']), prefix='DCIS1_X')
    dcis1_contact_radius = float(selection['analysis_parameters']['dcis1_contact_radius_multiplier']) * float(dcis1_clustering_qc.iloc[0]['median_nearest_neighbor_um'])
    dcis1_microstructure_context = xenium_microstructure_context(dcis1_cell_assignments, xenium_mapping, context_labels=['Myoepi_KRT15+', 'Myoepi_ACTA2+', 'Stromal'], contact_radius_um=dcis1_contact_radius)
    (dcis1_microstructure_to_visium, dcis1_visium_structure_summary) = map_xenium_microstructures_to_visium(dcis1_cell_assignments)
    context_wide_parts = []
    for metric in ['context_cell_count_within_radius', 'context_cells_per_100_dcis1', 'contacted_dcis1_fraction', 'median_nearest_context_distance_um']:
        pivot = dcis1_microstructure_context.pivot(index='microstructure_id', columns='context_label', values=metric)
        pivot.columns = [f'{metric}_{column}' for column in pivot.columns]
        context_wide_parts.append(pivot)
    dcis1_context_wide = pd.concat(context_wide_parts, axis=1).reset_index()
    dcis1_microstructure_summary = dcis1_microstructure_geometry.merge(dcis1_context_wide, on='microstructure_id', how='left')
    readout_features = [f'activation_{dcis1_a}', f'activation_{dcis1_b}', 'dcis1_activation_balance', 'frac_DCIS_1', 'frac_Myoepi_KRT15+', 'frac_Myoepi_ACTA2+', 'frac_Stromal']
    dcis1_structure_spot_readout = dcis1_microstructure_to_visium.merge(shared[['spot_id', *readout_features]], on='spot_id', how='left')
    readout_rows = []
    for (microstructure_id, group) in dcis1_structure_spot_readout.groupby('microstructure_id'):
        weights = group['dcis1_cell_count'].to_numpy(float)
        row = {'microstructure_id': microstructure_id, 'visium_spot_count': group['spot_id'].nunique(), 'mapped_dcis1_cells': int(weights.sum())}
        for feature in readout_features:
            values = pd.to_numeric(group[feature], errors='coerce').to_numpy(float)
            valid = np.isfinite(values)
            row[f'weighted_mean_{feature}'] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
        readout_rows.append(row)
    dcis1_structure_readout = dcis1_microstructure_summary.merge(pd.DataFrame(readout_rows), on='microstructure_id', how='left')
    dcis1_visium_structure_spots = dcis1_visium_structure_summary.merge(shared[['spot_id', *readout_features]], on='spot_id', how='left')
    component_spot_counts = dcis1_visium_structure_spots.groupby('dominant_microstructure_id')['spot_id'].transform('size')
    within_structure_spots = dcis1_visium_structure_spots.loc[dcis1_visium_structure_spots['dominant_microstructure_fraction'].ge(0.8) & dcis1_visium_structure_spots['mapped_dcis1_structure_cell_count'].ge(3) & component_spot_counts.ge(4)].copy()
    structure_association_rows = []
    between_context = ['contacted_dcis1_fraction_Myoepi_KRT15+', 'contacted_dcis1_fraction_Myoepi_ACTA2+', 'contacted_dcis1_fraction_Stromal']
    between_outcomes = [f'weighted_mean_activation_{dcis1_a}', f'weighted_mean_activation_{dcis1_b}', 'weighted_mean_dcis1_activation_balance']
    for outcome in between_outcomes:
        for context_variable in between_context:
            (rho, p_value, n) = safe_spearman(dcis1_structure_readout[outcome], dcis1_structure_readout[context_variable])
            structure_association_rows.append({'analysis_level': 'between_microstructures', 'outcome': outcome, 'context_variable': context_variable, 'n_units': n, 'rho': rho, 'p_value': p_value})
    within_outcomes = [f'activation_{dcis1_a}', f'activation_{dcis1_b}', 'dcis1_activation_balance']
    within_context = ['frac_Myoepi_KRT15+', 'frac_Myoepi_ACTA2+', 'frac_Stromal']
    for outcome in within_outcomes:
        outcome_residual = within_structure_spots[outcome] - within_structure_spots.groupby('dominant_microstructure_id')[outcome].transform('mean')
        for context_variable in within_context:
            context_residual = within_structure_spots[context_variable] - within_structure_spots.groupby('dominant_microstructure_id')[context_variable].transform('mean')
            (rho, p_value, n) = safe_spearman(outcome_residual, context_residual)
            structure_association_rows.append({'analysis_level': 'within_microstructure_centered', 'outcome': outcome, 'context_variable': context_variable, 'n_units': n, 'n_microstructures': within_structure_spots['dominant_microstructure_id'].nunique(), 'rho': rho, 'p_value': p_value})
    dcis1_structure_associations = pd.DataFrame(structure_association_rows)
    dcis1_structure_associations['q_value'] = dcis1_structure_associations.groupby(['analysis_level', 'outcome'])['p_value'].transform(benjamini_hochberg)
    dcis1_program_readback = combine_readbacks(('candidate_carrier', 'loading-set cell-source pair', dcis1_loading_source.loc[dcis1_loading_source['source_evidence'].eq('marker_enriched_source')]), ('pathway_annotation', 'loading gene-set pathway pair', significant_pathway_readback(dcis1_loading_enrichment)))
    export_table('dcis1_layer_program_readback.csv', dcis1_program_readback)
    export_table('dcis1_microstructure_readback.csv', dcis1_structure_readout)
    export_table('dcis1_microstructure_associations.csv', dcis1_structure_associations)
    current_section = 'DCIS#2'
    dcis2_epithelial = object_by_role('concentrated_dcis2_epithelial')['rsp_id']
    dcis2_myoepithelial = object_by_role('myoepithelial_with_discrete_dcis2')['rsp_id']
    dcis2_ids = [dcis2_epithelial, dcis2_myoepithelial]
    dcis2_loading_basis = loading_basis_pair(rsp_tables['gene_loadings'], dcis2_epithelial, dcis2_myoepithelial)
    dcis2_loading_source = loading_gene_set_marker_attribution(dcis2_loading_sets, scffpe_markers, set(rsp_tables['gene_loadings']['gene'].astype(str)))
    dcis2_set_ids = set(dcis2_loading_sets['gene_set_id'])
    dcis2_loading_enrichment = loading_ora_all.loc[loading_ora_all['gene_set_id'].isin(dcis2_set_ids)].copy()
    (dcis2_cell_assignments, dcis2_microstructure_geometry, dcis2_clustering_qc) = xenium_cell_microstructures(xenium_mapping, target_label='DCIS_2', eps_um=float(selection['analysis_parameters']['dcis2_xenium_eps_um']), min_samples=int(selection['analysis_parameters']['dcis2_xenium_min_samples']), minimum_cells=int(selection['analysis_parameters']['dcis2_xenium_min_cells']), prefix='DCIS2_X')
    dcis2_microstructure_geometry = dcis2_microstructure_geometry.rename(columns={'dcis1_cell_count': 'dcis2_cell_count'})
    dcis2_contact_radius = float(selection['analysis_parameters']['dcis2_contact_radius_multiplier']) * float(dcis2_clustering_qc.iloc[0]['median_nearest_neighbor_um'])
    dcis2_microstructure_context = xenium_microstructure_context(dcis2_cell_assignments, xenium_mapping, context_labels=['Myoepi_ACTA2+', 'Myoepi_KRT15+', 'Stromal', 'Invasive_Tumor', 'Prolif_Invasive_Tumor'], contact_radius_um=dcis2_contact_radius).rename(columns={'dcis1_cell_count': 'dcis2_cell_count', 'context_cells_per_100_dcis1': 'context_cells_per_100_dcis2', 'contacted_dcis1_fraction': 'contacted_dcis2_fraction'})
    invasive_contact_mapping = xenium_mapping.copy()
    invasive_contact_mapping['label'] = invasive_contact_mapping['label'].astype(str).replace({'Invasive_Tumor': 'Invasive', 'Prolif_Invasive_Tumor': 'Invasive'})
    dcis2_invasive_context = xenium_microstructure_context(dcis2_cell_assignments, invasive_contact_mapping, context_labels=['Invasive'], contact_radius_um=dcis2_contact_radius).rename(columns={'dcis1_cell_count': 'dcis2_cell_count', 'context_cells_per_100_dcis1': 'context_cells_per_100_dcis2', 'contacted_dcis1_fraction': 'contacted_dcis2_fraction'})
    dcis2_microstructure_context = pd.concat([dcis2_microstructure_context, dcis2_invasive_context], ignore_index=True)
    (dcis2_microstructure_to_visium, dcis2_visium_structure_spots) = map_xenium_microstructures_to_visium(dcis2_cell_assignments)
    dcis2_microstructure_to_visium = dcis2_microstructure_to_visium.rename(columns={'dcis1_cell_count': 'dcis2_cell_count'})
    dcis2_visium_structure_spots = dcis2_visium_structure_spots.rename(columns={'mapped_dcis1_structure_cell_count': 'mapped_dcis2_structure_cell_count'})
    context_pivot = dcis2_microstructure_context.pivot(index='microstructure_id', columns='context_label', values='contacted_dcis2_fraction')
    context_pivot.columns = [f'contacted_dcis2_fraction_{column}' for column in context_pivot.columns]
    dcis2_microstructure_summary = dcis2_microstructure_geometry.merge(context_pivot.reset_index(), on='microstructure_id', how='left')
    selected_dcis2_rsd = rsd_long.loc[rsd_long['rsp_id'].isin(dcis2_ids)].copy()
    dcis2_domain_meta = selected_domain_registry.loc[selected_domain_registry['rsp_id'].isin(dcis2_ids)].set_index(['rsp_id', 'domain_id'])
    global_size = dcis2_microstructure_geometry.set_index('microstructure_id')['dcis2_cell_count']
    rsd_microstructure_rows = []
    rsd_summary_rows = []
    small_max = int(selection['analysis_parameters']['dcis2_small_microstructure_max_cells'])
    for ((rsp_id, domain_id), membership) in selected_dcis2_rsd.groupby(['rsp_id', 'domain_id']):
        domain_meta = dcis2_domain_meta.loc[rsp_id, domain_id]
        domain_spots = set(membership['spot_id'].astype(str))
        spot_readout = shared.loc[shared['spot_id'].isin(domain_spots)].copy()
        mapped_cells = xenium_mapping.loc[xenium_mapping['visium_barcode'].astype(str).isin(domain_spots)].copy()
        assigned_cells = dcis2_cell_assignments.loc[dcis2_cell_assignments['microstructure_assigned'].astype(bool) & dcis2_cell_assignments['visium_barcode'].astype(str).isin(domain_spots)].copy()
        counts = assigned_cells.groupby('microstructure_id').size().sort_values(ascending=False)
        total_assigned = int(counts.sum())
        rsd_area_est = float(domain_meta['geo_area_est'])
        microstructure_shares = counts / total_assigned if total_assigned else pd.Series(dtype=float)
        microstructure_hhi = float(np.square(microstructure_shares).sum()) if len(microstructure_shares) else np.nan
        average_carrier_size = total_assigned / len(counts) if len(counts) else np.nan
        dominant_carrier_density = counts.iloc[0] / rsd_area_est if len(counts) and rsd_area_est > 0 else np.nan
        fragmentation_density = 100.0 / (microstructure_hhi * rsd_area_est) if np.isfinite(microstructure_hhi) and microstructure_hhi > 0 and (rsd_area_est > 0) else np.nan
        for (microstructure_id, count) in counts.items():
            rsd_microstructure_rows.append({'rsp_id': rsp_id, 'domain_id': domain_id, 'microstructure_id': microstructure_id, 'dcis2_cells_mapped_inside_rsd': int(count), 'fraction_of_rsd_assigned_dcis2_cells': count / total_assigned if total_assigned else np.nan, 'global_microstructure_dcis2_cells': int(global_size.get(microstructure_id, 0)), 'fraction_of_microstructure_captured_by_rsd': count / global_size.get(microstructure_id, np.nan)})
        label_counts = mapped_cells['label'].astype(str).value_counts()
        supported = counts.loc[counts.ge(3)]
        small_cells = sum((count for (microstructure_id, count) in counts.items() if global_size.get(microstructure_id, np.inf) <= small_max))
        rsd_summary_rows.append({'rsp_id': rsp_id, 'domain_id': domain_id, 'rsd_spot_count': len(domain_spots), 'rsd_centroid_x': domain_meta['geo_centroid_x'], 'rsd_centroid_y': domain_meta['geo_centroid_y'], 'rsd_area_est': rsd_area_est, 'rsd_compactness': domain_meta['geo_compactness'], 'rsd_boundary_ratio': domain_meta['geo_boundary_ratio'], 'rsd_elongation': domain_meta['geo_elongation'], 'rsd_components_count': domain_meta['components_count'], 'mean_activation': spot_readout[f'activation_{rsp_id}'].mean(), 'peak_activation': spot_readout[f'activation_{rsp_id}'].max(), 'mapped_total_cells': len(mapped_cells), 'mapped_DCIS_2_cells': int(label_counts.get('DCIS_2', 0)), 'mapped_Myoepi_ACTA2+_cells': int(label_counts.get('Myoepi_ACTA2+', 0)), 'mapped_Myoepi_KRT15+_cells': int(label_counts.get('Myoepi_KRT15+', 0)), 'mapped_Stromal_cells': int(label_counts.get('Stromal', 0)), 'mapped_Invasive_cells': int(label_counts.get('Invasive_Tumor', 0) + label_counts.get('Prolif_Invasive_Tumor', 0)), 'frac_DCIS_2_cells': label_counts.get('DCIS_2', 0) / len(mapped_cells) if len(mapped_cells) else np.nan, 'frac_Myoepi_ACTA2+_cells': label_counts.get('Myoepi_ACTA2+', 0) / len(mapped_cells) if len(mapped_cells) else np.nan, 'frac_Myoepi_KRT15+_cells': label_counts.get('Myoepi_KRT15+', 0) / len(mapped_cells) if len(mapped_cells) else np.nan, 'frac_Stromal_cells': label_counts.get('Stromal', 0) / len(mapped_cells) if len(mapped_cells) else np.nan, 'frac_Invasive_cells': (label_counts.get('Invasive_Tumor', 0) + label_counts.get('Prolif_Invasive_Tumor', 0)) / len(mapped_cells) if len(mapped_cells) else np.nan, 'assigned_DCIS_2_cells': total_assigned, 'n_overlapping_microstructures': len(counts), 'average_carrier_size': average_carrier_size, 'dominant_carrier_density': dominant_carrier_density, 'microstructure_hhi': microstructure_hhi, 'fragmentation_density': fragmentation_density, 'n_supported_microstructures_ge3_cells': len(supported), 'largest_microstructure_cells_inside_rsd': int(counts.iloc[0]) if len(counts) else 0, 'largest_microstructure_share': counts.iloc[0] / total_assigned if total_assigned else np.nan, 'dominant_microstructure_global_cells': int(global_size.get(counts.index[0], 0)) if len(counts) else 0, 'small_microstructure_cell_fraction': small_cells / total_assigned if total_assigned else np.nan})
    dcis2_rsd_microstructure_mapping = pd.DataFrame(rsd_microstructure_rows)
    dcis2_rsd_summary = pd.DataFrame(rsd_summary_rows).sort_values(['rsp_id', 'domain_id'])
    dcis2_rsd_morphology_export_columns = ['rsp_id', 'domain_id', 'rsd_area_est', 'assigned_DCIS_2_cells', 'n_overlapping_microstructures', 'largest_microstructure_cells_inside_rsd', 'frac_DCIS_2_cells', 'frac_Myoepi_ACTA2+_cells', 'frac_Myoepi_KRT15+_cells', 'frac_Stromal_cells', 'frac_Invasive_cells', 'average_carrier_size', 'dominant_carrier_density', 'microstructure_hhi', 'fragmentation_density']
    dcis2_rsd_morphology_export = dcis2_rsd_summary[dcis2_rsd_morphology_export_columns].copy()
    micro_readout_rows = []
    for (microstructure_id, group) in dcis2_microstructure_to_visium.groupby('microstructure_id'):
        weights = group['dcis2_cell_count'].to_numpy(float)
        local = group.merge(shared[['spot_id', f'activation_{dcis2_epithelial}', f'activation_{dcis2_myoepithelial}']], on='spot_id', how='left')
        row = {'microstructure_id': microstructure_id, 'mapped_visium_spots': local['spot_id'].nunique()}
        for rsp_id in dcis2_ids:
            values = local[f'activation_{rsp_id}'].to_numpy(float)
            row[f'weighted_mean_activation_{rsp_id}'] = float(np.average(values, weights=weights))
            selected_spots = set(selected_dcis2_rsd.loc[selected_dcis2_rsd['rsp_id'].eq(rsp_id), 'spot_id'].astype(str))
            row[f'cell_fraction_inside_selected_{rsp_id}_rsds'] = local.loc[local['spot_id'].isin(selected_spots), 'dcis2_cell_count'].sum() / weights.sum()
        micro_readout_rows.append(row)
    dcis2_microstructure_readout = dcis2_microstructure_summary.merge(pd.DataFrame(micro_readout_rows), on='microstructure_id', how='left')
    overlap_threshold = float(selection['analysis_parameters']['dcis2_rsd_overlap_fraction'])
    dcis2_microstructure_readout['is_small_structure'] = dcis2_microstructure_readout['dcis2_cell_count'].le(small_max)
    dcis2_microstructure_readout['small_structure_group'] = np.select([dcis2_microstructure_readout['is_small_structure'] & dcis2_microstructure_readout[f'cell_fraction_inside_selected_{dcis2_myoepithelial}_rsds'].ge(overlap_threshold), dcis2_microstructure_readout['is_small_structure']], ['RSP0018-associated', 'other-small'], default='not-in-comparison')
    small_microstructures = dcis2_microstructure_readout.loc[dcis2_microstructure_readout['is_small_structure']].copy()
    small_features = ['contacted_dcis2_fraction_Myoepi_ACTA2+', 'contacted_dcis2_fraction_Myoepi_KRT15+', 'contacted_dcis2_fraction_Stromal']
    comparison_rows = []
    for feature in small_features:
        left = small_microstructures.loc[small_microstructures['small_structure_group'].eq('RSP0018-associated'), feature].dropna()
        right = small_microstructures.loc[small_microstructures['small_structure_group'].eq('other-small'), feature].dropna()
        (statistic, p_value) = stats.mannwhitneyu(left, right, alternative='two-sided') if len(left) and len(right) else (np.nan, np.nan)
        comparison_rows.append({'feature': feature, 'n_rsp0018_associated': len(left), 'median_rsp0018_associated': left.median(), 'n_other_small': len(right), 'median_other_small': right.median(), 'mannwhitney_u': statistic, 'p_value': p_value})
    dcis2_small_structure_comparison = pd.DataFrame(comparison_rows)
    dcis2_small_structure_comparison['q_value'] = benjamini_hochberg(dcis2_small_structure_comparison['p_value'])
    d000048_spots = set(selected_dcis2_rsd.loc[selected_dcis2_rsd['domain_id'].eq('D000048'), 'spot_id'].astype(str))
    dcis2_d000048 = shared.loc[shared['spot_id'].isin(d000048_spots), ['spot_id', 'x_fullres', 'y_fullres', f'activation_{dcis2_epithelial}', 'frac_DCIS_2', 'frac_Invasive_combined', 'frac_Myoepi_ACTA2+', 'frac_Myoepi_KRT15+']].copy()
    d000048_associations = []
    for variable in ['frac_DCIS_2', 'frac_Invasive_combined', 'frac_Myoepi_ACTA2+', 'frac_Myoepi_KRT15+']:
        (rho, p_value, n) = safe_spearman(dcis2_d000048[f'activation_{dcis2_epithelial}'], dcis2_d000048[variable])
        d000048_associations.append({'variable': variable, 'n_spots': n, 'rho': rho, 'p_value': p_value})
    dcis2_d000048_associations = pd.DataFrame(d000048_associations)
    dcis2_d000048_associations['q_value'] = benjamini_hochberg(dcis2_d000048_associations['p_value'])
    dcis2_rsd_spot_sets = {domain_id: set(group['spot_id'].astype(str)) for (domain_id, group) in selected_dcis2_rsd.loc[selected_dcis2_rsd['rsp_id'].eq(dcis2_epithelial)].groupby('domain_id')}
    dcis2_rsd_expression = load_raw_expression_for_spots(VISIUM_H5AD, sorted(set().union(*dcis2_rsd_spot_sets.values())))

    def run_rsd_dge(case_id: str, reference_ids: list[str]) -> pd.DataFrame:
        case_spots = dcis2_rsd_spot_sets[case_id]
        reference_spots = set().union(*(dcis2_rsd_spot_sets[item] for item in reference_ids))
        case_raw = dcis2_rsd_expression.loc[sorted(case_spots)]
        reference_raw = dcis2_rsd_expression.loc[sorted(reference_spots)]
        case = case_raw.div(case_raw.sum(axis=1).replace(0, np.nan), axis=0).mul(10000.0)
        reference = reference_raw.div(reference_raw.sum(axis=1).replace(0, np.nan), axis=0).mul(10000.0)
        (statistic, p_values) = stats.ttest_ind(np.log1p(case), np.log1p(reference), axis=0, equal_var=False, nan_policy='omit')
        p_values = np.asarray(p_values, float)
        p_values[~np.isfinite(p_values)] = 1.0
        (mean_case, mean_reference) = (case.mean(axis=0), reference.mean(axis=0))
        out = pd.DataFrame({'contrast_id': f"{case_id}_vs_{'_'.join(reference_ids)}", 'case_domain': case_id, 'reference_domains': ';'.join(reference_ids), 'gene': dcis2_rsd_expression.columns, 'n_case_spots': len(case), 'n_reference_spots': len(reference), 'mean_case': mean_case.values, 'mean_reference': mean_reference.values, 'log2fc_case_vs_reference': np.log2((mean_case.values + 0.1) / (mean_reference.values + 0.1)), 't_statistic': statistic, 'p_value': p_values, 'detected_fraction_case': case_raw.gt(0).mean(axis=0).values, 'detected_fraction_reference': reference_raw.gt(0).mean(axis=0).values})
        out['q_value'] = benjamini_hochberg(out['p_value'])
        out['abs_log2fc'] = out['log2fc_case_vs_reference'].abs()
        return out.sort_values(['q_value', 'p_value', 'abs_log2fc'], ascending=[True, True, False]).reset_index(drop=True)
    dcis2_rsd_dge = run_rsd_dge('D000048', ['D000049', 'D000052'])
    dcis2_rsd_dge_signature = dcis2_rsd_dge.loc[dcis2_rsd_dge['q_value'].le(0.05) & dcis2_rsd_dge['abs_log2fc'].ge(0.5)].copy()
    dcis2_rsd_dge_signature['direction'] = np.where(dcis2_rsd_dge_signature['log2fc_case_vs_reference'].gt(0), 'D000048_up', 'D000048_down')
    dcis2_rsd_dge_signature = dcis2_rsd_dge_signature.sort_values(['direction', 'q_value', 'abs_log2fc'], ascending=[True, True, False]).reset_index(drop=True)
    ora_query_rows = []
    for (direction, group) in dcis2_rsd_dge_signature.groupby('direction'):
        selected = group.sort_values(['q_value', 'abs_log2fc'], ascending=[True, False]).head(500)
        for (rank, gene) in enumerate(selected['gene'].astype(str), start=1):
            ora_query_rows.append({'gene_set_id': direction, 'gene_set_type': 'pooled_reference_rsd_dge', 'rsp_id': dcis2_epithelial, 'scope': 'D000048_vs_pooled_D000049_D000052_top500', 'gene': gene, 'gene_set_rank': rank})
    dcis2_rsd_dge_gene_sets = pd.DataFrame(ora_query_rows)
    dcis2_rsd_dge_enrichment = loading_gene_set_ora(dcis2_rsd_dge_gene_sets, pathway_sets, set(dcis2_rsd_expression.columns.astype(str)))
    assert len(dcis2_rsd_summary) == 6
    assert not dcis2_microstructure_readout.empty and (not dcis2_loading_source.empty)
    dcis2_program_readback = combine_readbacks(('candidate_carrier', 'loading-set cell-source pair', dcis2_loading_source.loc[dcis2_loading_source['source_evidence'].eq('marker_enriched_source')]), ('pathway_annotation', 'loading gene-set pathway pair', significant_pathway_readback(dcis2_loading_enrichment)))
    dcis2_entity_state_de = dcis2_rsd_dge.copy()
    dcis2_entity_state_de['is_entity_state_signature'] = dcis2_entity_state_de['q_value'].le(0.05) & dcis2_entity_state_de['abs_log2fc'].ge(0.5)
    dcis2_entity_state_de['direction'] = np.select([dcis2_entity_state_de['is_entity_state_signature'] & dcis2_entity_state_de['log2fc_case_vs_reference'].gt(0), dcis2_entity_state_de['is_entity_state_signature'] & dcis2_entity_state_de['log2fc_case_vs_reference'].lt(0)], ['D000048_up', 'D000048_down'], default='not_signature')
    export_table('dcis2_layer_program_readback.csv', dcis2_program_readback)
    export_table('dcis2_rsd_morphology.csv', dcis2_rsd_morphology_export)
    export_table('dcis2_microstructure_readback.csv', dcis2_microstructure_readout)
    export_table('dcis2_small_structure_comparison.csv', dcis2_small_structure_comparison)
    export_table('dcis2_entity_state_de_readback.csv', dcis2_entity_state_de)
    export_table('dcis2_entity_state_pathway_readback.csv', dcis2_rsd_dge_enrichment)

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
    _CONFIG = document['fig06']
    prepare()
    missing = [name for name in TABLE_NAMES if not (TABLE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f'Missing prepared tables: {missing}')
    print(TABLE_DIR)

if __name__ == '__main__':
    main()
