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

from plot_fig05 import RESOURCE_PATHS

from plot_fig05 import FIGURE_SELECTION

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig05'
TABLE_NAMES = ('dcis_lcis_overlay_spot_readback.csv', 'dcis_lcis_overlay_statistics.csv', 'dcis_lcis_vs_idc_de_readback.csv', 'idc2_region_a_vs_b_de_readback.csv', 'loading_gene_go_annotation.csv', 'rsp0009_entity_realized_contribution.csv')

def _resolve(value):
    result = Path(Template(value).substitute(_ROOTS)).expanduser()
    return result if result.is_absolute() else _CONFIG_DIR / result

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
from typing import Any
import yaml
import os
import warnings
import scanpy as sc
from scipy import sparse, stats
DEFAULT_OBS_COLUMNS = ('manual_annotation', 'manual_annotation_fine', 'rctd_dominant', 'rctd_dominant_prop', 'rctd_epithelial_prop', 'rctd_cancer_epithelial_prop', 'rctd_normal_epithelial_prop', 'copykat_pred_default', 'copykat_pred_tirosh', 'cnv_score', 'cnv_cor', 'final_dominant', 'final_dominant_prop', 'final_flag')

def load_spot_context(h5ad_path: str | Path, *, sample_id: str, obs_columns: Iterable[str]=DEFAULT_OBS_COLUMNS, spatial_key: str='spatial') -> pd.DataFrame:
    """Load coordinates and selected obs context columns from an h5ad file."""
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        out = pd.DataFrame({'spot_id': adata.obs_names.astype(str), 'sample_id': sample_id})
        if spatial_key in adata.obsm:
            spatial = np.asarray(adata.obsm[spatial_key])[:, :2]
            out['x'] = spatial[:, 0].astype(float)
            out['y'] = spatial[:, 1].astype(float)
        elif {'pxl_col_in_fullres', 'pxl_row_in_fullres'}.issubset(adata.obs.columns):
            out['x'] = pd.to_numeric(adata.obs['pxl_col_in_fullres'], errors='coerce').to_numpy(float)
            out['y'] = pd.to_numeric(adata.obs['pxl_row_in_fullres'], errors='coerce').to_numpy(float)
        else:
            raise ValueError(f'Missing spatial coordinates in {h5ad_path}')
        for col in obs_columns:
            if col not in adata.obs:
                continue
            values = adata.obs[col]
            if pd.api.types.is_numeric_dtype(values):
                out[col] = pd.to_numeric(values, errors='coerce').to_numpy()
            else:
                out[col] = values.astype(str).to_numpy()
    finally:
        adata.file.close()
    return out

def load_normalized_expression(h5ad_path: str | Path, genes: Iterable[str] | None=None, *, target_sum: float=10000.0, layer: str | None=None) -> tuple[pd.DataFrame, list[str]]:
    """Return log1p library-normalized expression for selected genes or all genes."""
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        requested = None if genes is None else list(dict.fromkeys((str(g) for g in genes if str(g))))
        if requested is None:
            selected = adata.var_names.astype(str).tolist()
        else:
            selected = [gene for gene in requested if gene in adata.var_names]
        sub = adata[:, selected].to_memory()
        matrix = sub.layers[layer] if layer and layer in sub.layers else sub.X
        values = h5ad_utils__dense(matrix).astype(float)
        totals = _row_totals(adata)
        totals = np.maximum(totals.astype(float), 1.0)
        values = np.log1p(values / totals[:, None] * float(target_sum))
        expr = pd.DataFrame(values, index=sub.obs_names.astype(str), columns=selected)
    finally:
        adata.file.close()
    return (expr, selected)

def h5ad_utils__dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)

def _row_totals(adata: ad.AnnData) -> np.ndarray:
    if 'total_counts' in adata.obs:
        return pd.to_numeric(adata.obs['total_counts'], errors='coerce').fillna(0).to_numpy(dtype=float)
    matrix = adata.X
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if hasattr(matrix, 'to_memory'):
        in_memory = matrix.to_memory()
        if sparse.issparse(in_memory):
            return np.asarray(in_memory.sum(axis=1)).ravel().astype(float)
        return np.asarray(in_memory).sum(axis=1).astype(float)
    return np.asarray(matrix).sum(axis=1).astype(float)
REQUIRED_RSP_FILES = {'rsp': 'rsp.parquet', 'activation': 'rsp_activation.parquet', 'gene_loadings': 'rsp_gene_loadings.parquet', 'core_gene_loadings': 'rsp_core_gene_loadings.parquet'}
REQUIRED_RSMD_FILES = {'domains': 'domains.parquet', 'membership': 'domain_spot_membership.parquet', 'metadomains': 'metadomains.parquet'}

def load_selection():
    selection = deepcopy(FIGURE_SELECTION)
    selection['sample'].update(h5ad_path=str(path('h5ad')), sample_dir=str(path('sample_dir')))
    selection['verification_references'] = {'deepst_deg_table': {'path': str(resource('deepst_deg_table'))}}
    return selection

def rsp_out_dir(sample_dir: str | Path) -> Path:
    path = Path(sample_dir)
    return path if path.name == 'rsp_out' else path / 'rsp_out'

def rsmd_out_dir(sample_dir: str | Path) -> Path:
    path = Path(sample_dir)
    return path if path.name == 'rsmd_out' else path / 'rsmd_out'

def load_rsp_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = rsp_out_dir(sample_dir)
    tables: dict[str, pd.DataFrame] = {}
    for (key, filename) in REQUIRED_RSP_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing RSP table: {path}')
        tables[key] = pd.read_parquet(path)
    return tables

def load_rsmd_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = rsmd_out_dir(sample_dir)
    tables: dict[str, pd.DataFrame] = {}
    for (key, filename) in REQUIRED_RSMD_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing RSMD table: {path}')
        tables[key] = pd.read_parquet(path)
    return tables

def activation_matrix(rsp_activation: pd.DataFrame) -> pd.DataFrame:
    mat = rsp_activation.pivot_table(index='spot_id', columns='rsp_id', values='activation', aggfunc='first')
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    return mat.fillna(0.0).astype(float)

def parse_rsd_ref(rsd_ref: str) -> tuple[str, str]:
    parts = str(rsd_ref).split(':')
    if len(parts) != 2 or not parts[0].startswith('RSP') or (not parts[1].startswith('D')):
        raise ValueError(f'Invalid RSD reference {rsd_ref!r}; expected RSPxxxx:Dxxxxx')
    return (parts[0], parts[1])

def selected_rsd_context(selection: dict[str, Any], domains: pd.DataFrame, membership: pd.DataFrame, spot_context: pd.DataFrame) -> pd.DataFrame:
    domain_index = domains.set_index(['rsp_seed_id', 'domain_id'], drop=False)
    spot_context_indexed = spot_context.set_index('spot_id', drop=False)
    rows: list[dict[str, Any]] = []
    for item in selection['selected_rsp_objects']:
        rsp_id = str(item['rsp_id'])
        for rsd_ref in item.get('rsds', []):
            (ref_rsp, domain_id) = parse_rsd_ref(str(rsd_ref))
            if ref_rsp != rsp_id:
                raise ValueError(f'RSD {rsd_ref} is nested under {rsp_id}')
            if (rsp_id, domain_id) not in domain_index.index:
                raise ValueError(f'Selected RSD not found in domains.parquet: {rsd_ref}')
            dom = domain_index.loc[rsp_id, domain_id]
            spots = domain_spots(membership, str(dom['domain_key']))
            sub = spot_context_indexed.loc[[s for s in spots if s in spot_context_indexed.index]]
            rows.append({'sample_id': selection['sample']['sample_id'], 'rsp_id': rsp_id, 'rsmd_id': str(dom['rsmd_id']), 'domain_id': domain_id, 'rsd_ref': rsd_ref, 'domain_key': str(dom['domain_key']), 'spot_count': int(len(spots)), 'centroid_x': _float(dom.get('geo_centroid_x')), 'centroid_y': _float(dom.get('geo_centroid_y')), 'domain_reliability': _float(dom.get('domain_reliability')), 'qc_pass': bool(dom.get('qc_pass', True)), 'landscape_relations': ';'.join(map(str, item.get('landscape_relations', []))), 'used_in_panels': ';'.join(map(str, item.get('used_in_panels', []))), 'narrative_role': str(item.get('narrative_role', '')), 'dominant_annotation_fine': _dominant(sub, 'manual_annotation_fine'), 'dominant_annotation_fine_fraction': _dominant_fraction(sub, 'manual_annotation_fine'), 'dominant_final_context': _dominant(sub, 'final_dominant'), 'dominant_final_context_fraction': _dominant_fraction(sub, 'final_dominant'), 'mean_rctd_cancer_epithelial_prop': _mean(sub, 'rctd_cancer_epithelial_prop'), 'mean_rctd_epithelial_prop': _mean(sub, 'rctd_epithelial_prop'), 'mean_cnv_score': _mean(sub, 'cnv_score'), 'dominant_copykat_default': _dominant(sub, 'copykat_pred_default')})
    return pd.DataFrame(rows)

def check_inputs(selection: dict[str, Any]) -> pd.DataFrame:
    sample = selection['sample']
    sample_dir = Path(sample['sample_dir'])
    h5ad = Path(sample['h5ad_path'])
    rsp_root = rsp_out_dir(sample_dir)
    rsmd_root = rsmd_out_dir(sample_dir)
    missing_rsp = [f for f in REQUIRED_RSP_FILES.values() if not (rsp_root / f).exists()]
    missing_rsmd = [f for f in REQUIRED_RSMD_FILES.values() if not (rsmd_root / f).exists()]
    refs = selection.get('verification_references', {})
    missing_refs = [str(v.get('path', '')) for v in refs.values() if not Path(v.get('path', '')).exists()]
    return pd.DataFrame([{'sample_id': sample['sample_id'], 'h5ad_path': str(h5ad), 'sample_dir': str(sample_dir), 'h5ad_exists': h5ad.exists(), 'rsp_out': str(rsp_root), 'rsmd_out': str(rsmd_root), 'missing_rsp_files': ';'.join(missing_rsp), 'missing_rsmd_files': ';'.join(missing_rsmd), 'missing_verification_references': ';'.join(missing_refs), 'inputs_ready': h5ad.exists() and (not missing_rsp) and (not missing_rsmd)}])

def domain_spots(membership: pd.DataFrame, domain_key: str) -> list[str]:
    return membership.loc[membership['domain_key'].astype(str).eq(str(domain_key)), 'spot_id'].astype(str).tolist()

def reported_loading_genes(gene_loadings: pd.DataFrame, rsp_id: str, n: int=20) -> list[str]:
    sub = _loadings_for(gene_loadings, rsp_id)
    return sub.sort_values('loading_rank', kind='mergesort')['gene'].astype(str).head(n).tolist()

def contribution_genes(gene_loadings: pd.DataFrame, rsp_id: str, flag_col: str='in_contribution_70', n: int=50) -> list[str]:
    sub = _loadings_for(gene_loadings, rsp_id)
    if flag_col in sub.columns:
        sub = sub.loc[sub[flag_col].fillna(False).astype(bool)]
    return sub.sort_values('loading_rank', kind='mergesort')['gene'].astype(str).head(n).tolist()

def core_genes(core_gene_loadings: pd.DataFrame, rsp_id: str, n: int=100) -> list[str]:
    sub = core_gene_loadings.loc[core_gene_loadings['rsp_id'].astype(str).eq(str(rsp_id))].copy()
    if sub.empty:
        return []
    sort_col = 'core_rank' if 'core_rank' in sub.columns else 'loading_rank'
    return sub.sort_values(sort_col, kind='mergesort')['gene'].astype(str).head(n).tolist()

def _loadings_for(gene_loadings: pd.DataFrame, rsp_id: str) -> pd.DataFrame:
    return gene_loadings.loc[gene_loadings['rsp_id'].astype(str).eq(str(rsp_id))].copy()

def _dominant(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df:
        return ''
    vc = df[col].astype(str).value_counts(dropna=False)
    return str(vc.index[0]) if len(vc) else ''

def _dominant_fraction(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return float('nan')
    vc = df[col].astype(str).value_counts(dropna=False)
    return float(vc.iloc[0] / len(df)) if len(vc) else float('nan')

def _mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return float('nan')
    return float(pd.to_numeric(df[col], errors='coerce').mean())

def _float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float('nan')
        return float(value)
    except Exception:
        return float('nan')

def prepare():
    GO_GMT_PATH = resource('GO_Biological_Process_2023.gmt')
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-r3')
    warnings.filterwarnings('ignore', category=FutureWarning)
    DEG_PADJ_CUTOFF = 0.05
    DEG_ABS_LOGFC_CUTOFF = 2.0
    IMMUNE_MARKERS = {'pan_leukocyte': ['PTPRC'], 't_cell': ['CD3D', 'CD3E', 'TRAC'], 'b_cell_apc': ['MS4A1', 'CD79A', 'CD74'], 'myeloid': ['LST1', 'TYROBP', 'FCER1G'], 'nk_cytotoxic': ['NKG7', 'GNLY']}
    IMMUNE_MARKER_GENES = [gene for genes in IMMUNE_MARKERS.values() for gene in genes]
    IMMUNE_MARKER_SET_ID = 'fixed_immune_marker_panel_v1'
    selection = load_selection()
    sample = dict(selection['sample'], h5ad_path=str(path('h5ad')), sample_dir=str(path('sample_dir')))
    selection['sample'] = sample
    sample_id = sample['sample_id']
    h5ad_path = Path(sample['h5ad_path'])
    sample_dir = Path(sample['sample_dir'])
    input_integrity = check_inputs(selection)
    if not bool(input_integrity['inputs_ready'].iloc[0]):
        raise RuntimeError(input_integrity.to_string(index=False))
    rsp_tables = load_rsp_tables(sample_dir)
    rsmd_tables = load_rsmd_tables(sample_dir)
    rsp_activation = rsp_tables['activation']
    rsp_gene_loadings = rsp_tables['gene_loadings']
    rsp_core_gene_loadings = rsp_tables['core_gene_loadings']
    domains = rsmd_tables['domains']
    membership = rsmd_tables['membership']
    activation = activation_matrix(rsp_activation)
    spot_context = load_spot_context(h5ad_path, sample_id=sample_id)
    selected_rsd_context_table = selected_rsd_context(selection, domains, membership, spot_context)
    domain_key_by_ref = dict(zip(selected_rsd_context_table['rsd_ref'], selected_rsd_context_table['domain_key']))
    spots_by_ref = {ref: set(domain_spots(membership, key)) for (ref, key) in domain_key_by_ref.items()}
    adata = ad.read_h5ad(h5ad_path)
    gene_names = adata.var_names.astype(str).to_numpy()
    spot_index = pd.Index(adata.obs_names.astype(str))
    spot_to_pos = pd.Series(np.arange(adata.n_obs), index=spot_index)

    def bh_adjust(pvals):
        pvals = np.asarray(pvals, dtype=float)
        pvals = np.where(np.isfinite(pvals), pvals, 1.0)
        n = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        adjusted = np.empty(n, dtype=float)
        cumulative = 1.0
        for i in range(n - 1, -1, -1):
            rank = i + 1
            cumulative = min(cumulative, ranked[i] * n / rank)
            adjusted[order[i]] = cumulative
        return np.clip(adjusted, 0, 1)

    def _dense(matrix):
        return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)

    def normalized_log_subset(spot_ids):
        idx = spot_to_pos.loc[[s for s in spot_ids if s in spot_to_pos.index]].to_numpy()
        x = adata.X[idx, :]
        if sparse.issparse(x):
            x = x.tocsr().astype(float)
            totals = np.asarray(x.sum(axis=1)).ravel()
            factors = np.divide(10000.0, np.maximum(totals, 1.0))
            x = x.multiply(factors[:, None]).tocsr()
            x.data = np.log1p(x.data)
            return (x.toarray().astype(np.float32), idx)
        x = np.asarray(x, dtype=float)
        totals = np.maximum(x.sum(axis=1), 1.0)
        x = np.log1p(x / totals[:, None] * 10000.0)
        return (x.astype(np.float32), idx)

    def run_region_de(contrast_id, region_a, spots_a, region_b, spots_b):
        spots_a = sorted(set(spots_a))
        spots_b = sorted(set(spots_b))
        if len(spots_a) < 3 or len(spots_b) < 3:
            return pd.DataFrame()
        selected = [spot for spot in spots_a + spots_b if spot in spot_to_pos.index]
        (x, _) = normalized_log_subset(selected)
        labels = np.where(pd.Index(selected).isin(spots_a), region_a, region_b)
        sub = ad.AnnData(X=x, obs=pd.DataFrame(index=pd.Index(selected, name='spot_id')), var=pd.DataFrame(index=pd.Index(gene_names, name='gene')))
        sub.obs['de_group'] = pd.Categorical(labels, categories=[region_a, region_b])
        sc.tl.rank_genes_groups(sub, groupby='de_group', groups=[region_a], reference=region_b, method='wilcoxon', pts=True, tie_correct=True, n_genes=sub.n_vars)
        out = sc.get.rank_genes_groups_df(sub, group=region_a).rename(columns={'names': 'gene'})
        out.insert(0, 'contrast_id', contrast_id)
        out.insert(1, 'contrast_level', 'region')
        out.insert(2, 'region_definition', 'RSD_membership')
        out.insert(3, 'region_a', region_a)
        out.insert(4, 'region_b', region_b)
        out.insert(5, 'interpretation_scope', 'regional_expression_difference')
        out['n_a'] = int((labels == region_a).sum())
        out['n_b'] = int((labels == region_b).sum())
        out['higher_group'] = np.where(out['logfoldchanges'] >= 0, region_a, region_b)
        out['abs_logfoldchanges'] = out['logfoldchanges'].abs()
        out['passes_de_threshold'] = (out['pvals_adj'] < DEG_PADJ_CUTOFF) & (out['abs_logfoldchanges'] >= DEG_ABS_LOGFC_CUTOFF)
        out['minus_log10_padj'] = -np.log10(out['pvals_adj'].clip(lower=1e-300))
        return out.sort_values(['pvals_adj', 'abs_logfoldchanges'], ascending=[True, False]).reset_index(drop=True)

    def load_gmt(path):
        gene_sets = {}
        with Path(path).open() as handle:
            for line in handle:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3:
                    gene_sets[parts[0]] = set((g.upper() for g in parts[2:] if g))
        return gene_sets

    def go_id_from_term(term):
        match = pd.Series([str(term)]).str.extract('(GO:\\d+)').iloc[0, 0]
        return match if pd.notna(match) else ''

    def go_name_from_term(term):
        return pd.Series([str(term)]).str.replace('\\s*\\(GO:\\d+\\)\\s*$', '', regex=True).iloc[0].strip().lower()

    def run_go_enrichment(de, contrast_id, min_overlap=3, min_set_size=10, max_set_size=500):
        if de.empty:
            return pd.DataFrame()
        gene_sets = load_gmt(GO_GMT_PATH)
        universe = set(de['gene'].astype(str).str.upper())
        sig = de[(de['pvals_adj'] < DEG_PADJ_CUTOFF) & (de['abs_logfoldchanges'] >= DEG_ABS_LOGFC_CUTOFF)].copy()
        rows = []
        for (direction, sdf) in sig.groupby('higher_group', sort=False):
            query = set(sdf['gene'].astype(str).str.upper()) & universe
            if len(query) < min_overlap:
                continue
            for (term, raw_genes) in gene_sets.items():
                term_genes = raw_genes & universe
                if not min_set_size <= len(term_genes) <= max_set_size:
                    continue
                overlap = query & term_genes
                if len(overlap) < min_overlap:
                    continue
                a = len(overlap)
                b = len(query - term_genes)
                c = len(term_genes - query)
                d = len(universe - query - term_genes)
                (oddsratio, pval) = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                rows.append({'contrast_id': contrast_id, 'direction': direction, 'go_term': term, 'go_id': go_id_from_term(term), 'go_name_normalized': go_name_from_term(term), 'overlap_n': a, 'query_n': len(query), 'term_n': len(term_genes), 'universe_n': len(universe), 'oddsratio': oddsratio, 'pval': pval, 'overlap_genes': ';'.join(sorted(overlap))})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out['p_adj'] = bh_adjust(out['pval'].to_numpy())
        return out.sort_values(['p_adj', 'oddsratio'], ascending=[True, False]).reset_index(drop=True)

    def run_loading_gene_go_annotation(rsp_ids, contribution_flag='in_contribution_70', min_overlap=3, min_set_size=10, max_set_size=500):
        gene_sets = load_gmt(GO_GMT_PATH)
        universe = set(adata.var_names.astype(str).str.upper())
        rows = []
        for rsp_id in rsp_ids:
            load = rsp_gene_loadings[(rsp_gene_loadings['rsp_id'].astype(str) == rsp_id) & rsp_gene_loadings[contribution_flag].fillna(False).astype(bool) & ~rsp_gene_loadings['is_blacklist_gene'].fillna(False).astype(bool) & ~rsp_gene_loadings['is_housekeeping_gene'].fillna(False).astype(bool) & ~rsp_gene_loadings['is_low_info_gene'].fillna(False).astype(bool)].copy()
            query = set(load['gene'].astype(str).str.upper()) & universe
            if len(query) < min_overlap:
                continue
            for (term, raw_genes) in gene_sets.items():
                term_genes = raw_genes & universe
                if not min_set_size <= len(term_genes) <= max_set_size:
                    continue
                overlap = query & term_genes
                if len(overlap) < min_overlap:
                    continue
                a = len(overlap)
                b = len(query - term_genes)
                c = len(term_genes - query)
                d = len(universe - query - term_genes)
                (oddsratio, pval) = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                rows.append({'rsp_id': rsp_id, 'query_set': contribution_flag, 'go_term': term, 'go_id': go_id_from_term(term), 'go_name_normalized': go_name_from_term(term), 'overlap_n': a, 'query_n': len(query), 'term_n': len(term_genes), 'universe_n': len(universe), 'oddsratio': oddsratio, 'pval': pval, 'overlap_genes': ';'.join(sorted(overlap))})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out['p_adj'] = out.groupby('rsp_id')['pval'].transform(lambda x: bh_adjust(x.to_numpy()))
        return out.sort_values(['rsp_id', 'p_adj', 'oddsratio'], ascending=[True, True, False]).reset_index(drop=True)

    def context_card(region, spots, rsp_ids=()):
        spots = sorted(set(spots))
        sub = spot_context.loc[spot_context['spot_id'].isin(spots)].copy()
        row = {'region': region, 'spot_count': len(spots)}
        for col in ['manual_annotation', 'manual_annotation_fine', 'final_dominant', 'copykat_pred_default', 'copykat_pred_tirosh']:
            if col in sub and len(sub):
                vc = sub[col].astype(str).value_counts(dropna=False)
                row[f'dominant_{col}'] = vc.index[0]
                row[f'dominant_{col}_fraction'] = vc.iloc[0] / len(sub)
        for col in ['rctd_epithelial_prop', 'rctd_cancer_epithelial_prop', 'rctd_normal_epithelial_prop', 'cnv_score', 'final_dominant_prop']:
            if col in sub:
                row[f'mean_{col}'] = pd.to_numeric(sub[col], errors='coerce').mean()
        for rsp_id in rsp_ids:
            if rsp_id in activation.columns:
                row[f'mean_activation_{rsp_id}'] = activation.loc[[s for s in spots if s in activation.index], rsp_id].mean()
        return row

    def overlap_row(label_a, spots_a, label_b, spots_b):
        (a, b) = (set(spots_a), set(spots_b))
        inter = a & b
        union = a | b
        return {'entity_a': label_a, 'entity_b': label_b, 'n_a': len(a), 'n_b': len(b), 'overlap_n': len(inter), 'jaccard': len(inter) / len(union) if union else np.nan, 'fraction_of_a': len(inter) / len(a) if a else np.nan, 'fraction_of_b': len(inter) / len(b) if b else np.nan}

    def mean_expression_for_genes(spots, genes):
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            return pd.Series(dtype=float)
        (mat, _) = normalized_log_subset(sorted(spots))
        gene_pos = pd.Series(np.arange(len(gene_names)), index=gene_names).loc[genes].to_numpy()
        return pd.Series(mat[:, gene_pos].mean(axis=0), index=genes)

    def realized_contribution_table(rsp_id, region_spots, contribution_flag='in_contribution_90'):
        load = rsp_gene_loadings[(rsp_gene_loadings['rsp_id'].astype(str) == rsp_id) & ~rsp_gene_loadings['is_blacklist_gene'].fillna(False).astype(bool) & ~rsp_gene_loadings['is_housekeeping_gene'].fillna(False).astype(bool) & ~rsp_gene_loadings['is_low_info_gene'].fillna(False).astype(bool)].copy()
        if contribution_flag in load.columns:
            load = load.loc[load[contribution_flag].fillna(False).astype(bool)].copy()
        genes = load['gene'].astype(str).tolist()
        rows = []
        for (region, spots) in region_spots.items():
            means = mean_expression_for_genes(spots, genes)
            sub = load.loc[load['gene'].astype(str).isin(means.index)].copy()
            sub['region'] = region
            sub['mean_logexpr_in_region'] = sub['gene'].map(means)
            sub['realized_contribution'] = sub['loading_fraction'].astype(float) * sub['mean_logexpr_in_region'].astype(float)
            rows.append(sub)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    selected_rsp_ids = [str(item['rsp_id']) for item in selection['selected_rsp_objects']]
    loading_gene_go_annotation = run_loading_gene_go_annotation(selected_rsp_ids, contribution_flag='in_contribution_70')
    write_table(loading_gene_go_annotation, TABLE_DIR / 'loading_gene_go_annotation.csv')
    loading_gene_go_annotation.groupby('rsp_id').head(5)
    rsp0003 = spots_by_ref['RSP0003:D000011']
    rsp0004 = spots_by_ref['RSP0004:D000014'] & rsp0003
    rsp0008 = spots_by_ref['RSP0008:D000029'] & rsp0003
    manual_idc2 = set(spot_context.loc[spot_context['manual_annotation_fine'].astype(str).eq('IDC_2'), 'spot_id'].astype(str))
    rsp0003_manual_idc2_overlap = rsp0003 & manual_idc2
    region_a = rsp0004 - rsp0008
    region_b = rsp0008 - rsp0004
    region_c = rsp0004 & rsp0008
    idc2_activity_layer_summary = pd.DataFrame([{'analysis_block': 'IDC_2 activity-layer landscape', 'broad_field': 'RSP0003:D000011', 'manual_idc2_spots': len(manual_idc2), 'manual_idc2_covered_by_broad_field': len(rsp0003_manual_idc2_overlap), 'manual_idc2_coverage_fraction': len(rsp0003_manual_idc2_overlap) / len(manual_idc2), 'sublayers': 'RSP0004:D000014;RSP0008:D000029', 'region_A': 'RSP0004-only within RSP0003', 'region_A_spots': len(region_a), 'region_B': 'RSP0008-only within RSP0003', 'region_B_spots': len(region_b), 'region_C': 'RSP0004 and RSP0008 overlap within RSP0003', 'region_C_spots': len(region_c), 'claim': 'RSP0003-backed IDC_2-like broad field contains partially overlapping RSP0004 and RSP0008 activity subregions.'}])
    idc2_region_a_vs_b_de_readback = run_region_de('IDC2_RSP0004_only_vs_RSP0008_only', 'region_A_RSP0004_only_within_RSP0003', region_a, 'region_B_RSP0008_only_within_RSP0003', region_b)
    idc2_region_a_vs_b_go_readback = run_go_enrichment(idc2_region_a_vs_b_de_readback, 'IDC2_RSP0004_only_vs_RSP0008_only')
    for (name, table) in {'idc2_activity_layer_summary.csv': idc2_activity_layer_summary, 'idc2_region_a_vs_b_de_readback.csv': idc2_region_a_vs_b_de_readback, 'idc2_region_a_vs_b_go_readback.csv': idc2_region_a_vs_b_go_readback}.items():
        write_table(table, TABLE_DIR / name)
    idc2_de_pass = idc2_region_a_vs_b_de_readback.query('passes_de_threshold').copy()
    idc2_loading_backbone = '; '.join((f'{rsp_id}=' + ','.join(contribution_genes(rsp_gene_loadings, rsp_id, 'in_contribution_50', 8)) for rsp_id in ['RSP0003', 'RSP0004', 'RSP0008']))
    idc2_a_genes = idc2_de_pass.loc[idc2_de_pass['higher_group'].eq('region_A_RSP0004_only_within_RSP0003')].sort_values('pvals_adj')['gene'].head(10).tolist()
    idc2_b_genes = idc2_de_pass.loc[idc2_de_pass['higher_group'].eq('region_B_RSP0008_only_within_RSP0003')].sort_values('pvals_adj')['gene'].head(10).tolist()
    idc2_notebook_summary = pd.DataFrame([{'readout': 'RSP loading backbone', 'value': idc2_loading_backbone, 'interpretation': 'RSP0003, RSP0004, and RSP0008 are interpreted first as activity signals defined by their loading genes.'}, {'readout': 'spatial field', 'value': f'RSP0003:D000011 contains {len(rsp0003)} spots and covers {len(rsp0003_manual_idc2_overlap)}/{len(manual_idc2)} manually annotated IDC_2 spots', 'interpretation': 'RSP0003 defines the broad field used as the background for internal activity-layer analysis.'}, {'readout': 'subregion split', 'value': f'A={len(region_a)} spots; B={len(region_b)} spots; overlap={len(region_c)} spots', 'interpretation': 'RSP0004-only and RSP0008-only define two simple, non-overlapping regions within the RSP0003 field.'}, {'readout': 'regional DEG support', 'value': f'{len(idc2_de_pass)} genes pass |logFC| >= {DEG_ABS_LOGFC_CUTOFF} and adjusted P < {DEG_PADJ_CUTOFF}', 'interpretation': 'DE is interpreted as expression difference between RSD-defined regions, not as an intrinsic identity label for either RSP.'}, {'readout': 'region A biased genes', 'value': ';'.join(idc2_a_genes), 'interpretation': 'Genes higher in the RSP0004-only portion of the RSP0003 field.'}, {'readout': 'region B biased genes', 'value': ';'.join(idc2_b_genes), 'interpretation': 'Genes higher in the RSP0008-only portion of the RSP0003 field.'}, {'readout': 'main conclusion', 'value': 'coexisting activity layers', 'interpretation': 'The RSP0003 field contains partially overlapping RSP-backed RSD subregions.'}])
    dcis_entity = spots_by_ref['RSP0005:D000015']
    idc_entity = spots_by_ref['RSP0003:D000011']
    dcis_lcis_vs_idc_activity_backbone = pd.DataFrame([context_card('RSP0005:D000015_DCIS_LCIS_associated_entity', dcis_entity, ['RSP0005']), context_card('RSP0003:D000011_IDC_associated_entity', idc_entity, ['RSP0003'])])
    for rsp_id in ['RSP0005', 'RSP0003']:
        row_mask = dcis_lcis_vs_idc_activity_backbone['region'].str.startswith(rsp_id)
        dcis_lcis_vs_idc_activity_backbone.loc[row_mask, 'core_genes_12'] = ';'.join(core_genes(rsp_core_gene_loadings, rsp_id, 12))
        dcis_lcis_vs_idc_activity_backbone.loc[row_mask, 'contribution_50_genes_12'] = ';'.join(contribution_genes(rsp_gene_loadings, rsp_id, 'in_contribution_50', 12))
    dcis_lcis_vs_idc_de_readback = run_region_de('RSP0005_DCIS_LCIS_vs_RSP0003_IDC', 'RSP0005_D000015_DCIS_LCIS_associated_region', dcis_entity, 'RSP0003_D000011_IDC_associated_region', idc_entity)
    dcis_lcis_vs_idc_go_readback = run_go_enrichment(dcis_lcis_vs_idc_de_readback, 'RSP0005_DCIS_LCIS_vs_RSP0003_IDC')
    deepst_reference_path = Path(selection['verification_references']['deepst_deg_table']['path'])
    deepst_reference = pd.read_csv(deepst_reference_path)
    if len(deepst_reference) != 298:
        raise AssertionError(f'Expected 298 published DeepST genes, found {len(deepst_reference)}')
    deepst_annotation = deepst_reference.loc[:, ['SYMBOL', 'logFC', 'type']].rename(columns={'SYMBOL': 'gene', 'logFC': 'deepst_logfc', 'type': 'deepst_direction'}).assign(deepst_reported=True).loc[:, ['gene', 'deepst_reported', 'deepst_logfc', 'deepst_direction']]
    missing_genes = deepst_annotation.loc[~deepst_annotation['gene'].isin(dcis_lcis_vs_idc_de_readback['gene']), 'gene'].tolist()
    if missing_genes:
        raise ValueError(f'DeepST genes missing from the REGAL contrast: {missing_genes}')
    dcis_lcis_vs_idc_de_readback = dcis_lcis_vs_idc_de_readback.merge(deepst_annotation, on='gene', how='left', validate='one_to_one')
    dcis_lcis_vs_idc_de_readback['deepst_reported'] = dcis_lcis_vs_idc_de_readback['deepst_reported'].fillna(False).astype(bool)
    deepst_mask = dcis_lcis_vs_idc_de_readback['deepst_reported']
    dcis_lcis_vs_idc_de_readback['deepst_direction_concordant'] = pd.Series(pd.NA, index=dcis_lcis_vs_idc_de_readback.index, dtype='boolean')
    dcis_lcis_vs_idc_de_readback.loc[deepst_mask, 'deepst_direction_concordant'] = (np.sign(dcis_lcis_vs_idc_de_readback.loc[deepst_mask, 'deepst_logfc']) == np.sign(dcis_lcis_vs_idc_de_readback.loc[deepst_mask, 'logfoldchanges'])).to_numpy()
    if int(deepst_mask.sum()) != 298:
        raise AssertionError(f'Expected 298 matched DeepST genes, found {int(deepst_mask.sum())}')
    if not dcis_lcis_vs_idc_de_readback.loc[deepst_mask, 'deepst_direction_concordant'].all():
        raise AssertionError('Not all published DeepST genes have concordant log-fold-change directions')
    write_table(dcis_lcis_vs_idc_activity_backbone, TABLE_DIR / 'dcis_lcis_vs_idc_activity_backbone.csv')
    write_table(dcis_lcis_vs_idc_de_readback, TABLE_DIR / 'dcis_lcis_vs_idc_de_readback.csv')
    write_table(dcis_lcis_vs_idc_go_readback, TABLE_DIR / 'dcis_lcis_vs_idc_go_readback.csv')
    dcis_idc_de_pass = dcis_lcis_vs_idc_de_readback.query('passes_de_threshold').copy()
    deepst_matched_de = dcis_lcis_vs_idc_de_readback.query('deepst_reported').copy()
    dcis_idc_loading_backbone = '; '.join((f'{rsp_id}=' + ','.join(contribution_genes(rsp_gene_loadings, rsp_id, 'in_contribution_50', 8)) for rsp_id in ['RSP0005', 'RSP0003']))
    dcis_genes = dcis_idc_de_pass.loc[dcis_idc_de_pass['higher_group'].eq('RSP0005_D000015_DCIS_LCIS_associated_region')].sort_values('pvals_adj')['gene'].head(10).tolist()
    idc_genes = dcis_idc_de_pass.loc[dcis_idc_de_pass['higher_group'].eq('RSP0003_D000011_IDC_associated_region')].sort_values('pvals_adj')['gene'].head(10).tolist()
    dcis_idc_notebook_summary = pd.DataFrame([{'readout': 'RSP loading backbone', 'value': dcis_idc_loading_backbone, 'interpretation': 'The DCIS/LCIS-associated and IDC-associated regions are compared through their source RSP activity backbones before DE validation.'}, {'readout': 'contrast definition', 'value': f'RSP0005:D000015={len(dcis_entity)} spots; RSP0003:D000011={len(idc_entity)} spots', 'interpretation': 'The comparison is defined by two selected RSD entities, not by annotation labels alone.'}, {'readout': 'regional DEG support', 'value': f'{len(dcis_idc_de_pass)} genes pass |logFC| >= {DEG_ABS_LOGFC_CUTOFF} and adjusted P < {DEG_PADJ_CUTOFF}', 'interpretation': 'The DE table reports expression differences between the two RSD-defined tumor regions.'}, {'readout': 'published DeepST directional concordance', 'value': f"{deepst_matched_de['deepst_direction_concordant'].sum()}/{len(deepst_matched_de)} matched genes have concordant log-fold-change signs", 'interpretation': 'Published DeepST genes provide post-selection directional verification of the fixed RSD contrast.'}, {'readout': 'RSP0005-region biased genes', 'value': ';'.join(dcis_genes), 'interpretation': 'Representative genes higher in the DCIS/LCIS-associated RSD-defined region.'}, {'readout': 'RSP0003-region biased genes', 'value': ';'.join(idc_genes), 'interpretation': 'Representative genes higher in the IDC-associated RSD-defined region.'}, {'readout': 'main conclusion', 'value': 'distinct tumor activity backbones', 'interpretation': 'DCIS/LCIS-associated and IDC-associated regions are supported by different RSP activity backbones.'}])
    rsp0018_entity = spots_by_ref['RSP0018:D000089']
    rsp0005_entity = spots_by_ref['RSP0005:D000015']
    overlay = rsp0018_entity & rsp0005_entity
    rsp0005_only = rsp0005_entity - rsp0018_entity
    rsp0018_only = rsp0018_entity - rsp0005_entity
    rsp0005_rsp0018_overlay_summary = pd.DataFrame([overlap_row('RSP0005:D000015', rsp0005_entity, 'RSP0018:D000089', rsp0018_entity)])
    rsp0005_rsp0018_overlay_summary['spatial_relation'] = 'immune-like overlay on DCIS/LCIS-associated tumor entity'
    spot_state = pd.Series('neither', index=activation.index.astype(str), name='state')
    spot_state.loc[spot_state.index.isin(rsp0005_only)] = 'RSP0005-only'
    spot_state.loc[spot_state.index.isin(rsp0018_only)] = 'RSP0018-only'
    spot_state.loc[spot_state.index.isin(overlay)] = 'RSP0005+RSP0018'
    (immune_expression, present_immune_markers) = load_normalized_expression(h5ad_path, IMMUNE_MARKER_GENES)
    missing_immune_markers = sorted(set(IMMUNE_MARKER_GENES) - set(present_immune_markers))
    if missing_immune_markers:
        raise ValueError(f'Immune markers missing from h5ad: {missing_immune_markers}')
    common_spots = activation.index.astype(str).intersection(immune_expression.index.astype(str))
    dcis_lcis_overlay_spot_readback = pd.DataFrame({'spot_id': common_spots, 'state': spot_state.loc[common_spots].to_numpy(), 'RSP0005 activation': activation.loc[common_spots, 'RSP0005'].to_numpy(float), 'RSP0018 activation': activation.loc[common_spots, 'RSP0018'].to_numpy(float), 'immune loading score': immune_expression.loc[common_spots, IMMUNE_MARKER_GENES].mean(axis=1).to_numpy(float), 'marker_set_id': IMMUNE_MARKER_SET_ID})
    dcis_lcis_overlay_spot_readback = dcis_lcis_overlay_spot_readback.loc[dcis_lcis_overlay_spot_readback['state'].isin(['RSP0005-only', 'RSP0005+RSP0018', 'RSP0018-only'])].reset_index(drop=True)
    overlap_scores = dcis_lcis_overlay_spot_readback.loc[dcis_lcis_overlay_spot_readback['state'].eq('RSP0005+RSP0018'), 'immune loading score'].to_numpy(float)
    rsp0005_only_scores = dcis_lcis_overlay_spot_readback.loc[dcis_lcis_overlay_spot_readback['state'].eq('RSP0005-only'), 'immune loading score'].to_numpy(float)
    immune_test = stats.mannwhitneyu(overlap_scores, rsp0005_only_scores, alternative='two-sided')
    immune_rank_biserial = 2 * float(immune_test.statistic) / (len(overlap_scores) * len(rsp0005_only_scores)) - 1
    dcis_lcis_overlay_statistics = pd.DataFrame([{'comparison': 'RSP0005+RSP0018 vs RSP0005-only', 'endpoint': 'immune loading score', 'marker_set_id': IMMUNE_MARKER_SET_ID, 'markers': ';'.join(IMMUNE_MARKER_GENES), 'test': 'two-sided Mann-Whitney U', 'analysis_unit': 'spot', 'n_overlap': len(overlap_scores), 'n_rsp0005_only': len(rsp0005_only_scores), 'median_overlap': float(np.median(overlap_scores)), 'median_rsp0005_only': float(np.median(rsp0005_only_scores)), 'median_difference': float(np.median(overlap_scores) - np.median(rsp0005_only_scores)), 'mannwhitney_u': float(immune_test.statistic), 'rank_biserial': immune_rank_biserial, 'p_value': float(immune_test.pvalue)}])
    write_table(rsp0005_rsp0018_overlay_summary, TABLE_DIR / 'rsp0005_rsp0018_overlay_summary.csv')
    write_table(dcis_lcis_overlay_spot_readback, TABLE_DIR / 'dcis_lcis_overlay_spot_readback.csv')
    write_table(dcis_lcis_overlay_statistics, TABLE_DIR / 'dcis_lcis_overlay_statistics.csv')
    overlay_stats = rsp0005_rsp0018_overlay_summary.iloc[0]
    overlay_loading_backbone = '; '.join((f'{rsp_id}=' + ','.join(contribution_genes(rsp_gene_loadings, rsp_id, 'in_contribution_50', 8)) for rsp_id in ['RSP0005', 'RSP0018']))
    immune_stats = dcis_lcis_overlay_statistics.iloc[0]
    overlay_notebook_summary = pd.DataFrame([{'readout': 'RSP loading backbone', 'value': overlay_loading_backbone, 'interpretation': 'RSP0018 is interpreted as an immune-like activity signal and RSP0005 as a tumor-associated signal through their loading genes.'}, {'readout': 'RSD overlap', 'value': f"{int(overlay_stats['overlap_n'])} shared spots; {overlay_stats['fraction_of_a']:.3f} of RSP0005 entity; {overlay_stats['fraction_of_b']:.3f} of RSP0018 entity", 'interpretation': 'The overlap is a local composite region rather than a mutually exclusive partition.'}, {'readout': 'three spatial groups', 'value': f'RSP0005-only={len(rsp0005_only)}; RSP0018-only={len(rsp0018_only)}; overlap={len(overlay)}', 'interpretation': 'The three-way split separates tumor-only, immune-like-only, and composite activity contexts.'}, {'readout': 'fixed-panel immune readback', 'value': f"median={immune_stats['median_overlap']:.3f} in overlap versus {immune_stats['median_rsp0005_only']:.3f} in RSP0005-only; rank-biserial={immune_stats['rank_biserial']:.3f}; P={immune_stats['p_value']:.3g}", 'interpretation': 'The overlap carries stronger expression of the predefined immune marker panel than the tumor-associated RSP0005-only region.'}, {'readout': 'main conclusion', 'value': 'activity-layer superposition', 'interpretation': 'RSP0018 locally overlays the RSP0005-backed DCIS/LCIS tumor entity.'}])
    rsp0009_refs = ['RSP0009:D000030', 'RSP0009:D000031', 'RSP0009:D000032', 'RSP0009:D000033']
    rsp0009_region_spots = {ref: spots_by_ref[ref] for ref in rsp0009_refs}
    rsp0009_entity_realized_contribution = realized_contribution_table('RSP0009', rsp0009_region_spots, contribution_flag='in_contribution_90')
    write_table(rsp0009_entity_realized_contribution, TABLE_DIR / 'rsp0009_entity_realized_contribution.csv')
    rsp0009_loading_backbone = ','.join(contribution_genes(rsp_gene_loadings, 'RSP0009', 'in_contribution_50', 12))
    rsp0009_spot_summary = '; '.join((f'{ref}={len(spots)} spots' for (ref, spots) in rsp0009_region_spots.items()))
    rsp0009_notebook_summary = pd.DataFrame([{'readout': 'RSP loading backbone', 'value': rsp0009_loading_backbone, 'interpretation': 'RSP0009 is interpreted through one shared loading-gene backbone before comparing how its RSD entities realize that backbone.'}, {'readout': 'discrete entities', 'value': rsp0009_spot_summary, 'interpretation': 'The same RSP0009 activity layer is instantiated as multiple RSD entities.'}, {'readout': 'realized contribution matrix', 'value': f"{rsp0009_entity_realized_contribution['gene'].nunique()} genes x {rsp0009_entity_realized_contribution['region'].nunique()} entities", 'interpretation': 'Gene usage is evaluated at the RSD entity level through loading weight and local expression.'}, {'readout': 'main conclusion', 'value': 'discrete instantiation with differential realized contribution', 'interpretation': 'RSP0009 supports multiple spatial entities whose gene usage is not identical.'}])
    integrated_landscape_summary = pd.DataFrame([{'analysis_block': 'IDC_2 activity-layer landscape', 'landscape_relation': 'coexisting activity layers', 'selected_rsp_objects': 'RSP0003;RSP0004;RSP0008', 'selected_rsd_objects': 'RSP0003:D000011;RSP0004:D000014;RSP0008:D000029', 'spatial_relation': 'RSP0004 and RSP0008 are embedded sublayers within RSP0003 broad field', 'gene_readback': 'loading-backbone comparison plus region_A vs region_B DE/GO validation', 'reference_support': 'annotation/RCTD/CopyKAT context card', 'main_claim': 'IDC_2-like broad field contains partially overlapping activity subregions.', 'panel_id': 'idc2_activity_layers', 'source_tables': 'idc2_activity_layer_summary.csv;idc2_region_a_vs_b_de_readback.csv;idc2_region_a_vs_b_go_readback.csv'}, {'analysis_block': 'DCIS/LCIS versus IDC activity backbone', 'landscape_relation': 'distinct tumor activity backbones', 'selected_rsp_objects': 'RSP0005;RSP0003', 'selected_rsd_objects': 'RSP0005:D000015;RSP0003:D000011', 'spatial_relation': 'two RSD-defined tumor structure regions', 'gene_readback': 'loading-backbone comparison plus RSD-defined regional DE/GO validation', 'reference_support': 'annotation/RCTD/CopyKAT context card plus published DeepST directional verification', 'main_claim': 'DCIS/LCIS-associated and IDC-associated entities are supported by distinct RSP backbones.', 'panel_id': 'dcis_lcis_vs_idc_backbone', 'source_tables': 'dcis_lcis_vs_idc_activity_backbone.csv;dcis_lcis_vs_idc_de_readback.csv;dcis_lcis_vs_idc_go_readback.csv'}, {'analysis_block': 'Immune-like overlay', 'landscape_relation': 'activity-layer superposition', 'selected_rsp_objects': 'RSP0005;RSP0018', 'selected_rsd_objects': 'RSP0005:D000015;RSP0018:D000089', 'spatial_relation': 'RSP0018 overlaps RSP0005-backed DCIS/LCIS entity', 'gene_readback': 'fixed-panel immune loading score in overlap and single-RSP spot groups', 'reference_support': 'annotation/RCTD/CopyKAT context card', 'main_claim': 'immune-like activity locally overlays a DCIS/LCIS-associated tumor entity.', 'panel_id': 'immune_like_overlay', 'source_tables': 'rsp0005_rsp0018_overlay_summary.csv;dcis_lcis_overlay_spot_readback.csv;dcis_lcis_overlay_statistics.csv'}, {'analysis_block': 'RSP0009 discrete entities', 'landscape_relation': 'discrete instantiation and realized contribution', 'selected_rsp_objects': 'RSP0009', 'selected_rsd_objects': ';'.join(rsp0009_refs), 'spatial_relation': 'one RSP forms multiple discrete DCIS/LCIS-associated entities', 'gene_readback': 'entity-level realized contribution of one shared loading-gene backbone', 'reference_support': 'annotation/RCTD/CopyKAT context card', 'main_claim': 'RSP0009 entities differentially realize the same source loading-gene backbone.', 'panel_id': 'rsp0009_discrete_entities;entity_realized_contribution', 'source_tables': 'rsp0009_entity_realized_contribution.csv'}])

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
    _CONFIG = document['fig05']
    prepare()
    missing = [name for name in TABLE_NAMES if not (TABLE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f'Missing prepared tables: {missing}')
    print(TABLE_DIR)

if __name__ == '__main__':
    main()
