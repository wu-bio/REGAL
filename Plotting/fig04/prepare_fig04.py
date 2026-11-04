"""Compute figure source tables from configured inputs and saved model results."""
from __future__ import annotations
import argparse
from string import Template
from copy import deepcopy
from plot_fig04 import LIVER_REFERENCE, MOB_REFERENCE
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_CONFIG: dict
_CONFIG_DIR: Path
_ROOTS: dict

from plot_fig04 import RESOURCE_PATHS

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig04'
TABLE_NAMES = ('liver_representative_rsps.csv', 'mob_representative_rsps.csv')

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
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from dataclasses import dataclass
from typing import Any
import yaml
from scipy import stats

def load_obs_coordinates(h5ad_path: str | Path, *, sample_id: str, spatial_key: str='spatial', obs_columns: list[str] | tuple[str, ...]=()) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        if spatial_key not in adata.obsm:
            raise ValueError(f'Missing obsm[{spatial_key!r}] in {h5ad_path}')
        spatial = np.asarray(adata.obsm[spatial_key])[:, :2]
        out = pd.DataFrame({'spot_id': adata.obs_names.astype(str), 'sample_id': sample_id, 'x': spatial[:, 0].astype(float), 'y': spatial[:, 1].astype(float)})
        for col in obs_columns:
            if col in adata.obs:
                out[col] = adata.obs[col].astype(str).to_numpy()
    finally:
        adata.file.close()
    return out

def load_selected_gene_expression(h5ad_path: str | Path, genes: list[str] | tuple[str, ...], *, sample_id: str, layer: str | None=None, normalize: bool=True) -> pd.DataFrame:
    requested = list(dict.fromkeys((str(g) for g in genes if str(g))))
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        present = [gene for gene in requested if gene in adata.var_names]
        if not present:
            out = pd.DataFrame({'spot_id': adata.obs_names.astype(str), 'sample_id': sample_id})
            return out
        sub = adata[:, present].to_memory()
        matrix = sub.layers[layer] if layer and layer in sub.layers else sub.X
        values = _dense(matrix).astype(float)
        if normalize:
            totals = _row_totals(adata)
            totals = np.maximum(totals.astype(float), 1.0)
            values = np.log1p(values / totals[:, None] * 10000.0)
        out = pd.DataFrame(values, columns=present)
        out.insert(0, 'sample_id', sample_id)
        out.insert(0, 'spot_id', sub.obs_names.astype(str))
    finally:
        adata.file.close()
    return out

def marker_availability(sample_id: str, marker_panel: pd.DataFrame, present_genes: list[str] | tuple[str, ...]) -> pd.DataFrame:
    present = set(map(str, present_genes))
    out = marker_panel.copy()
    out.insert(0, 'sample_id', sample_id)
    out['available'] = out['gene'].astype(str).isin(present)
    out['matched_gene'] = out['gene'].where(out['available'], '')
    return out

def build_group_scores(marker_expression: pd.DataFrame, marker_panel: pd.DataFrame, *, group_col: str, use_col: str='use_in_scoring') -> pd.DataFrame:
    expr = marker_expression.set_index('spot_id')
    sample_id = str(expr['sample_id'].iloc[0]) if 'sample_id' in expr.columns and (not expr.empty) else ''
    expr = expr.drop(columns=['sample_id'], errors='ignore')
    panel = marker_panel.loc[marker_panel[use_col].astype(bool)].copy() if use_col in marker_panel else marker_panel.copy()
    rows: list[pd.Series] = []
    for (group, sub) in panel.groupby(group_col, sort=False):
        genes = [gene for gene in sub['gene'].astype(str).drop_duplicates().tolist() if gene in expr.columns]
        if genes:
            score = expr.loc[:, genes].mean(axis=1)
            score.name = str(group)
            rows.append(score)
    if not rows:
        return pd.DataFrame({'spot_id': marker_expression['spot_id'], 'sample_id': sample_id})
    scores = pd.concat(rows, axis=1).reset_index()
    scores.insert(1, 'sample_id', sample_id)
    return scores

def _dense(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)

def _row_totals(adata: ad.AnnData) -> np.ndarray:
    if 'total_counts' in adata.obs:
        return pd.to_numeric(adata.obs['total_counts'], errors='coerce').fillna(0).to_numpy(dtype=float)
    matrix = adata.X
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return np.asarray(matrix).sum(axis=1).astype(float)
REQUIRED_RSP_FILES = {'rsp': 'rsp.parquet', 'activation': 'rsp_activation.parquet', 'gene_loadings': 'rsp_gene_loadings.parquet', 'core_gene_loadings': 'rsp_core_gene_loadings.parquet', 'qc': 'rsp_qc.parquet', 'family_members': 'rsp_family_members.parquet'}

@dataclass(frozen=True)
class SampleConfig:
    dataset: str
    sample_id: str
    platform: str
    h5ad_path: Path
    sample_dir: Path
    include: bool = True
    exclusion_reason: str = ''

def resolve_rsp_out(sample_dir: str | Path) -> Path:
    path = Path(sample_dir)
    if path.name == 'rsp_out':
        return path
    return path / 'rsp_out'

def load_rsp_tables(sample_dir: str | Path) -> dict[str, pd.DataFrame]:
    rsp_out = resolve_rsp_out(sample_dir)
    if not rsp_out.exists():
        raise FileNotFoundError(f'Missing rsp_out directory: {rsp_out}')
    tables: dict[str, pd.DataFrame] = {}
    for (key, filename) in REQUIRED_RSP_FILES.items():
        path = rsp_out / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing RSP table: {path}')
        tables[key] = pd.read_parquet(path)
    return tables

def activation_matrix(rsp_activation: pd.DataFrame) -> pd.DataFrame:
    required = {'spot_id', 'rsp_id', 'activation'}
    missing = required - set(rsp_activation.columns)
    if missing:
        raise ValueError(f'rsp_activation is missing columns: {sorted(missing)}')
    mat = rsp_activation.pivot_table(index='spot_id', columns='rsp_id', values='activation', aggfunc='first')
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    return mat.fillna(0.0).astype(float)

def top_genes_for_rsp(rsp_gene_loadings: pd.DataFrame, rsp_id: str, n: int=30) -> list[str]:
    sub = rsp_gene_loadings.loc[rsp_gene_loadings['rsp_id'].astype(str) == str(rsp_id)].copy()
    if sub.empty:
        return []
    if 'loading_rank' in sub.columns:
        sub = sub.sort_values('loading_rank', kind='mergesort')
    elif 'loading' in sub.columns:
        sub = sub.sort_values('loading', ascending=False, kind='mergesort')
    return sub['gene'].astype(str).head(int(n)).tolist()

def core_genes_for_rsp(rsp_core_gene_loadings: pd.DataFrame, rsp_id: str) -> list[str]:
    return top_genes_for_rsp(rsp_core_gene_loadings, rsp_id, n=1000)

def rsp_inventory(sample_id: str, platform: str, rsp: pd.DataFrame, gene_loadings: pd.DataFrame, core_gene_loadings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = rsp.sort_values('rsp_score', ascending=False).copy() if 'rsp_score' in rsp.columns else rsp.copy()
    for rec in ordered.to_dict(orient='records'):
        rid = str(rec['rsp_id'])
        sub = gene_loadings.loc[gene_loadings['rsp_id'].astype(str) == rid]
        rows.append({'sample_id': sample_id, 'platform': platform, 'rsp_id': rid, 'rsp_score': _float(rec.get('rsp_score')), 'quality_score': _float(rec.get('quality_score')), 'source_axis': str(rec.get('source_axis', '')), 'source_k': _int_or_blank(rec.get('source_k')), 'family_id': str(rec.get('family_id', '')), 'top_genes_10': ','.join(top_genes_for_rsp(gene_loadings, rid, 10)), 'top_genes_30': ','.join(top_genes_for_rsp(gene_loadings, rid, 30)), 'top_genes': ','.join(top_genes_for_rsp(gene_loadings, rid, 30)), 'core_genes': ','.join(core_genes_for_rsp(core_gene_loadings, rid)), 'n_genes_contribution_70': int(sub.get('in_contribution_70', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())})
    return pd.DataFrame(rows)

def check_inputs(sample_configs: list[SampleConfig] | list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in sample_configs:
        cfg = _coerce_config(raw)
        rsp_out = resolve_rsp_out(cfg.sample_dir)
        missing_rsp_files = [filename for filename in REQUIRED_RSP_FILES.values() if not (rsp_out / filename).exists()]
        rows.append({'dataset': cfg.dataset, 'sample_id': cfg.sample_id, 'platform': cfg.platform, 'include': bool(cfg.include), 'exclusion_reason': cfg.exclusion_reason, 'h5ad_path': str(cfg.h5ad_path), 'sample_dir': str(cfg.sample_dir), 'rsp_out': str(rsp_out), 'h5ad_exists': cfg.h5ad_path.exists(), 'rsp_out_exists': rsp_out.exists(), 'missing_rsp_files': ','.join(missing_rsp_files), 'rsp_required_files_present': not missing_rsp_files})
    return pd.DataFrame(rows)

def _coerce_config(raw: SampleConfig | dict[str, Any]) -> SampleConfig:
    if isinstance(raw, SampleConfig):
        return raw
    return SampleConfig(dataset=str(raw['dataset']), sample_id=str(raw['sample_id']), platform=str(raw.get('platform', '')), h5ad_path=Path(raw['h5ad_path']), sample_dir=Path(raw['sample_dir']), include=bool(raw.get('include', True)), exclusion_reason=str(raw.get('exclusion_reason', '')))

def _float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float('nan')
        return float(value)
    except Exception:
        return float('nan')

def _int_or_blank(value: Any) -> int | str:
    try:
        if pd.isna(value):
            return ''
        return int(value)
    except Exception:
        return ''

def prepare():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_liver():

        def safe_spearman(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
                return (np.nan, np.nan)
            (rho, pvalue) = stats.spearmanr(x[mask], y[mask])
            return (float(rho), float(pvalue))

        def top_mask(values, fraction=0.1):
            values = pd.Series(values).astype(float)
            n_top = max(1, int(np.ceil(values.shape[0] * fraction)))
            threshold = values.sort_values(ascending=False).iloc[n_top - 1]
            return values >= threshold

        def log2_ratio(a, b, eps=1e-09):
            if not np.isfinite(a) or not np.isfinite(b):
                return np.nan
            return float(np.log2((a + eps) / (b + eps)))

        def mean_or_nan(values):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else np.nan
        liver_cfg = dict(deepcopy(LIVER_REFERENCE), axis={'label_column': option('liver_label_column'), 'proxy_scores': option('liver_proxy_scores')})
        configs = [SampleConfig('liver', option('liver_sample_id'), 'MERFISH', path('liver_h5ad'), path('liver_sample_dir'))]
        liver_input_integrity = check_inputs(configs)
        rsp_tables_by_sample = {}
        inventory_tables = []
        for cfg in configs:
            tables = load_rsp_tables(cfg.sample_dir)
            rsp_tables_by_sample[cfg.sample_id] = tables
            inventory_tables.append(rsp_inventory(cfg.sample_id, cfg.platform, tables['rsp'], tables['gene_loadings'], tables['core_gene_loadings']))
        liver_rsp_inventory = pd.concat(inventory_tables, ignore_index=True)
        liver_rsp_inventory[['sample_id', 'rsp_id', 'rsp_score', 'quality_score', 'top_genes_10', 'core_genes']].head(12)
        proxy_scores = {str(k): float(v) for (k, v) in liver_cfg['axis']['proxy_scores'].items()}
        label_col = liver_cfg['axis']['label_column']

        def derived_zonation_label(label):
            text = str(label)
            if text in proxy_scores:
                return text
            if text.startswith('Hep_') or 'Hep' in text:
                return 'other_hepatocyte'
            return 'non_hepatocyte'

        def build_zonation_axis(obs):
            out = obs[['spot_id', label_col]].copy()
            out['derived_zonation_label'] = out[label_col].map(derived_zonation_label)
            out['zonation_axis'] = out['derived_zonation_label'].map(proxy_scores)
            out['is_zonation_proxy'] = out['zonation_axis'].notna()
            return out
        axis_tables = []
        for cfg in configs:
            obs = load_obs_coordinates(cfg.h5ad_path, sample_id=cfg.sample_id, obs_columns=[label_col])
            axis = build_zonation_axis(obs)
            axis.insert(0, 'sample_id', cfg.sample_id)
            axis_tables.append(axis)
        liver_zonation_axis_values = pd.concat(axis_tables, ignore_index=True)
        liver_zonation_axis_values.groupby(['sample_id', 'derived_zonation_label']).size().reset_index(name='n_spots')
        portal_markers = list(liver_cfg['markers']['portal'])
        central_markers = list(liver_cfg['markers']['central'])
        zonation_markers = portal_markers + central_markers
        marker_rows = []
        for cfg in configs:
            expr = load_selected_gene_expression(cfg.h5ad_path, zonation_markers, sample_id=cfg.sample_id, layer='lognorm', normalize=False)
            present = set(expr.drop(columns=['spot_id', 'sample_id'], errors='ignore').columns.astype(str))
            for (direction, genes) in [('portal', portal_markers), ('central', central_markers)]:
                for gene in genes:
                    marker_rows.append({'sample_id': cfg.sample_id, 'gene': gene, 'marker_direction': direction, 'available': gene in present})
        liver_marker_availability = pd.DataFrame(marker_rows)
        alignment_rows = []
        for cfg in configs:
            tables = rsp_tables_by_sample[cfg.sample_id]
            mat = activation_matrix(tables['activation'])
            axis = liver_zonation_axis_values.loc[liver_zonation_axis_values['sample_id'] == cfg.sample_id].set_index('spot_id')
            inv = liver_rsp_inventory.loc[liver_rsp_inventory['sample_id'] == cfg.sample_id].set_index('rsp_id')
            common = mat.index.intersection(axis.index.astype(str))
            proxy = axis.loc[common, 'zonation_axis'].astype(float)
            valid = proxy.notna()
            for rsp_id in mat.columns.astype(str):
                values = mat.loc[common, rsp_id].astype(float)
                (rho, pvalue) = safe_spearman(values.loc[valid], proxy.loc[valid])
                top = top_mask(values.loc[valid], 0.1)
                alignment_rows.append({'sample_id': cfg.sample_id, 'platform': cfg.platform, 'rsp_id': rsp_id, 'spearman_rho_zonation': rho, 'spearman_pvalue_zonation': pvalue, 'mean_zonation_axis_top_activation': float(proxy.loc[valid].loc[top].mean()), 'n_proxy_spots': int(valid.sum()), 'top_genes': inv.loc[rsp_id, 'top_genes_30'] if rsp_id in inv.index else '', 'core_genes': inv.loc[rsp_id, 'core_genes'] if rsp_id in inv.index else ''})
        liver_rsp_zonation_alignment = pd.DataFrame(alignment_rows)
        write_table(liver_rsp_zonation_alignment, TABLE_DIR / 'liver_rsp_zonation_alignment.csv')
        liver_rsp_zonation_alignment.sort_values(['sample_id', 'spearman_rho_zonation'], ascending=[True, False]).head(12)

        def flatten_representative_selections(selection_rows):
            rows = []
            for block in selection_rows:
                if 'representatives' not in block:
                    rows.append(block)
                    continue
                sample_id = block['sample_id']
                for rec in block.get('representatives', []):
                    row = dict(rec)
                    row['sample_id'] = sample_id
                    rows.append(row)
            return rows

        def selected_liver_representatives(selection_rows, alignment, claim_boundary):
            rows = []
            for sel in flatten_representative_selections(selection_rows):
                sample_id = str(sel['sample_id'])
                rsp_id = str(sel['rsp_id'])
                sub = alignment.loc[(alignment['sample_id'].astype(str) == sample_id) & (alignment['rsp_id'].astype(str) == rsp_id)]
                if sub.empty:
                    rows.append({'sample_id': sample_id, 'platform': '', 'rsp_id': rsp_id, 'spearman_rho_zonation': np.nan, 'spearman_pvalue_zonation': np.nan, 'mean_zonation_axis_top_activation': np.nan, 'n_proxy_spots': np.nan, 'top_genes': '', 'core_genes': '', 'selection_reason': sel.get('selection_note', 'manual representative selection'), 'evidence_regime': 'continuous zonation axis', 'evidence_context': sel.get('evidence_context', ''), 'claim_boundary': claim_boundary})
                    continue
                row = sub.iloc[0].to_dict()
                row['selection_reason'] = sel.get('selection_note', 'manual representative selection')
                row['evidence_regime'] = 'continuous zonation axis'
                row['evidence_context'] = sel.get('evidence_context', 'central' if float(row['spearman_rho_zonation']) > 0 else 'portal')
                row['claim_boundary'] = claim_boundary
                rows.append(row)
            return pd.DataFrame(rows)
        liver_representative_rsps = selected_liver_representatives(liver_cfg.get('selected_representatives', []), liver_rsp_zonation_alignment, liver_cfg['claim_boundary'])
        write_table(liver_representative_rsps, TABLE_DIR / 'liver_representative_rsps.csv')
    prepare_liver()

    def prepare_mob():

        def safe_spearman(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
                return (np.nan, np.nan)
            (rho, pvalue) = stats.spearmanr(x[mask], y[mask])
            return (float(rho), float(pvalue))

        def top_mask(values, fraction=0.1):
            values = pd.Series(values).astype(float)
            n_top = max(1, int(np.ceil(values.shape[0] * fraction)))
            threshold = values.sort_values(ascending=False).iloc[n_top - 1]
            return values >= threshold

        def log2_ratio(a, b, eps=1e-09):
            if not np.isfinite(a) or not np.isfinite(b):
                return np.nan
            return float(np.log2((a + eps) / (b + eps)))

        def mean_or_nan(values):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else np.nan
        mob_cfg = dict(deepcopy(MOB_REFERENCE), evidence_thresholds=option('mob_evidence_thresholds'))
        thresholds = mob_cfg['evidence_thresholds']
        configs = [SampleConfig('mob', 'Puck_200127_15', 'Slide-seqV2', path('puck_h5ad'), path('puck_sample_dir')), SampleConfig('mob', 'Stereo_seq', 'Stereo-seq', path('stereo_h5ad'), path('stereo_sample_dir'))]
        mob_input_integrity = check_inputs(configs)
        marker_rows = []
        marker_fields = [('canonical_primary', 'primary', True, 'canonical primary marker'), ('canonical_secondary', 'secondary', False, 'canonical secondary/context marker'), ('st_contrast', 'st_contrast', False, 'high-contrast ST marker from prior method-study usage'), ('primary', 'primary', True, 'primary marker'), ('secondary', 'secondary', False, 'secondary/context marker')]
        for (layer, spec) in mob_cfg['layers'].items():
            for (field, evidence_tier, use_in_scoring, note) in marker_fields:
                for gene in spec.get(field, []):
                    marker_rows.append({'gene': gene, 'layer': layer, 'marker_role': field, 'evidence_tier': evidence_tier, 'use_in_scoring': use_in_scoring, 'notes': f'{layer} {note}'})
        mob_marker_panel = pd.DataFrame(marker_rows).drop_duplicates(['gene', 'layer', 'marker_role']).reset_index(drop=True)
        mob_marker_panel.groupby(['layer', 'evidence_tier']).size().reset_index(name='n_markers')
        included_configs = [cfg for cfg in configs if cfg.include]
        rsp_tables_by_sample = {}
        marker_expr_by_sample = {}
        layer_scores_by_sample = {}
        inventory_tables = []
        availability_tables = []
        for cfg in included_configs:
            tables = load_rsp_tables(cfg.sample_dir)
            rsp_tables_by_sample[cfg.sample_id] = tables
            inventory_tables.append(rsp_inventory(cfg.sample_id, cfg.platform, tables['rsp'], tables['gene_loadings'], tables['core_gene_loadings']))
            expr = load_selected_gene_expression(cfg.h5ad_path, mob_marker_panel['gene'].tolist(), sample_id=cfg.sample_id)
            marker_expr_by_sample[cfg.sample_id] = expr
            layer_scores_by_sample[cfg.sample_id] = build_group_scores(expr, mob_marker_panel, group_col='layer')
            availability_tables.append(marker_availability(cfg.sample_id, mob_marker_panel, expr.columns.tolist()))
        mob_rsp_inventory = pd.concat(inventory_tables, ignore_index=True)
        mob_marker_availability = pd.concat(availability_tables, ignore_index=True)
        mob_rsp_inventory[['sample_id', 'rsp_id', 'rsp_score', 'top_genes_10', 'core_genes']].head(12)

        def marker_overlap(sample_id, platform, rsp, gene_loadings, marker_panel, group_col, top_ns=(10, 30)):
            rows = []
            for rsp_id in rsp['rsp_id'].astype(str):
                for top_n in top_ns:
                    genes = top_genes_for_rsp(gene_loadings, rsp_id, top_n)
                    ranked = pd.DataFrame({'gene': genes, 'gene_rank': range(1, len(genes) + 1)})
                    merged = ranked.merge(marker_panel, on='gene', how='inner')
                    for rec in merged.to_dict(orient='records'):
                        rows.append({'sample_id': sample_id, 'platform': platform, 'rsp_id': rsp_id, 'top_n': top_n, 'gene': rec['gene'], 'gene_rank': rec['gene_rank'], group_col: rec[group_col], 'evidence_tier': rec.get('evidence_tier', ''), 'use_in_scoring': bool(rec.get('use_in_scoring', True))})
            return pd.DataFrame(rows)

        def marker_spearman(sample_id, platform, activation, marker_expression, marker_panel, group_col):
            mat = activation_matrix(activation)
            expr = marker_expression.set_index('spot_id').drop(columns=['sample_id'], errors='ignore')
            common = mat.index.intersection(expr.index.astype(str))
            mat = mat.loc[common]
            expr = expr.loc[common]
            marker_meta = marker_panel.loc[marker_panel['gene'].astype(str).isin(expr.columns.astype(str))]
            rows = []
            for rsp_id in mat.columns.astype(str):
                y = mat[rsp_id].to_numpy(dtype=float)
                for rec in marker_meta.to_dict(orient='records'):
                    gene = rec['gene']
                    (rho, pvalue) = safe_spearman(y, expr[gene].to_numpy(dtype=float))
                    rows.append({'sample_id': sample_id, 'platform': platform, 'rsp_id': rsp_id, 'gene': gene, group_col: rec[group_col], 'evidence_tier': rec.get('evidence_tier', ''), 'use_in_scoring': bool(rec.get('use_in_scoring', True)), 'rho': rho, 'pvalue': pvalue, 'n_spots': int(len(common))})
            return pd.DataFrame(rows)

        def group_spearman(sample_id, platform, activation, group_scores, group_col):
            mat = activation_matrix(activation)
            scores = group_scores.set_index('spot_id').drop(columns=['sample_id'], errors='ignore')
            common = mat.index.intersection(scores.index.astype(str))
            mat = mat.loc[common]
            scores = scores.loc[common]
            rows = []
            for rsp_id in mat.columns.astype(str):
                y = mat[rsp_id].to_numpy(dtype=float)
                for group in scores.columns.astype(str):
                    (rho, pvalue) = safe_spearman(y, scores[group].to_numpy(dtype=float))
                    rows.append({'sample_id': sample_id, 'platform': platform, 'rsp_id': rsp_id, group_col: group, 'rho': rho, 'pvalue': pvalue, 'n_spots': int(len(common))})
            return pd.DataFrame(rows)

        def top_spot_enrichment(sample_id, platform, activation, marker_expression, marker_panel, group_col, top_fraction=0.1):
            mat = activation_matrix(activation)
            expr = marker_expression.set_index('spot_id').drop(columns=['sample_id'], errors='ignore')
            common = mat.index.intersection(expr.index.astype(str))
            mat = mat.loc[common]
            expr = expr.loc[common]
            marker_meta = marker_panel.loc[marker_panel['gene'].astype(str).isin(expr.columns.astype(str))]
            rows = []
            for rsp_id in mat.columns.astype(str):
                top = top_mask(mat[rsp_id], top_fraction)
                for rec in marker_meta.to_dict(orient='records'):
                    gene = rec['gene']
                    vals = expr[gene].astype(float)
                    mean_top = mean_or_nan(vals.loc[top].to_numpy())
                    mean_bg = mean_or_nan(vals.loc[~top].to_numpy())
                    rows.append({'sample_id': sample_id, 'platform': platform, 'rsp_id': rsp_id, 'gene': gene, group_col: rec[group_col], 'evidence_tier': rec.get('evidence_tier', ''), 'use_in_scoring': bool(rec.get('use_in_scoring', True)), 'top_fraction': top_fraction, 'mean_top': mean_top, 'mean_background': mean_bg, 'log2fc_top_vs_background': log2_ratio(mean_top, mean_bg), 'n_spots': int(len(common))})
            return pd.DataFrame(rows)

        def build_candidate_matrix(sample_id, platform, rsp, overlap, group_corr, topspot, group_col, thresholds):
            groups = sorted(set(overlap.get(group_col, pd.Series(dtype=str)).dropna().astype(str)) | set(topspot.get(group_col, pd.Series(dtype=str)).dropna().astype(str)) | set(group_corr.get(group_col, pd.Series(dtype=str)).dropna().astype(str)))
            rows = []
            for rsp_id in rsp['rsp_id'].astype(str):
                for group in groups:
                    ov = overlap.loc[(overlap['rsp_id'].astype(str) == rsp_id) & (overlap[group_col].astype(str) == group) & overlap['use_in_scoring']] if not overlap.empty else pd.DataFrame()
                    gc = group_corr.loc[(group_corr['rsp_id'].astype(str) == rsp_id) & (group_corr[group_col].astype(str) == group)] if not group_corr.empty else pd.DataFrame()
                    te = topspot.loc[(topspot['rsp_id'].astype(str) == rsp_id) & (topspot[group_col].astype(str) == group) & topspot['use_in_scoring']] if not topspot.empty else pd.DataFrame()
                    overlap_genes = sorted(set(ov.loc[ov['top_n'] == 30, 'gene'].astype(str))) if not ov.empty else []
                    primary_genes = sorted(set(ov.loc[(ov['top_n'] == 30) & (ov['evidence_tier'].astype(str) == 'primary'), 'gene'].astype(str))) if not ov.empty else []
                    best_rho = float(gc['rho'].max()) if not gc.empty else np.nan
                    best_fc = float(te['log2fc_top_vs_background'].max()) if not te.empty else np.nan
                    votes = int(bool(primary_genes)) + int(np.isfinite(best_rho) and best_rho >= thresholds['spearman']) + int(np.isfinite(best_fc) and best_fc >= thresholds['log2fc'])
                    if overlap_genes or np.isfinite(best_rho) or np.isfinite(best_fc):
                        rows.append({'sample_id': sample_id, 'platform': platform, 'rsp_id': rsp_id, group_col: group, 'evidence_votes': votes, 'primary_marker_overlap': ','.join(primary_genes), 'top_gene_marker_overlap': ','.join(overlap_genes), 'group_spearman': best_rho, 'marker_topspot_log2fc': best_fc, 'passes_candidate_screen': votes >= 2})
            return pd.DataFrame(rows)

        def add_inventory_fields(selected, inventory):
            rows = []
            for rec in selected.to_dict(orient='records'):
                inv = inventory.loc[(inventory['sample_id'].astype(str) == str(rec['sample_id'])) & (inventory['rsp_id'].astype(str) == str(rec['rsp_id']))]
                out = dict(rec)
                out['top_genes'] = inv['top_genes_30'].iloc[0] if not inv.empty else ''
                out['core_genes'] = inv['core_genes'].iloc[0] if not inv.empty else ''
                rows.append(out)
            return pd.DataFrame(rows)
        overlap_tables = []
        spearman_tables = []
        layer_spearman_tables = []
        topspot_tables = []
        candidate_tables = []
        for cfg in included_configs:
            tables = rsp_tables_by_sample[cfg.sample_id]
            expr = marker_expr_by_sample[cfg.sample_id]
            layer_scores = layer_scores_by_sample[cfg.sample_id]
            overlap = marker_overlap(cfg.sample_id, cfg.platform, tables['rsp'], tables['gene_loadings'], mob_marker_panel, 'layer')
            spearman = marker_spearman(cfg.sample_id, cfg.platform, tables['activation'], expr, mob_marker_panel, 'layer')
            layer_corr = group_spearman(cfg.sample_id, cfg.platform, tables['activation'], layer_scores, 'layer')
            topspot = top_spot_enrichment(cfg.sample_id, cfg.platform, tables['activation'], expr, mob_marker_panel, 'layer')
            candidates = build_candidate_matrix(cfg.sample_id, cfg.platform, tables['rsp'], overlap, layer_corr, topspot, 'layer', thresholds)
            overlap_tables.append(overlap)
            spearman_tables.append(spearman)
            layer_spearman_tables.append(layer_corr)
            topspot_tables.append(topspot)
            candidate_tables.append(candidates)
        mob_rsp_marker_overlap = pd.concat(overlap_tables, ignore_index=True)
        mob_rsp_marker_spearman = pd.concat(spearman_tables, ignore_index=True)
        mob_rsp_layer_score_spearman = pd.concat(layer_spearman_tables, ignore_index=True)
        mob_rsp_topspot_enrichment = pd.concat(topspot_tables, ignore_index=True)
        mob_rsp_marker_candidate_matrix = pd.concat(candidate_tables, ignore_index=True)
        write_table(mob_rsp_marker_candidate_matrix, TABLE_DIR / 'mob_rsp_marker_candidate_matrix.csv')
        mob_rsp_marker_candidate_matrix.groupby(['sample_id', 'passes_candidate_screen']).size().reset_index(name='n_rsp_layer_candidates')

        def split_genes(value):
            return [x.strip() for x in str(value).split(',') if x.strip()]

        def flatten_representative_selections(selection_rows):
            rows = []
            for block in selection_rows:
                if 'representatives' not in block:
                    rows.append(block)
                    continue
                sample_id = block['sample_id']
                for rec in block.get('representatives', []):
                    row = dict(rec)
                    row['sample_id'] = sample_id
                    rows.append(row)
            return rows

        def manual_representatives(selection_rows, candidates, inventory, group_col, default_boundary):
            rows = []
            for sel in flatten_representative_selections(selection_rows):
                sample_id = str(sel['sample_id'])
                rsp_id = str(sel['rsp_id'])
                context = str(sel['evidence_context'])
                sub = candidates.loc[(candidates['sample_id'].astype(str) == sample_id) & (candidates['rsp_id'].astype(str) == rsp_id) & (candidates[group_col].astype(str) == context)]
                if sub.empty:
                    row = {'sample_id': sample_id, 'platform': '', 'rsp_id': rsp_id, 'evidence_context': context, 'evidence_votes': np.nan, 'primary_marker_overlap': '', 'top_gene_marker_overlap': '', 'group_spearman': np.nan, 'marker_topspot_log2fc': np.nan, 'passes_candidate_screen': False}
                else:
                    row = sub.sort_values(['evidence_votes', 'group_spearman', 'marker_topspot_log2fc'], ascending=[False, False, False]).iloc[0].to_dict()
                    row['evidence_context'] = row.pop(group_col)
                row['interpretation_context'] = sel.get('interpretation_context', context)
                row['selection_reason'] = sel.get('selection_note', 'manual representative selection')
                row['claim_boundary'] = default_boundary
                rows.append(row)
            selected = pd.DataFrame(rows)
            return add_inventory_fields(selected, inventory)
        mob_representative_rsps = manual_representatives(mob_cfg.get('selected_representatives', []), mob_rsp_marker_candidate_matrix, mob_rsp_inventory, 'layer', mob_cfg['claim_boundary'])
        write_table(mob_representative_rsps, TABLE_DIR / 'mob_representative_rsps.csv')
    prepare_mob()

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
    _CONFIG = document['fig04']
    prepare()
    missing = [name for name in TABLE_NAMES if not (TABLE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f'Missing prepared tables: {missing}')
    print(TABLE_DIR)

if __name__ == '__main__':
    main()
