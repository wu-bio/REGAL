"""Compute figure source tables from configured inputs and saved model results."""
from __future__ import annotations
import argparse
from string import Template
import warnings
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_CONFIG: dict
_CONFIG_DIR: Path
_ROOTS: dict

from plot_fig08 import RESOURCE_PATHS

TABLE_DIR = Path(__file__).resolve().parents[1] / 'output' / 'tables' / 'fig08'
TABLE_NAMES = ('interface_lr_readback.csv', 'regional_celltype_readback.csv', 'regional_interface_spots.csv', 'regional_pathology_readback.csv', 'regional_program_enrichment.csv', 'rsp_hypothesis_retrieval.csv')

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

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from collections.abc import Mapping
from collections.abc import Iterable, Mapping
import yaml
from scipy.stats import fisher_exact
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr

@contextmanager
def open_backed_h5ad(path: str | Path) -> Iterator[ad.AnnData]:
    """Open a read-only backed AnnData object and always close its file handle."""
    adata = ad.read_h5ad(path, backed='r')
    try:
        yield adata
    finally:
        adata.file.close()

def read_gene_names(path: str | Path) -> list[str]:
    """Read h5ad variable names without loading the expression matrix."""
    with open_backed_h5ad(path) as adata:
        return adata.var_names.astype(str).tolist()

def log_normalize_counts(matrix: object, totals: np.ndarray, *, target_sum: float=10000.0, return_dense: bool=False) -> object:
    """Library-size normalize a selected count matrix and apply log1p."""
    totals = np.asarray(totals, dtype=float).copy()
    totals[~np.isfinite(totals) | (totals <= 0)] = 1.0
    if sparse.issparse(matrix):
        normalized = sparse.diags(float(target_sum) / totals) @ matrix.tocsr().astype(float)
        normalized.data = np.log1p(normalized.data)
        return normalized.toarray() if return_dense else normalized
    normalized = np.log1p(np.asarray(matrix, dtype=float) * (float(target_sum) / totals[:, None]))
    return normalized

def bh_adjust(values: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjustment preserving missing values and row order."""
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    finite = np.isfinite(raw)
    if finite.any():
        result[finite] = multipletests(raw[finite], method='fdr_bh')[1]
    return result

def assemble_r5_output_tables(*, sc_state_genes: pd.DataFrame, translation_scores: pd.DataFrame, selected_objects: pd.DataFrame, selected_gene_sets: pd.DataFrame, selected_enrichment: pd.DataFrame, source_caf_profile: pd.DataFrame, source_caf_contrasts: pd.DataFrame, source_caf_hotspots: pd.DataFrame, rsp_celltypes: pd.DataFrame, carrier_pathology: pd.DataFrame, regional_pathology: pd.DataFrame, pathology_contrasts: pd.DataFrame, celltype_profile: pd.DataFrame, celltype_contrasts: pd.DataFrame, regional_context: pd.DataFrame, author_lr: pd.DataFrame, regal_lr: pd.DataFrame, leave_lr: pd.DataFrame, concordance: pd.DataFrame, leave_manifest: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Package R5 results into a compact set of figure-facing readback tables."""
    retrieval = translation_scores.merge(selected_objects.rename(columns={'object_role': 'selected_object_role'}), on='rsp_id', how='left', validate='many_to_one')
    retrieval['selected_for_r5'] = retrieval['selected_object_role'].notna()
    source_caf = _source_caf_readback(source_caf_profile, source_caf_contrasts, source_caf_hotspots)
    pathology = _regional_pathology_readback(regional_pathology, pathology_contrasts, carrier_pathology)
    celltypes = _regional_celltype_readback(celltype_profile, celltype_contrasts, rsp_celltypes)
    interface_lr = _interface_lr_readback(author_lr, regal_lr, leave_lr, concordance, leave_manifest, regional_context)
    return {'sc_state_query_genes.csv': sc_state_genes.copy(), 'rsp_hypothesis_retrieval.csv': retrieval, 'regional_program_genes.csv': selected_gene_sets.copy(), 'regional_program_enrichment.csv': selected_enrichment.copy(), 'source_caf_spatial_readback.csv': source_caf, 'regional_pathology_readback.csv': pathology, 'regional_celltype_readback.csv': celltypes, 'regional_interface_spots.csv': regional_context.reset_index(), 'interface_lr_readback.csv': interface_lr}

def _source_caf_readback(profile: pd.DataFrame, contrasts: pd.DataFrame, hotspots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ('iCAF', 'myCAF'):
        state_profile = profile[['Classification', 'n_spots', f'{state}_mean', f'{state}_median']].rename(columns={'Classification': 'pathology', f'{state}_mean': 'mean_abundance', f'{state}_median': 'median_abundance'})
        state_profile.insert(0, 'caf_state', state)
        rows.append(state_profile)
    result = pd.concat(rows, ignore_index=True)
    result = result.merge(hotspots, on=['caf_state', 'pathology'], how='left')
    result[['n_hotspots', 'fraction_within_hotspots', 'fraction_of_pathology_class']] = result[['n_hotspots', 'fraction_within_hotspots', 'fraction_of_pathology_class']].fillna(0)
    return result.merge(contrasts, on='caf_state', how='left', validate='many_to_one')

def _regional_pathology_readback(profile: pd.DataFrame, contrasts: pd.DataFrame, carrier_pathology: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()
    result['context_n_spots'] = result.groupby('regional_context')['n_spots'].transform('sum')
    result = result.merge(contrasts.rename(columns={'pathology': 'Classification'}), on='Classification', how='left', validate='many_to_one', suffixes=('', '_overlap_contrast'))
    carrier = carrier_pathology.pivot(index='Classification', columns='carrier_scope', values=['n_spots', 'fraction_within_scope'])
    carrier.columns = [f'carrier_{scope}_{metric}' for (metric, scope) in carrier.columns]
    return result.merge(carrier.reset_index(), on='Classification', how='left')

def _regional_celltype_readback(profile: pd.DataFrame, contrasts: pd.DataFrame, carrier_readback: pd.DataFrame) -> pd.DataFrame:
    result = profile.merge(contrasts, on='celltype', how='left', validate='many_to_one', suffixes=('', '_overlap_contrast'))
    carrier = carrier_readback.add_prefix('carrier_').rename(columns={'carrier_celltype': 'celltype'})
    return result.merge(carrier, on='celltype', how='left', validate='many_to_one')

def _interface_lr_readback(author_lr: pd.DataFrame, regal_lr: pd.DataFrame, leave_lr: pd.DataFrame, concordance: pd.DataFrame, leave_manifest: pd.DataFrame, regional_context: pd.DataFrame) -> pd.DataFrame:
    keys = ['candidate_axis', 'ligand', 'receptor']
    result = _prefix_metrics(author_lr, 'author', keys).merge(_prefix_metrics(regal_lr, 'regal', keys), on=keys, validate='one_to_one')
    result = result.merge(_prefix_metrics(leave_lr, 'leaveout', keys), on=keys, validate='one_to_one')
    result['leaveout_direction_retained'] = result['regal_fold_change'].gt(1) == result['leaveout_fold_change'].gt(1)
    for (column, value) in concordance.iloc[0].items():
        result[f'interface_{column}'] = value
    carrier_rsp = str(regional_context['carrier_rsp'].iloc[0])
    immune_rsp = str(regional_context['immune_rsp'].iloc[0])
    for (role, rsp_id) in (('carrier', carrier_rsp), ('immune', immune_rsp)):
        row = leave_manifest.loc[leave_manifest['rsp_id'].astype(str).eq(rsp_id)].iloc[0]
        result[f'{role}_rsp'] = rsp_id
        result[f'{role}_genes_after_lr_exclusion'] = row['n_program_genes_after_exclusion']
        result[f'{role}_excluded_candidate_genes'] = row['excluded_candidate_genes_present']
    return result

def _prefix_metrics(table: pd.DataFrame, prefix: str, keys: list[str]) -> pd.DataFrame:
    rename = {column: f'{prefix}_{column}' for column in table.columns if column not in keys}
    return table.rename(columns=rename)
MIN_CELLS_PER_PATIENT = 20
MIN_MEAN_EFFECT = 0.1
MIN_POSITIVE_PATIENT_FRACTION = 0.6

def load_hypothesis(path: str | Path) -> dict:
    with Path(path).open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)

def hypothesis_evidence_roles(spec: dict) -> pd.DataFrame:
    """Create a compact provenance table for the biological inputs used in R5."""
    rows = []
    for (query, definition) in spec['state_queries'].items():
        rows.append({'evidence': query, 'source': 'Wu et al. single-cell atlas', 'analysis_role': definition['role'], 'spatial_use': 'query all pre-learned RSPs'})
    rows.extend([{'evidence': 'pathology_annotation', 'source': 'CID44971 author annotation', 'analysis_role': 'regional tissue context', 'spatial_use': 'describe selected RSP environments'}, {'evidence': 'stereoscope_abundance', 'source': 'Wu TNBC single-cell reference and CID44971 Visium', 'analysis_role': 'cellular readback', 'spatial_use': 'characterize selected regional contexts'}, {'evidence': 'candidate_CAF_T_axes', 'source': spec['candidate_ligand_receptor_pairs']['source'], 'analysis_role': 'candidate molecular readback', 'spatial_use': 'test the continuous RSP interface'}])
    return pd.DataFrame(rows)

def gene_availability_by_modality(spec: dict, sc_genes: Iterable[str], st_genes: Iterable[str]) -> pd.DataFrame:
    sc_available = set(map(str, sc_genes))
    st_available = set(map(str, st_genes))
    rows = []
    for pair in spec['candidate_ligand_receptor_pairs']['pairs']:
        pair_name = f"{pair['ligand']}–{pair['receptor']}"
        for role in ('ligand', 'receptor'):
            gene = str(pair[role])
            rows.append({'candidate_axis': pair_name, 'gene_role': role, 'gene': gene, 'present_in_sc': gene in sc_available, 'present_in_st': gene in st_available})
    return pd.DataFrame(rows)

def build_single_cell_state_queries(sc_h5ad: str | Path, spec: dict, gene_universe: Iterable[str], *, excluded_patient: str, min_cells: int=MIN_CELLS_PER_PATIENT, min_mean_effect: float=MIN_MEAN_EFFECT, min_positive_fraction: float=MIN_POSITIVE_PATIENT_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Define patient-consistent state genes without using the spatial sample."""
    requested_genes = list(dict.fromkeys(map(str, gene_universe)))
    with open_backed_h5ad(sc_h5ad) as adata:
        required_obs = {'Patient'}
        for definition in spec['state_queries'].values():
            required_obs.add(str(definition['target']['column']))
            required_obs.add(str(definition['reference']['column']))
        missing = required_obs.difference(adata.obs.columns)
        if missing:
            raise ValueError(f'Single-cell reference is missing columns: {sorted(missing)}')
        keep = adata.obs['Patient'].astype(str).ne(excluded_patient).to_numpy()
        obs = adata.obs.loc[keep].copy()
        available = {str(gene): index for (index, gene) in enumerate(adata.var_names.astype(str))}
        present_genes = [gene for gene in requested_genes if gene in available]
        positions = sorted(((available[gene], gene) for gene in present_genes))
        ordered_positions = [position for (position, _) in positions]
        ordered_genes = [gene for (_, gene) in positions]
        sub = adata[keep, ordered_positions].to_memory()
        matrix = sub.X.tocsr().astype(np.float64) if sparse.issparse(sub.X) else np.asarray(sub.X, dtype=float)
        totals = pd.to_numeric(obs.get('nCount_RNA'), errors='coerce').to_numpy(float)
        if not np.isfinite(totals).all() or (totals <= 0).any():
            full = adata[keep].X
            totals = np.asarray(full.sum(axis=1)).ravel().astype(float) if sparse.issparse(full) else np.asarray(full).sum(axis=1).astype(float)
    matrix = log_normalize_counts(matrix, totals)
    patients = obs['Patient'].astype(str)
    contrast_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (query, definition) in spec['state_queries'].items():
        target = _annotation_mask(obs, definition['target'])
        reference = _annotation_mask(obs, definition['reference'])
        effects = []
        patient_records = []
        for patient in sorted(patients.unique()):
            patient_mask = patients.eq(patient).to_numpy()
            target_mask = target & patient_mask
            reference_mask = reference & patient_mask
            if target_mask.sum() < min_cells or reference_mask.sum() < min_cells:
                continue
            effects.append(_row_mean(matrix, target_mask) - _row_mean(matrix, reference_mask))
            patient_records.append({'patient': patient, 'n_target': int(target_mask.sum()), 'n_reference': int(reference_mask.sum())})
        if not effects:
            raise ValueError(f'No eligible patient-level contrasts for query {query!r}')
        effect_matrix = np.vstack(effects)
        mean_effect = effect_matrix.mean(axis=0)
        positive_fraction = (effect_matrix > 0).mean(axis=0)
        selected = (mean_effect >= min_mean_effect) & (positive_fraction >= min_positive_fraction)
        for (gene, effect, consistency, use_gene) in zip(ordered_genes, mean_effect, positive_fraction, selected):
            contrast_rows.append({'query': query, 'gene': gene, 'n_patients': len(patient_records), 'mean_patient_effect': float(effect), 'positive_patient_fraction': float(consistency), 'selected': bool(use_gene)})
        summary_rows.append({'query': query, 'role': definition['role'], 'excluded_patient': excluded_patient, 'n_patients': len(patient_records), 'n_selected_genes': int(selected.sum()), 'min_cells_per_group': min_cells, 'min_mean_effect': min_mean_effect, 'min_positive_patient_fraction': min_positive_fraction, 'patient_counts': ';'.join((f"{row['patient']}:{row['n_target']}/{row['n_reference']}" for row in patient_records))})
    contrasts = pd.DataFrame(contrast_rows)
    selected_genes = contrasts.loc[contrasts['selected']].copy()
    selected_genes = selected_genes.merge(pd.DataFrame(summary_rows)[['query', 'role']], on='query', how='left')
    return (contrasts, selected_genes)

def query_rsp_enrichment(selected_genes: pd.DataFrame, loadings: pd.DataFrame) -> pd.DataFrame:
    """Test each state query against every REGAL core/contribution-70 union."""
    load = loadings.copy()
    load['rsp_id'] = load['rsp_id'].astype(str)
    load['gene'] = load['gene'].astype(str)
    universe = set(load['gene'])
    programs = {rsp_id: set(group.loc[group['is_core_gene'] | group['in_contribution_70'], 'gene']) for (rsp_id, group) in load.groupby('rsp_id')}
    rows = []
    for (query, group) in selected_genes.groupby('query', sort=False):
        signature = set(group['gene'].astype(str)) & universe
        for (rsp_id, program) in programs.items():
            overlap = signature & program
            a = len(overlap)
            b = len(signature - program)
            c = len(program - signature)
            d = len(universe - signature - program)
            (odds_ratio, p_value) = fisher_exact([[a, b], [c, d]], alternative='greater')
            rows.append({'query': query, 'rsp_id': rsp_id, 'n_signature_genes': len(signature), 'n_program_genes': len(program), 'n_overlap': a, 'odds_ratio': float(odds_ratio), 'p_value': float(p_value), 'overlap_genes': ';'.join(sorted(overlap))})
    result = pd.DataFrame(rows)
    result['fdr_bh'] = result.groupby('query', group_keys=False)['p_value'].transform(bh_adjust)
    result['enrichment_rank'] = result.groupby('query')['p_value'].rank(method='min').astype(int)
    return result.sort_values(['query', 'fdr_bh', 'p_value', 'rsp_id']).reset_index(drop=True)

def select_rsp_objects(enrichment: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-specified iCAF-carrier and T-cell-context selection rules."""
    caf = enrichment.loc[enrichment['query'].isin(['broad_caf', 'icaf', 'mycaf'])].pivot(index='rsp_id', columns='query', values=['odds_ratio', 'p_value', 'fdr_bh'])
    caf.columns = [f'{metric}__{query}' for (metric, query) in caf.columns]
    caf['icaf_to_mycaf_odds_ratio'] = caf['odds_ratio__icaf'] / caf['odds_ratio__mycaf'].replace(0, np.nan)
    eligible = caf.loc[caf['fdr_bh__broad_caf'].le(0.05) & caf['fdr_bh__icaf'].le(0.05) & caf['fdr_bh__mycaf'].le(0.05) & caf['icaf_to_mycaf_odds_ratio'].gt(1)].copy()
    if eligible.empty:
        raise RuntimeError('No RSP satisfied the pre-specified iCAF carrier rule')
    eligible = eligible.sort_values(['icaf_to_mycaf_odds_ratio', 'fdr_bh__broad_caf', 'fdr_bh__icaf'], ascending=[False, True, True])
    carrier_rsp = str(eligible.index[0])
    immune = enrichment.loc[enrichment['query'].eq('t_cell')].sort_values(['fdr_bh', 'p_value', 'odds_ratio'], ascending=[True, True, False])
    if immune.empty:
        raise RuntimeError('No T-cell RSP candidate was available')
    immune_rsp = str(immune.iloc[0]['rsp_id'])
    return pd.DataFrame([{'object_role': 'iCAF_associated_carrier', 'rsp_id': carrier_rsp, 'selection_rule': 'broad CAF, iCAF and myCAF FDR <= 0.05; iCAF/myCAF odds ratio > 1; maximize relative iCAF tilt', 'broad_caf_fdr': eligible.loc[carrier_rsp, 'fdr_bh__broad_caf'], 'icaf_fdr': eligible.loc[carrier_rsp, 'fdr_bh__icaf'], 'mycaf_fdr': eligible.loc[carrier_rsp, 'fdr_bh__mycaf'], 'icaf_to_mycaf_odds_ratio': eligible.loc[carrier_rsp, 'icaf_to_mycaf_odds_ratio']}, {'object_role': 'T_cell_regional_context', 'rsp_id': immune_rsp, 'selection_rule': 'lowest T-cell FDR, then p value and odds ratio', 't_cell_fdr': float(immune.iloc[0]['fdr_bh']), 't_cell_odds_ratio': float(immune.iloc[0]['odds_ratio'])}])

def selected_rsp_programs(loadings: pd.DataFrame, selected_rsp_ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(map(str, selected_rsp_ids))
    load = loadings.loc[loadings['rsp_id'].astype(str).isin(selected_ids)].copy()
    load['rsp_id'] = load['rsp_id'].astype(str)
    load['gene'] = load['gene'].astype(str)
    gene_sets = load.loc[load['is_core_gene'] | load['in_contribution_70']].copy()
    gene_sets['gene_set'] = 'core_or_contribution70'
    keep_columns = ['rsp_id', 'gene_set', 'gene', 'loading', 'loading_rank', 'is_core_gene', 'in_contribution_70', 'cumulative_fraction']
    loading_profile = load.sort_values(['rsp_id', 'loading_rank']).reset_index(drop=True)
    return (gene_sets[keep_columns].sort_values(['rsp_id', 'loading_rank']), loading_profile)

def pathway_enrichment(gene_sets: pd.DataFrame, loadings: pd.DataFrame, libraries: Mapping[str, str | Path]) -> pd.DataFrame:
    universe = set(loadings['gene'].astype(str))
    rows = []
    for (rsp_id, group) in gene_sets.groupby('rsp_id', sort=False):
        program = set(group['gene'].astype(str))
        for (library, path) in libraries.items():
            for (term, raw_genes) in _read_gmt(path).items():
                genes = raw_genes & universe
                if not 5 <= len(genes) <= 1000:
                    continue
                overlap = program & genes
                if not overlap:
                    continue
                a = len(overlap)
                b = len(program - genes)
                c = len(genes - program)
                d = len(universe - program - genes)
                (odds_ratio, p_value) = fisher_exact([[a, b], [c, d]], alternative='greater')
                rows.append({'rsp_id': rsp_id, 'library': library, 'term': term, 'n_program_genes': len(program), 'n_term_genes': len(genes), 'n_overlap': a, 'odds_ratio': float(odds_ratio), 'p_value': float(p_value), 'overlap_genes': ';'.join(sorted(overlap))})
    result = pd.DataFrame(rows)
    result['fdr_bh'] = result.groupby(['rsp_id', 'library'], group_keys=False)['p_value'].transform(bh_adjust)
    return result.sort_values(['rsp_id', 'library', 'fdr_bh', 'p_value']).reset_index(drop=True)

def _annotation_mask(obs: pd.DataFrame, definition: Mapping[str, object]) -> np.ndarray:
    column = str(definition['column'])
    labels = set(map(str, definition['labels']))
    return obs[column].astype(str).isin(labels).to_numpy()

def _row_mean(matrix: object, mask: np.ndarray) -> np.ndarray:
    selected = matrix[mask]
    if sparse.issparse(selected):
        return np.asarray(selected.mean(axis=0)).ravel()
    return np.asarray(selected).mean(axis=0)

def _read_gmt(path: str | Path) -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 3:
                terms[fields[0]] = set(fields[2:])
    return terms
ELIGIBLE_PATHOLOGY = {'Invasive cancer + lymphocytes', 'DCIS', 'Stroma', 'Lymphocytes'}
SOURCE_CAF_PATHOLOGY = {'Invasive cancer + lymphocytes', 'DCIS', 'Stroma', 'Stroma + adipose tissue', 'Lymphocytes', 'Normal + stroma + lymphocytes'}
ICAF_STROMAL_LYMPHOID_PATHOLOGY = {'Stroma', 'Stroma + adipose tissue', 'Lymphocytes', 'Normal + stroma + lymphocytes'}
INVASIVE_PATHOLOGY = 'Invasive cancer + lymphocytes'

def load_spatial_inputs(st_h5ad: str | Path, artifact_dir: str | Path, stereoscope_path: str | Path, *, eligible_pathology: Iterable[str]=ELIGIBLE_PATHOLOGY) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align pathology, RSP activity and Stereoscope abundance by spot."""
    with open_backed_h5ad(st_h5ad) as adata:
        if 'Classification' not in adata.obs:
            raise ValueError("CID44971 h5ad is missing obs['Classification']")
        pathology = adata.obs[['Classification']].copy()
        pathology.index = pathology.index.astype(str)
        if 'spatial' in adata.obsm:
            coordinates = np.asarray(adata.obsm['spatial'])[:, :2]
            pathology['spatial_x'] = coordinates[:, 0]
            pathology['spatial_y'] = coordinates[:, 1]
    artifact_dir = Path(artifact_dir)
    activation_long = pd.read_parquet(artifact_dir / 'rsp_out/rsp_activation.parquet')
    activation_long['spot_id'] = activation_long['spot_id'].astype(str)
    activation_long['rsp_id'] = activation_long['rsp_id'].astype(str)
    activation = activation_long.pivot(index='spot_id', columns='rsp_id', values='activation')
    active = activation_long.pivot(index='spot_id', columns='rsp_id', values='activation_active').astype(bool)
    stereoscope = pd.read_parquet(stereoscope_path)
    stereoscope.index = stereoscope.index.astype(str)
    common = pathology.index.intersection(active.index).intersection(stereoscope.index)
    classes = set(map(str, eligible_pathology))
    common = common[pathology.loc[common, 'Classification'].astype(str).isin(classes)]
    if not common.is_unique:
        raise ValueError('Spatial spot identifiers are not unique')
    return (pathology.loc[common].copy(), active.loc[common].copy(), activation.loc[common].copy(), stereoscope.loc[common].copy())

def source_caf_pathology_tables(pathology: pd.DataFrame, stereoscope: pd.DataFrame, *, icaf_column: str='CAFs MSC iCAF-like', mycaf_column: str='CAFs myCAF-like', hotspot_fraction: float=0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize the source CAF spatial partition without using it to select RSPs."""
    aligned = pathology[['Classification']].join(stereoscope[[icaf_column, mycaf_column]], how='inner')
    aligned['Classification'] = aligned['Classification'].astype(str)
    aligned = aligned.rename(columns={icaf_column: 'iCAF', mycaf_column: 'myCAF'})
    profile = aligned.groupby('Classification', observed=True).agg(n_spots=('iCAF', 'size'), iCAF_mean=('iCAF', 'mean'), iCAF_median=('iCAF', 'median'), myCAF_mean=('myCAF', 'mean'), myCAF_median=('myCAF', 'median')).reset_index()
    profile['iCAF_to_myCAF_mean_ratio'] = profile['iCAF_mean'] / profile['myCAF_mean']
    invasive = aligned['Classification'].astype(str).eq(INVASIVE_PATHOLOGY)
    stromal_lymphoid = aligned['Classification'].astype(str).isin(ICAF_STROMAL_LYMPHOID_PATHOLOGY)
    contrast_specs = [('myCAF', 'invasive > other annotated tissue', invasive, ~invasive), ('iCAF', 'stromal/lymphoid > invasive', stromal_lymphoid, invasive)]
    contrast_rows = []
    for (state, comparison, target_mask, reference_mask) in contrast_specs:
        target = aligned.loc[target_mask, state].to_numpy(float)
        reference = aligned.loc[reference_mask, state].to_numpy(float)
        test = mannwhitneyu(target, reference, alternative='greater')
        contrast_rows.append({'caf_state': state, 'comparison': comparison, 'n_target': len(target), 'n_reference': len(reference), 'target_mean': float(target.mean()), 'reference_mean': float(reference.mean()), 'fold_change': _fold_change(target, reference), 'mannwhitney_u': float(test.statistic), 'p_value': float(test.pvalue)})
    contrasts = pd.DataFrame(contrast_rows)
    contrasts['fdr_bh'] = bh_adjust(contrasts['p_value'])
    class_totals = aligned['Classification'].value_counts()
    hotspot_rows = []
    for state in ('iCAF', 'myCAF'):
        hotspot = _top_fraction(aligned[state], hotspot_fraction)
        counts = aligned.loc[hotspot, 'Classification'].value_counts()
        for (label, count) in counts.items():
            hotspot_rows.append({'caf_state': state, 'pathology': str(label), 'n_hotspots': int(count), 'fraction_within_hotspots': float(count / hotspot.sum()), 'fraction_of_pathology_class': float(count / class_totals.loc[label])})
    hotspots = pd.DataFrame(hotspot_rows)
    return (profile, contrasts, hotspots)

def build_regional_context(pathology: pd.DataFrame, active: pd.DataFrame, activation: pd.DataFrame, *, carrier_rsp: str, immune_rsp: str) -> pd.DataFrame:
    carrier_active = active[carrier_rsp].astype(bool)
    immune_active = active[immune_rsp].astype(bool)
    labels = np.select([carrier_active & immune_active, carrier_active & ~immune_active, ~carrier_active & immune_active], ['RSP_overlap', 'iCAF_RSP_only', 'immune_RSP_only'], default='neither')
    context = pathology.copy()
    context.index.name = 'spot_id'
    context['regional_context'] = labels
    context['carrier_rsp'] = carrier_rsp
    context['immune_rsp'] = immune_rsp
    context['carrier_active'] = carrier_active
    context['immune_active'] = immune_active
    context['carrier_activation'] = activation.loc[context.index, carrier_rsp]
    context['immune_activation'] = activation.loc[context.index, immune_rsp]
    context['regal_interface_score'] = context['carrier_activation'] * context['immune_activation']
    return context

def rsp_celltype_readback(context: pd.DataFrame, stereoscope: pd.DataFrame, *, icaf_column: str='CAFs MSC iCAF-like', mycaf_column: str='CAFs myCAF-like') -> pd.DataFrame:
    """Relate the selected carrier activation to its expected CAF abundances."""
    rows = []
    for celltype in (icaf_column, mycaf_column):
        x = context['carrier_activation'].to_numpy(float)
        y = stereoscope.loc[context.index, celltype].to_numpy(float)
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        active_mask = context['regional_context'].isin(['iCAF_RSP_only', 'RSP_overlap'])
        active_values = y[active_mask.to_numpy()]
        inactive_values = y[~active_mask.to_numpy()]
        test = mannwhitneyu(active_values, inactive_values, alternative='greater')
        rows.append({'celltype': celltype, 'pearson_r': float(pearson.statistic), 'pearson_p': float(pearson.pvalue), 'spearman_rho': float(spearman.statistic), 'spearman_p': float(spearman.pvalue), 'active_mean': float(active_values.mean()), 'inactive_mean': float(inactive_values.mean()), 'active_inactive_fold_change': _fold_change(active_values, inactive_values), 'active_greater_p': float(test.pvalue)})
    result = pd.DataFrame(rows)
    result['active_greater_fdr_bh'] = bh_adjust(result['active_greater_p'])
    return result

def pathology_context_tables(context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = context.groupby(['regional_context', 'Classification'], observed=True).size().rename('n_spots').reset_index()
    totals = summary.groupby('regional_context')['n_spots'].transform('sum')
    summary['fraction_within_context'] = summary['n_spots'] / totals
    target = context['regional_context'].eq('RSP_overlap')
    reference = context['regional_context'].eq('iCAF_RSP_only')
    rows = []
    for label in sorted(context['Classification'].astype(str).unique()):
        is_label = context['Classification'].astype(str).eq(label)
        a = int((target & is_label).sum())
        b = int((target & ~is_label).sum())
        c = int((reference & is_label).sum())
        d = int((reference & ~is_label).sum())
        (odds_ratio, p_value) = fisher_exact([[a, b], [c, d]], alternative='two-sided')
        rows.append({'pathology': label, 'overlap_n': a, 'overlap_fraction': a / max(a + b, 1), 'icaf_rsp_only_n': c, 'icaf_rsp_only_fraction': c / max(c + d, 1), 'odds_ratio': float(odds_ratio), 'p_value': float(p_value)})
    contrasts = pd.DataFrame(rows)
    contrasts['fdr_bh'] = bh_adjust(contrasts['p_value'])
    return (summary, contrasts.sort_values(['fdr_bh', 'p_value']).reset_index(drop=True))

def celltype_context_tables(context: pd.DataFrame, stereoscope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = stereoscope.loc[context.index].copy()
    aligned['regional_context'] = context['regional_context']
    long = aligned.melt(id_vars='regional_context', var_name='celltype', value_name='abundance')
    summary = long.groupby(['regional_context', 'celltype'], observed=True)['abundance'].agg(n_spots='size', mean='mean', median='median', q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
    rows = []
    target_mask = context['regional_context'].eq('RSP_overlap')
    reference_mask = context['regional_context'].eq('iCAF_RSP_only')
    for celltype in stereoscope.columns:
        target = stereoscope.loc[context.index[target_mask], celltype].to_numpy(float)
        reference = stereoscope.loc[context.index[reference_mask], celltype].to_numpy(float)
        test = mannwhitneyu(target, reference, alternative='greater')
        rows.append({'celltype': celltype, 'comparison': 'RSP_overlap > iCAF_RSP_only', 'n_overlap': len(target), 'n_icaf_rsp_only': len(reference), 'overlap_mean': float(target.mean()), 'icaf_rsp_only_mean': float(reference.mean()), 'fold_change': _fold_change(target, reference), 'mannwhitney_u': float(test.statistic), 'p_value': float(test.pvalue), 'endpoint_role': 'primary' if celltype in {'T cells CD4+', 'T cells CD8+'} else 'environment_profile'})
    contrasts = pd.DataFrame(rows)
    contrasts['fdr_bh'] = bh_adjust(contrasts['p_value'])
    return (summary, contrasts.sort_values(['fdr_bh', 'p_value']).reset_index(drop=True))

def add_author_interface_score(context: pd.DataFrame, stereoscope: pd.DataFrame, *, icaf_column: str='CAFs MSC iCAF-like', cd4_column: str='T cells CD4+', cd8_column: str='T cells CD8+') -> pd.DataFrame:
    result = context.copy()
    result['author_abundance_score'] = stereoscope.loc[result.index, icaf_column] * (stereoscope.loc[result.index, cd4_column] + stereoscope.loc[result.index, cd8_column])
    result['author_top10'] = _top_fraction(result['author_abundance_score'], 0.1)
    result['regal_top10'] = _top_fraction(result['regal_interface_score'], 0.1)
    return result

def load_log_normalized_expression(st_h5ad: str | Path, spots: Iterable[str], genes: Iterable[str]) -> pd.DataFrame:
    requested_spots = pd.Index(map(str, spots))
    requested_genes = list(dict.fromkeys(map(str, genes)))
    with open_backed_h5ad(st_h5ad) as adata:
        spot_positions = pd.Index(adata.obs_names.astype(str)).get_indexer(requested_spots)
        if (spot_positions < 0).any():
            raise ValueError('Some requested Visium spots are absent from the h5ad')
        gene_index = pd.Index(adata.var_names.astype(str))
        gene_positions = gene_index.get_indexer(requested_genes)
        present_pairs = sorted(((position, gene) for (position, gene) in zip(gene_positions, requested_genes) if position >= 0))
        positions = [position for (position, _) in present_pairs]
        ordered_genes = [gene for (_, gene) in present_pairs]
        sub = adata[spot_positions, positions].to_memory()
        matrix = sub.X
        totals = pd.to_numeric(adata.obs.iloc[spot_positions]['nCount_RNA'], errors='coerce').to_numpy(float)
    matrix = log_normalize_counts(matrix, totals, return_dense=True)
    return pd.DataFrame(matrix, index=requested_spots, columns=ordered_genes).reindex(columns=[gene for gene in requested_genes if gene in ordered_genes])

def candidate_lr_readback(expression: pd.DataFrame, high_spots: pd.Series, pairs: Iterable[Mapping[str, str]], *, analysis: str) -> pd.DataFrame:
    high = high_spots.reindex(expression.index).fillna(False).astype(bool)
    rows = []
    for pair in pairs:
        ligand = str(pair['ligand'])
        receptor = str(pair['receptor'])
        pair_name = f'{ligand}–{receptor}'
        if ligand not in expression or receptor not in expression:
            rows.append({'analysis': analysis, 'candidate_axis': pair_name, 'ligand': ligand, 'receptor': receptor, 'available': False})
            continue
        score = expression[ligand] * expression[receptor]
        selected = score.loc[high]
        remainder = score.loc[~high]
        test = mannwhitneyu(selected, remainder, alternative='greater')
        rows.append({'analysis': analysis, 'candidate_axis': pair_name, 'ligand': ligand, 'receptor': receptor, 'available': True, 'n_selected': len(selected), 'n_remainder': len(remainder), 'selected_mean_score': float(selected.mean()), 'remainder_mean_score': float(remainder.mean()), 'fold_change': _fold_change(selected.to_numpy(), remainder.to_numpy()), 'selected_detected_fraction': float((selected > 0).mean()), 'remainder_detected_fraction': float((remainder > 0).mean()), 'mannwhitney_u': float(test.statistic), 'p_value': float(test.pvalue)})
    result = pd.DataFrame(rows)
    result['fdr_bh'] = np.nan
    available = result['available'].fillna(False)
    result.loc[available, 'fdr_bh'] = bh_adjust(result.loc[available, 'p_value'])
    return result

def interface_concordance(context: pd.DataFrame) -> pd.DataFrame:
    author = context['author_top10'].astype(bool)
    regal = context['regal_top10'].astype(bool)
    intersection = int((author & regal).sum())
    union = int((author | regal).sum())
    pearson = pearsonr(context['author_abundance_score'], context['regal_interface_score'])
    spearman = spearmanr(context['author_abundance_score'], context['regal_interface_score'])
    return pd.DataFrame([{'n_spots': len(context), 'n_author_top10': int(author.sum()), 'n_regal_top10': int(regal.sum()), 'n_intersection': intersection, 'jaccard': intersection / union if union else np.nan, 'pearson_r': float(pearson.statistic), 'pearson_p': float(pearson.pvalue), 'spearman_rho': float(spearman.statistic), 'spearman_p': float(spearman.pvalue)}])

def leave_lr_out_interface(expression: pd.DataFrame, loadings: pd.DataFrame, *, carrier_rsp: str, immune_rsp: str, excluded_genes: Iterable[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Rebuild an interface after removing every tested ligand and receptor."""
    excluded = set(map(str, excluded_genes))
    score_table = pd.DataFrame(index=expression.index)
    manifest_rows = []
    for rsp_id in (carrier_rsp, immune_rsp):
        full_program = loadings.loc[loadings['rsp_id'].astype(str).eq(rsp_id) & (loadings['is_core_gene'] | loadings['in_contribution_70'])].copy()
        removed = excluded & set(full_program['gene'].astype(str))
        program = full_program.loc[~full_program['gene'].astype(str).isin(excluded)].copy()
        genes = [gene for gene in program['gene'].astype(str) if gene in expression]
        program = program.set_index(program['gene'].astype(str)).loc[genes]
        weights = program['loading'].astype(float).clip(lower=0).to_numpy()
        values = expression[genes].to_numpy(float)
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scales[scales == 0] = 1.0
        z_values = (values - means) / scales
        if weights.sum() <= 0:
            weights = np.ones(len(genes), dtype=float)
        weighted = np.average(z_values, axis=1, weights=weights)
        score_table[rsp_id] = pd.Series(weighted, index=expression.index).rank(pct=True)
        manifest_rows.append({'rsp_id': rsp_id, 'n_program_genes_after_exclusion': len(genes), 'excluded_candidate_genes_present': ';'.join(sorted(removed)), 'score_definition': 'loading-weighted mean of spot-wise standardized log expression, converted to percentile rank'})
    interface = score_table[carrier_rsp] * score_table[immune_rsp]
    interface.name = 'leave_lr_out_interface_score'
    return (interface, pd.DataFrame(manifest_rows))

def _top_fraction(values: pd.Series, fraction: float) -> pd.Series:
    n_select = max(1, int(np.ceil(len(values) * fraction)))
    selected_index = values.sort_values(ascending=False, kind='mergesort').head(n_select).index
    result = pd.Series(False, index=values.index)
    result.loc[selected_index] = True
    return result

def _fold_change(target: np.ndarray, reference: np.ndarray) -> float:
    target_mean = float(np.mean(target)) if len(target) else np.nan
    reference_mean = float(np.mean(reference)) if len(reference) else np.nan
    return target_mean / reference_mean if np.isfinite(reference_mean) and reference_mean > 0 else np.nan

def prepare():
    SC_H5AD = path('sc_h5ad')
    ST_H5AD = path('h5ad')
    ARTIFACT_DIR = path('sample_dir')
    STEREOSCOPE = path('stereoscope')
    HYPOTHESIS_YAML = resource('hypothesis_anchors.yaml')
    EXCLUDED_PATIENT = option('excluded_patient')
    PATHWAY_LIBRARIES = {name: resource(filename) for (name, filename) in [('Reactome', 'c2.cp.reactome.v2026.1.Hs.symbols.gmt'), ('BioPlanet', 'BioPlanet_2019.Enrichr.gmt')]}
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore', category=FutureWarning)
    spec = load_hypothesis(HYPOTHESIS_YAML)
    rsp_table = pd.read_parquet(ARTIFACT_DIR / 'rsp_out' / 'rsp.parquet')
    rsp_loadings = pd.read_parquet(ARTIFACT_DIR / 'rsp_out' / 'rsp_gene_loadings.parquet')
    sc_genes = read_gene_names(SC_H5AD)
    st_genes = read_gene_names(ST_H5AD)
    evidence_roles = hypothesis_evidence_roles(spec)
    gene_availability = gene_availability_by_modality(spec, sc_genes, st_genes)
    assert len(rsp_table) == rsp_loadings['rsp_id'].nunique()
    assert gene_availability['present_in_st'].all()
    (sc_state_contrasts, sc_state_genes) = build_single_cell_state_queries(SC_H5AD, spec, rsp_loadings['gene'].astype(str).unique(), excluded_patient=EXCLUDED_PATIENT)
    translation_scores = query_rsp_enrichment(sc_state_genes, rsp_loadings)
    selected_objects = select_rsp_objects(translation_scores)
    carrier_rsp = selected_objects.loc[selected_objects['object_role'].eq('iCAF_associated_carrier'), 'rsp_id'].iloc[0]
    immune_rsp = selected_objects.loc[selected_objects['object_role'].eq('T_cell_regional_context'), 'rsp_id'].iloc[0]
    selected_ids = [carrier_rsp, immune_rsp]
    translation_specificity = translation_scores.loc[translation_scores['rsp_id'].isin(selected_ids)].copy()
    query_summary = sc_state_genes.groupby(['query', 'role'], as_index=False).agg(n_genes=('gene', 'nunique'), n_patients=('n_patients', 'max'))
    (selected_gene_sets, selected_loading_profiles) = selected_rsp_programs(rsp_loadings, selected_ids)
    selected_enrichment = pathway_enrichment(selected_gene_sets, rsp_loadings, PATHWAY_LIBRARIES)
    program_sizes = selected_gene_sets.groupby('rsp_id')['gene'].nunique().rename('n_program_genes')
    significant_enrichment = selected_enrichment.loc[selected_enrichment['fdr_bh'].lt(0.05)]
    (source_pathology, _, _, source_stereoscope) = load_spatial_inputs(ST_H5AD, ARTIFACT_DIR, STEREOSCOPE, eligible_pathology=SOURCE_CAF_PATHOLOGY)
    (source_caf_profile, source_caf_contrasts, source_caf_hotspots) = source_caf_pathology_tables(source_pathology, source_stereoscope)
    (pathology, active, activation, stereoscope) = load_spatial_inputs(ST_H5AD, ARTIFACT_DIR, STEREOSCOPE)
    regional_context = build_regional_context(pathology, active, activation, carrier_rsp=carrier_rsp, immune_rsp=immune_rsp)
    rsp_celltypes = rsp_celltype_readback(regional_context, stereoscope)
    carrier_pathology = regional_context.assign(carrier_scope=np.where(regional_context['carrier_active'], 'active', 'inactive')).groupby(['carrier_scope', 'Classification'], observed=True).size().rename('n_spots').reset_index()
    carrier_pathology['fraction_within_scope'] = carrier_pathology['n_spots'] / carrier_pathology.groupby('carrier_scope')['n_spots'].transform('sum')
    (regional_pathology, pathology_contrasts) = pathology_context_tables(regional_context)
    (celltype_profile, celltype_contrasts) = celltype_context_tables(regional_context, stereoscope)
    regional_summary = regional_context['regional_context'].value_counts().rename_axis('regional_context').rename('n_spots').reset_index()
    regional_context = add_author_interface_score(regional_context, stereoscope)
    candidate_pairs = spec['candidate_ligand_receptor_pairs']['pairs']
    candidate_genes = sorted({gene for pair in candidate_pairs for gene in (pair['ligand'], pair['receptor'])})
    program_genes = selected_gene_sets['gene'].astype(str).unique().tolist()
    spatial_expression = load_log_normalized_expression(ST_H5AD, regional_context.index, sorted(set(candidate_genes) | set(program_genes)))
    author_lr = candidate_lr_readback(spatial_expression[candidate_genes], regional_context['author_top10'], candidate_pairs, analysis='author_abundance_top10')
    regal_lr = candidate_lr_readback(spatial_expression[candidate_genes], regional_context['regal_top10'], candidate_pairs, analysis='regal_interface_top10')
    concordance = interface_concordance(regional_context)
    (leave_score, leave_manifest) = leave_lr_out_interface(spatial_expression, rsp_loadings, carrier_rsp=carrier_rsp, immune_rsp=immune_rsp, excluded_genes=candidate_genes)
    regional_context['leave_lr_out_interface_score'] = leave_score
    regional_context['leave_lr_out_top10'] = leave_score.rank(method='first', ascending=False) <= int(np.ceil(len(leave_score) * 0.1))
    leave_lr = candidate_lr_readback(spatial_expression[candidate_genes], regional_context['leave_lr_out_top10'], candidate_pairs, analysis='leave_lr_out_interface_top10')
    leave_sensitivity = regal_lr.merge(leave_lr, on=['candidate_axis', 'ligand', 'receptor'], suffixes=('_primary', '_leave_out'))
    leave_sensitivity['direction_retained'] = (leave_sensitivity['fold_change_primary'] > 1) == (leave_sensitivity['fold_change_leave_out'] > 1)
    significant_regal_axes = regal_lr.loc[regal_lr['available'] & regal_lr['fold_change'].gt(1) & regal_lr['fdr_bh'].lt(0.05)]
    assert carrier_rsp == 'RSP0026'
    assert immune_rsp == 'RSP0024'
    assert regional_summary.set_index('regional_context').loc['RSP_overlap', 'n_spots'] > 0
    assert len(candidate_pairs) == 12
    assert len(significant_regal_axes) >= 8
    assert leave_sensitivity['direction_retained'].all()
    output_tables = assemble_r5_output_tables(sc_state_genes=sc_state_genes, translation_scores=translation_scores, selected_objects=selected_objects, selected_gene_sets=selected_gene_sets, selected_enrichment=selected_enrichment, source_caf_profile=source_caf_profile, source_caf_contrasts=source_caf_contrasts, source_caf_hotspots=source_caf_hotspots, rsp_celltypes=rsp_celltypes, carrier_pathology=carrier_pathology, regional_pathology=regional_pathology, pathology_contrasts=pathology_contrasts, celltype_profile=celltype_profile, celltype_contrasts=celltype_contrasts, regional_context=regional_context, author_lr=author_lr, regal_lr=regal_lr, leave_lr=leave_lr, concordance=concordance, leave_manifest=leave_manifest)
    for (filename, table) in output_tables.items():
        write_table(table, TABLE_DIR / filename)
    assert len(output_tables) == 9
    table_manifest = pd.DataFrame([{'source_table': filename, 'n_rows': len(table), 'n_columns': len(table.columns)} for (filename, table) in output_tables.items()])
    result_synthesis = pd.DataFrame([{'capability': 'Hypothesis retrieval', 'result': f'iCAF and T-cell queries select {carrier_rsp} and {immune_rsp}'}, {'capability': 'Regional program completion', 'result': 'selected RSP genes define tissue-specific molecular environments'}, {'capability': 'Context-dependent spatial testing', 'result': f'{len(significant_regal_axes)}/12 candidate axes are enriched at the regional interface'}])

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
    _CONFIG = document['fig08']
    prepare()
    missing = [name for name in TABLE_NAMES if not (TABLE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f'Missing prepared tables: {missing}')
    print(TABLE_DIR)

if __name__ == '__main__':
    main()
