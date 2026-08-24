from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .common import quantiles


def _reason_counter(domains: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for d in domains:
        for r in d.get("qc_reject_reasons", []):
            counts[str(r)] += 1
    return {k: int(v) for k, v in sorted(counts.items(), key=lambda x: x[0])}


def build_rsp_metadomain_summary_table(
    rsp_summary_rows: list[dict],
    domains: list[dict],
) -> pd.DataFrame:
    by_rsp: dict[str, dict] = {str(r["rsp_id"]): dict(r) for r in rsp_summary_rows}
    for d in domains:
        pid = str(d["rsp_seed_id"])
        row = by_rsp.setdefault(pid, {"rsp_id": pid})
        row["domain_count_total"] = int(row.get("domain_count_total", 0)) + 1
        if bool(d.get("qc_pass", False)):
            row["domain_count_pass"] = int(row.get("domain_count_pass", 0)) + 1
        else:
            row["domain_count_reject"] = int(row.get("domain_count_reject", 0)) + 1

    df = pd.DataFrame(list(by_rsp.values()))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "rsp_id",
                "candidate_domain_count",
                "screening_pass_count",
                "domain_count_total",
                "domain_count_pass",
                "domain_count_reject",
                "no_domain_rsp",
            ]
        )

    for col in ["domain_count_total", "domain_count_pass", "domain_count_reject"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(np.int32)
    df["no_domain_rsp"] = df["domain_count_pass"] <= 0
    return df.sort_values("rsp_id").reset_index(drop=True)


def build_qc_report(
    sample_id: str,
    n_spots: int,
    n_rsps: int,
    segmentation_summary: dict,
    domains: list[dict],
    rsp_summary_table: pd.DataFrame,
    acceptance_gates: dict,
) -> dict:
    spot_counts = np.asarray([float(x.get("spot_count", 0)) for x in domains], dtype=np.float32)
    peak_vals = np.asarray([float(x.get("rsp_peak_value", 0.0)) for x in domains], dtype=np.float32)
    prominences = np.asarray([float(x.get("rsp_prominence", 0.0)) for x in domains], dtype=np.float32)
    mass_vals = np.asarray([float(x.get("rsp_seed_sum", 0.0)) for x in domains], dtype=np.float32)
    enrich_ratio_vals = np.asarray([float(x.get("rsp_mean_enrichment_ratio", 0.0)) for x in domains], dtype=np.float32)
    enrich_delta_vals = np.asarray([float(x.get("rsp_mean_enrichment_delta", 0.0)) for x in domains], dtype=np.float32)

    passed = [x for x in domains if bool(x.get("qc_pass", False))]
    passed_spot_counts = np.asarray([float(x.get("spot_count", 0)) for x in passed], dtype=np.float32)
    passed_peak = np.asarray([float(x.get("rsp_peak_value", 0.0)) for x in passed], dtype=np.float32)
    passed_prom = np.asarray([float(x.get("rsp_prominence", 0.0)) for x in passed], dtype=np.float32)
    passed_enrich_ratio = np.asarray([float(x.get("rsp_mean_enrichment_ratio", 0.0)) for x in passed], dtype=np.float32)
    passed_enrich_delta = np.asarray([float(x.get("rsp_mean_enrichment_delta", 0.0)) for x in passed], dtype=np.float32)

    accepted_rsp_count = 0
    no_domain_rsp_count = 0
    if not rsp_summary_table.empty and "no_domain_rsp" in rsp_summary_table.columns:
        no_domain_rsp_count = int(np.sum(rsp_summary_table["no_domain_rsp"].to_numpy(dtype=bool)))
        accepted_rsp_count = int(rsp_summary_table.shape[0] - no_domain_rsp_count)

    return {
        "sample_id": sample_id,
        "inputs_summary": {
            "n_spots": int(n_spots),
            "n_rsps": int(n_rsps),
        },
        "segmentation_summary": segmentation_summary,
        "candidate_domain_summary": {
            "candidate_domain_count": int(len(domains)),
            "candidate_spot_count_quantiles": quantiles(spot_counts),
            "candidate_peak_quantiles": quantiles(peak_vals),
            "candidate_prominence_quantiles": quantiles(prominences),
            "candidate_mass_quantiles": quantiles(mass_vals),
            "candidate_mean_enrichment_ratio_quantiles": quantiles(enrich_ratio_vals),
            "candidate_mean_enrichment_delta_quantiles": quantiles(enrich_delta_vals),
            "qc_pass_domain_count": int(len(passed)),
            "qc_pass_spot_count_quantiles": quantiles(passed_spot_counts),
            "qc_pass_peak_quantiles": quantiles(passed_peak),
            "qc_pass_prominence_quantiles": quantiles(passed_prom),
            "qc_pass_mean_enrichment_ratio_quantiles": quantiles(passed_enrich_ratio),
            "qc_pass_mean_enrichment_delta_quantiles": quantiles(passed_enrich_delta),
        },
        "filter_summary": {
            "reject_reason_counts": _reason_counter(domains),
        },
        "rsp_domain_summary": {
            "accepted_rsp_count": int(accepted_rsp_count),
            "no_domain_rsp_count": int(no_domain_rsp_count),
            "qc_pass_domains_per_rsp_quantiles": quantiles(
                rsp_summary_table["domain_count_pass"].to_numpy(dtype=np.float32)
                if ("domain_count_pass" in rsp_summary_table.columns and not rsp_summary_table.empty)
                else np.zeros(0, dtype=np.float32)
            ),
        },
        "acceptance": acceptance_gates,
    }
