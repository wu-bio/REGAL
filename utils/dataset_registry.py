from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DatasetSpec:
    dataset_family: Literal["pan_cancer", "external_dataset"]
    dataset_id: str
    sample_ids: tuple[str, ...]
    species: str = "human"
    tissue: str | None = None
    condition: str | None = None
    modality: str = "ST"
    work_dir: Path = Path("/Users/wuyang/Documents/SC-ST data")
    data_layer: str = "X"
    spatial_obsm_key: str = "spatial"
    spot_id_field: str | None = None
    annotation_col: str | None = None
    gene_id_source: str = "var_names"
    expression_state: Literal["raw_counts", "normalized", "log_normalized", "unknown"] = "raw_counts"

    def h5ad_path(self, sample_id: str) -> Path:
        return self.work_dir / self.dataset_id / self.modality / f"{sample_id}.h5ad"

    def out_root(self) -> Path:
        return self.work_dir / self.dataset_id / self.modality

    def sample_dir(self, sample_id: str) -> Path:
        return self.out_root() / sample_id


DATASETS: dict[str, DatasetSpec] = {
    "SIMULATION": DatasetSpec(
        "external_dataset",
        "SIMULATION",
        (
            "regional_activity_16_heterogeneous_seed2026",
        ),
        species="synthetic",
        condition="simulated regional gene activities",
        expression_state="raw_counts",
    ),
    "COAD": DatasetSpec(
        "pan_cancer",
        "COAD",
        ("TENX89", "TENX90", "TENX91", "TENX92"),
        tissue="colon",
        condition="cancer",
    ),
    "PAAD": DatasetSpec(
        "pan_cancer",
        "PAAD",
        ("NCBI569", "NCBI570", "NCBI571", "NCBI572"),
        tissue="pancreas",
        condition="cancer",
    ),
    "IDC": DatasetSpec(
        "pan_cancer",
        "IDC",
        ("NCBI681", "NCBI682", "NCBI683", "TENX13", "TENX14"),
        tissue="breast",
        condition="cancer",
    ),
    "PRAD": DatasetSpec(
        "pan_cancer",
        "PRAD",
        ("INT25", "INT26", "INT27", "INT28", "TENX40", "TENX46"),
        tissue="prostate",
        condition="cancer",
    ),
    "EPM": DatasetSpec(
        "pan_cancer",
        "EPM",
        ("NCBI629", "NCBI630", "NCBI631", "NCBI632", "NCBI633"),
        tissue="ependymoma",
        condition="cancer",
    ),
    "CRC": DatasetSpec(
            "external_dataset",
            "CRC",
            (
                "ZEN36",
                "ZEN40",
                "ZEN42",
                "ZEN43",
                "ZEN46",
                "ZEN47",
                "ZEN48",
                "ZEN49",
            ),
            tissue="colorectum",
            condition="cancer",
            annotation_col="Pathologist Annotation",
            expression_state="raw_counts",
        ),

    "BREAST": DatasetSpec(
        "external_dataset",
        "BREAST",
        (
            "Breast_BlockA_Section1",
        ),
        tissue="breast",
        condition="cancer",
        annotation_col="manual_annotation_fine",
    ),
    "JANESICK": DatasetSpec(
        "external_dataset",
        "Janesick",
        (
            "Janesick_Visium_Sample1",
        ),
        tissue="breast",
        condition="cancer",
        annotation_col="janesick_annotation",
        expression_state="raw_counts",
    ),
    "WU": DatasetSpec(
        "external_dataset",
        "Wu",
        (
            "Wu_Breast_CID44971",
        ),
        tissue="breast",
        condition="triple-negative breast cancer",
        annotation_col="manual_annotation_fine",
        expression_state="raw_counts",
    ),
    "LIVER": DatasetSpec(
        "external_dataset",
        "LIVER",
        (
            "AM042",
            "AM048",
            "AM061",
        ),
        tissue="liver",
        condition="healthy",
        annotation_col="cell_type_annotation",
        expression_state="raw_counts",
    ),
    "BRAIN": DatasetSpec(
        "external_dataset",
        "BRAIN",
        (
            "MouseBrain_Xenium",
        ),
        species="mouse",
        tissue="brain",
        condition="normal_brain",
        expression_state="raw_counts",
    ),
    "MOB": DatasetSpec(
        "external_dataset",
        "MOB",
        (
            "Puck_200127_15",
            "Stereo_seq",
        ),
        species="mouse",
        tissue="olfactory bulb",
        condition="normal_brain",
        expression_state="raw_counts",
    ),
}


def get_dataset(dataset_id: str) -> DatasetSpec:
    key = str(dataset_id).strip().upper()
    try:
        return DATASETS[key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset_id {dataset_id!r}. Available datasets: {available}") from exc
