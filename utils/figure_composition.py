from __future__ import annotations

"""File-level vector composition for manuscript figure rows."""

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


@dataclass(frozen=True)
class VectorPage:
    path: Path
    width: float
    height: float


@dataclass(frozen=True)
class VectorCrop:
    """Normalized crop in Cartesian coordinates."""

    left: float = 0.0
    bottom: float = 0.0
    right: float = 1.0
    top: float = 1.0

    def validate(self) -> None:
        if not (0 <= self.left < self.right <= 1):
            raise ValueError(f"Invalid horizontal crop: {self}")
        if not (0 <= self.bottom < self.top <= 1):
            raise ValueError(f"Invalid vertical crop: {self}")


def compose_vector_rows(
    row_stems: Sequence[str | Path],
    output_stem: str | Path,
    *,
    gap_points: float = 0.0,
    crops: Sequence[VectorCrop] | None = None,
) -> list[Path]:
    """Stack row-level SVG/PDF files while preserving vector content."""

    if not row_stems:
        raise ValueError("At least one row stem is required")
    if gap_points < 0:
        raise ValueError("gap_points must be non-negative")
    stems = [Path(stem) for stem in row_stems]
    if crops is None:
        crops = [VectorCrop() for _ in stems]
    if len(crops) != len(stems):
        raise ValueError("crops and row_stems must have the same length")
    for crop in crops:
        crop.validate()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_output = output_stem.with_suffix(".svg")
    pdf_output = output_stem.with_suffix(".pdf")
    _compose_svg_rows(
        [stem.with_suffix(".svg") for stem in stems],
        svg_output,
        gap_points,
        crops,
    )
    _compose_pdf_rows(
        [stem.with_suffix(".pdf") for stem in stems],
        pdf_output,
        gap_points,
        crops,
    )
    return [svg_output, pdf_output]


def png_content_crop(
    path: str | Path,
    *,
    pad_px: int,
    horizontal: bool = True,
    vertical: bool = True,
) -> VectorCrop:
    """Match a white-margin crop used by the raster composite."""

    image = np.asarray(Image.open(path).convert("RGBA"))
    foreground = (image[:, :, 3] > 12) & np.any(image[:, :, :3] < 251, axis=2)
    if not foreground.any():
        return VectorCrop()
    y, x = np.where(foreground)
    height, width = foreground.shape
    left_px = max(0, int(x.min()) - pad_px) if horizontal else 0
    right_px = min(width, int(x.max()) + pad_px + 1) if horizontal else width
    top_px = max(0, int(y.min()) - pad_px) if vertical else 0
    bottom_px = min(height, int(y.max()) + pad_px + 1) if vertical else height
    return VectorCrop(
        left=left_px / width,
        bottom=1 - bottom_px / height,
        right=right_px / width,
        top=1 - top_px / height,
    )


def _compose_svg_rows(
    row_paths: Sequence[Path],
    output_path: Path,
    gap_points: float,
    crops: Sequence[VectorCrop],
) -> None:
    rows: list[tuple[ET.Element, VectorPage, VectorCrop, tuple[float, float, float, float]]] = []
    for index, (path, crop) in enumerate(zip(row_paths, crops, strict=True)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing row SVG: {path}")
        root = ET.parse(path).getroot()
        viewbox = _svg_viewbox(root, path)
        _prefix_svg_ids(root, prefix=f"row{index}_")
        page = VectorPage(path=path, width=viewbox[2], height=viewbox[3])
        cropped_viewbox = (
            viewbox[0] + crop.left * viewbox[2],
            viewbox[1] + (1 - crop.top) * viewbox[3],
            (crop.right - crop.left) * viewbox[2],
            (crop.top - crop.bottom) * viewbox[3],
        )
        rows.append((root, page, crop, cropped_viewbox))

    width = max(viewbox[2] for _, _, _, viewbox in rows)
    height = sum(viewbox[3] for _, _, _, viewbox in rows) + gap_points * (len(rows) - 1)
    output = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{width:g}pt",
            "height": f"{height:g}pt",
            "viewBox": f"0 0 {width:g} {height:g}",
            "version": "1.1",
        },
    )
    y = 0.0
    for root, _, _, viewbox in rows:
        nested = ET.SubElement(
            output,
            f"{{{SVG_NS}}}svg",
            {
                "x": "0",
                "y": f"{y:g}",
                "width": f"{viewbox[2]:g}",
                "height": f"{viewbox[3]:g}",
                "viewBox": " ".join(f"{value:g}" for value in viewbox),
                "preserveAspectRatio": "xMinYMin meet",
            },
        )
        for child in list(root):
            nested.append(child)
        y += viewbox[3] + gap_points

    ET.ElementTree(output).write(output_path, encoding="utf-8", xml_declaration=True)


def _svg_viewbox(root: ET.Element, path: Path) -> tuple[float, float, float, float]:
    value = root.attrib.get("viewBox")
    if value is None:
        raise ValueError(f"SVG lacks a viewBox: {path}")
    parts = [float(part) for part in value.replace(",", " ").split()]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"Invalid SVG viewBox in {path}: {value!r}")
    return parts[0], parts[1], parts[2], parts[3]


def _prefix_svg_ids(root: ET.Element, *, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for element in root.iter():
        old_id = element.attrib.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            element.set("id", new_id)
    if not id_map:
        return

    url_pattern = re.compile(r"url\(#([^)]+)\)")
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if not isinstance(value, str):
                continue
            rewritten = url_pattern.sub(
                lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
                value,
            )
            if rewritten.startswith("#"):
                rewritten = f"#{id_map.get(rewritten[1:], rewritten[1:])}"
            element.set(key, rewritten)
        if element.text and "url(#" in element.text:
            element.text = url_pattern.sub(
                lambda match: f"url(#{id_map.get(match.group(1), match.group(1))})",
                element.text,
            )


def _compose_pdf_rows(
    row_paths: Sequence[Path],
    output_path: Path,
    gap_points: float,
    crops: Sequence[VectorCrop],
) -> None:
    pdftex = shutil.which("pdflatex")
    if pdftex is None:
        raise RuntimeError(
            "pdfLaTeX is required to assemble the vector PDF; "
            "row-level PDF and composite SVG exports remain available."
        )
    pages = [_pdf_page(path) for path in row_paths]
    visible_widths = [
        page.width * (crop.right - crop.left)
        for page, crop in zip(pages, crops, strict=True)
    ]
    visible_heights = [
        page.height * (crop.top - crop.bottom)
        for page, crop in zip(pages, crops, strict=True)
    ]
    width = max(visible_widths)
    rows = []
    for page, crop, visible_width in zip(
        pages,
        crops,
        visible_widths,
        strict=True,
    ):
        trim_left = crop.left * page.width
        trim_bottom = crop.bottom * page.height
        trim_right = (1 - crop.right) * page.width
        trim_top = (1 - crop.top) * page.height
        path = str(page.path.resolve()).replace("\\", "/")
        rows.append(
            "\\hbox to "
            f"{width:.8f}bp{{"
            "\\includegraphics["
            f"trim={trim_left:.8f}bp {trim_bottom:.8f}bp "
            f"{trim_right:.8f}bp {trim_top:.8f}bp,"
            f"clip,width={visible_width:.8f}bp"
            f"]{{\\detokenize{{{path}}}}}"
            "\\hss}"
        )
    body = f"\\kern {gap_points:.8f}bp\n".join(rows)
    tex_source = f"""\\documentclass[border=0pt]{{standalone}}
\\usepackage{{graphicx}}
\\begin{{document}}
\\vbox{{\\offinterlineskip
{body}
}}
\\end{{document}}
"""
    with tempfile.TemporaryDirectory(prefix="regal-vector-pdf-") as temp_dir:
        control_path = Path(temp_dir) / "compose_rows.tex"
        control_path.write_text(tex_source, encoding="utf-8")
        command = [
            pdftex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={temp_dir}",
            str(control_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        generated_pdf = Path(temp_dir) / "compose_rows.pdf"
        if result.returncode == 0 and generated_pdf.is_file():
            shutil.copyfile(generated_pdf, output_path)
    if result.returncode != 0:
        details = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        raise RuntimeError(
            "pdfLaTeX failed to assemble the vector PDF:\n"
            f"{details}"
        )


def _pdf_page(path: Path) -> VectorPage:
    if not path.is_file():
        raise FileNotFoundError(f"Missing row PDF: {path}")
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        raise RuntimeError("Poppler pdfinfo is required to inspect row-level PDF sizes.")
    completed = subprocess.run(
        [pdfinfo, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {path}: {completed.stderr.strip()}")
    match = re.search(
        r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        completed.stdout,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not read PDF page size from {path}")
    return VectorPage(path=path, width=float(match.group(1)), height=float(match.group(2)))
