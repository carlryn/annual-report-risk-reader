"""Render report pages and group adjacent pages into one call each.

Pages are rendered up front, sequentially (PyMuPDF is not thread-safe), so the calls only receive immutable data.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PageImage:
    """One rendered report page."""

    page: int  # 1-indexed PDF page
    png: bytes

    @property
    def data_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


@dataclass(frozen=True)
class PageGroup:
    """A run of adjacent pages, sent in one call."""

    images: tuple[PageImage, ...]

    @property
    def pages(self) -> list[int]:
        return [i.page for i in self.images]

    @property
    def label(self) -> str:
        """'p.85' for one page, 'p.71-74' for a run."""
        first, last = self.images[0].page, self.images[-1].page
        return f"p.{first}" if first == last else f"p.{first}-{last}"


def render_pages(pdf_path: Path, pages: Sequence[int], dpi: int) -> list[PageImage]:
    """Render `pages` (1-indexed) to PNG, in the given order."""
    with pymupdf.open(pdf_path) as doc:
        if outside := [p for p in pages if not 1 <= p <= len(doc)]:
            raise ValueError(f"{pdf_path.name} has {len(doc)} pages; asked for {outside}")
        return [PageImage(p, doc[p - 1].get_pixmap(dpi=dpi).tobytes("png")) for p in pages]


def adjacent_runs(images: Sequence[PageImage]) -> list[PageGroup]:
    """Group pages into runs of consecutive page numbers: 50, 51, 71, 72, 85 -> [50, 51], [71, 72], [85]."""
    runs: list[list[PageImage]] = []
    for image in sorted(images, key=lambda i: i.page):
        if runs and image.page == runs[-1][-1].page + 1:
            runs[-1].append(image)
        else:
            runs.append([image])
    return [PageGroup(tuple(run)) for run in runs]
