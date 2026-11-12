"""Which pages of an extraction run can be scored.

Only pages that are both labelled and sent are scored. A page whose extraction call failed is scored: its golden
risks count as missed, because that is what the analyst would get.
"""

from __future__ import annotations

import pandas as pd

from ..extraction.runs import ExtractionRun
from .golden import Golden


def coverage(run: ExtractionRun, golden: Golden) -> pd.DataFrame:
    """Per page: golden count, status (ok | failed | not sent | not labelled), and whether it is scored."""
    sent, failed = set(run.config.pages), set(run.failed_pages)
    rows = []
    for p in sorted(sent | set(golden)):
        status = (
            "not sent" if p not in sent else "not labelled" if p not in golden else "failed" if p in failed else "ok"
        )
        rows.append(
            {"page": p, "golden": len(golden.get(p, [])), "status": status, "scored": p in sent and p in golden}
        )
    return pd.DataFrame(rows).set_index("page")
