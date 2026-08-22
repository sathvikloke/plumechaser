"""Detection stack: background climatology, QA, blob extraction, persistence."""

from plumechaser.detect.background import robust_zscores, rolling_background, threshold_mask
from plumechaser.detect.naive import Blob, extract_blobs
from plumechaser.detect.qa import qa_mask
from plumechaser.detect.verify import Candidate, persist_candidates

__all__ = [
    "rolling_background",
    "robust_zscores",
    "threshold_mask",
    "Blob",
    "extract_blobs",
    "qa_mask",
    "Candidate",
    "persist_candidates",
]
