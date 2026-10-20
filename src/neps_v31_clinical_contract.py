"""
NEPS V3.1 Clinical Contract & Feature Definition
================================================
Official 73-Item NSES (NEPS Stress Experience Scale) Feature Specifications
Frozen by NEPS Board & Data Platform Handover (October 2026).

This module provides the exact feature column names, subscale mappings, and
validation helpers for ML risk prediction, sentiment extraction, and trajectory models.
"""

import json
from typing import Dict, List, Any, Optional

# The 73 core items in official dictionary order
CORE73_FIELD_NAMES: List[str] = [
    # EPI (Educational Pressure Index - 10 items)
    "epi_expectations_1",
    "epi_expectations_2",
    "epi_failure_1",
    "epi_failure_2",
    "epi_caregivers_1",
    "epi_caregivers_2",
    "epi_future_1",
    "epi_future_2",
    "epi_progress_1",
    "epi_progress_2",
    # ESI (Economic Stress Index - 10 items)
    "esi_finstrain_1",
    "esi_finstrain_2",
    "esi_food_1",
    "esi_food_2",
    "esi_educost_1",
    "esi_educost_2",
    "esi_materials_1",
    "esi_materials_2",
    "esi_transport_1",
    "esi_transport_2",
    # SBI (Stigma and Burden Index - 19 items)
    "sbi_loved_1",
    "sbi_loved_2",
    "sbi_loved_3",
    "sbi_loved_4",
    "sbi_loved_5",
    "sbi_loved_6",
    "sbi_loved_7",
    "sbi_loved_8",
    "sbi_comm_1",
    "sbi_comm_2",
    "sbi_comm_3",
    "sbi_comm_4",
    "sbi_self_1",
    "sbi_self_2",
    "sbi_self_3",
    "sbi_self_4",
    "sbi_help_1",
    "sbi_help_2",
    "sbi_help_3",
    # K6 (Kessler Psychological Distress Scale - 6 items)
    "k6_1",
    "k6_2",
    "k6_3",
    "k6_4",
    "k6_5",
    "k6_6",
    # BRS (Brief Resilience Scale - 6 items)
    "brs_01",
    "brs_02",
    "brs_03",
    "brs_04",
    "brs_05",
    "brs_06",
    # MSPSS (Multidimensional Scale of Perceived Social Support - 12 items)
    "mspss_01",
    "mspss_02",
    "mspss_03",
    "mspss_04",
    "mspss_05",
    "mspss_06",
    "mspss_07",
    "mspss_08",
    "mspss_09",
    "mspss_10",
    "mspss_11",
    "mspss_12",
    # GSE (General Self-Efficacy Scale - 10 items)
    "gse_01",
    "gse_02",
    "gse_03",
    "gse_04",
    "gse_05",
    "gse_06",
    "gse_07",
    "gse_08",
    "gse_09",
    "gse_10",
]

# Clinical subscale grouping and metadata
SUBSCALES: Dict[str, Dict[str, Any]] = {
    "EPI": {
        "name": "Educational Pressure Index",
        "item_count": 10,
        "scale_range": (1, 5),
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("epi_")],
    },
    "ESI": {
        "name": "Economic Stress Index",
        "item_count": 10,
        "scale_range": (1, 5),
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("esi_")],
    },
    "SBI": {
        "name": "Stigma & Burden Index",
        "item_count": 19,
        "scale_range": (1, 5),
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("sbi_")],
    },
    "K6": {
        "name": "Kessler Psychological Distress Scale",
        "item_count": 6,
        "scale_range": (0, 4),  # Standard K6 0-4 frequency scale
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("k6_")],
    },
    "BRS": {
        "name": "Brief Resilience Scale",
        "item_count": 6,
        "scale_range": (1, 5),
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("brs_")],
    },
    "MSPSS": {
        "name": "Multidimensional Scale of Perceived Social Support",
        "item_count": 12,
        "scale_range": (1, 7),
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("mspss_")],
    },
    "GSE": {
        "name": "General Self-Efficacy Scale",
        "item_count": 10,
        "scale_range": (1, 4),
        "fields": [f for f in CORE73_FIELD_NAMES if f.startswith("gse_")],
    },
}


def extract_core73_features(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extracts the verified 73 features from a raw REDCap survey dictionary or database payload.
    Missing or unrecorded fields are represented as None.
    """
    features: Dict[str, Optional[float]] = {}
    for field in CORE73_FIELD_NAMES:
        val = record.get(field)
        if val is not None and val != "":
            try:
                features[field] = float(val)
            except (ValueError, TypeError):
                features[field] = None
        else:
            features[field] = None
    return features


def calculate_subscale_means(features: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Calculates subscale mean scores across the 7 validated domains.
    Useful as tabular feature inputs into ML classifiers.
    """
    subscale_scores: Dict[str, float] = {}
    for code, meta in SUBSCALES.items():
        vals = [features.get(f) for f in meta["fields"] if features.get(f) is not None]
        if vals:
            subscale_scores[code] = sum(vals) / len(vals)
        else:
            subscale_scores[code] = 0.0
    return subscale_scores


# ─── DERIVED SCORE CONTRACT (nds_v31_derived.result) ─────────────────────────
# The scores emitted by the Data Platform V3.1 scoring engine
# (etl/scoring/v31.py -> ScoringResult.scores). Values are 0–100 normalised
# except K6_total (0–24 raw scale). Any of these may be null when the
# underlying items are incomplete or (for EPI/NSES) applicability is unverified.
DERIVED_SCORE_KEYS: List[str] = [
    "EPI",
    "ESI",
    "SBI",
    "MHDI",
    "K6_total",
    "BRS_100",
    "MSPSS_100",
    "GSE_100",
    "PFI",
    "NSES",
]

DERIVED_SCORE_LABELS: Dict[str, str] = {
    "EPI": "Educational Pressure Index (0–100)",
    "ESI": "Economic Stress Index (0–100)",
    "SBI": "Stigma & Burden Index (0–100)",
    "MHDI": "Mental Health Distress Index / K6 normalised (0–100)",
    "K6_total": "Kessler-6 total raw score (0–24)",
    "BRS_100": "Brief Resilience Scale normalised (0–100)",
    "MSPSS_100": "Perceived Social Support normalised (0–100)",
    "GSE_100": "General Self-Efficacy normalised (0–100)",
    "PFI": "Protective Factors Index composite (0–100)",
    "NSES": "NEPS Stress Experience Scale composite (0–100; null unless EPI applicable)",
}


def _coerce_score(value: Any) -> Optional[float]:
    """Coerce a single stored score value to float, tolerating both stored shapes.

    The Data Platform has stored `nds_v31_derived.result` in two shapes:
      1. Flat:   {"NSES": 62.5, ...}
      2. Nested: {"scores": {"NSES": {"value": {"numerator": 5, "denominator": 8},
                                      "issues": [...]}}, "algorithm_version": ...}
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, dict):
        if "numerator" in value and "denominator" in value:
            numerator, denominator = value["numerator"], value["denominator"]
            try:
                denominator = float(denominator)
                return float(numerator) / denominator if denominator else None
            except (TypeError, ValueError):
                return None
        if "value" in value:
            return _coerce_score(value["value"])
    return None


def normalize_derived_result(result: Any) -> tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """Normalise a `nds_v31_derived.result` payload into flat scores + metadata.

    Returns:
        (scores, metadata) where `scores` is keyed by DERIVED_SCORE_KEYS and
        `metadata` holds any non-score fields (algorithm_version, provisional,
        nds_participant_id, mapping_version, ...).
    """
    if result is None or result == "":
        return {key: None for key in DERIVED_SCORE_KEYS}, {}
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return {key: None for key in DERIVED_SCORE_KEYS}, {}
    if not isinstance(result, dict):
        return {key: None for key in DERIVED_SCORE_KEYS}, {}

    nested = isinstance(result.get("scores"), dict)
    raw_scores = result.get("scores") if nested else result
    metadata = {k: v for k, v in result.items() if k != "scores"} if nested else {}

    scores = {key: _coerce_score(raw_scores.get(key)) for key in DERIVED_SCORE_KEYS}
    return scores, metadata
