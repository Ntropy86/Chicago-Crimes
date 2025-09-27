"""Utility helpers for grouping Chicago crime types into higher-level categories."""

from __future__ import annotations

from typing import Dict, Iterable, Optional


# Baseline taxonomy derived from CPD IUCR groupings and City of Chicago dashboards.
CRIME_CATEGORY_MAP: Dict[str, str] = {
    # Violent crime
    "HOMICIDE": "Violent",
    "CRIM SEXUAL ASSAULT": "Violent",
    "SEX OFFENSE": "Violent",
    "BATTERY": "Violent",
    "ASSAULT": "Violent",
    "ROBBERY": "Violent",
    "KIDNAPPING": "Violent",
    "HUMAN TRAFFICKING": "Violent",
    "ARSON": "Violent",

    # Property crime
    "BURGLARY": "Property",
    "THEFT": "Property",
    "MOTOR VEHICLE THEFT": "Property",
    "CRIMINAL DAMAGE": "Property",
    "CRIMINAL TRESPASS": "Property",
    "DECEPTIVE PRACTICE": "Property",
    "OTHER OFFENSE": "Property",
    "WEAPONS VIOLATION": "Property",  # CPD lumps into property/contraband
    "GAMBLING": "Property",

    # Narcotics / vice
    "NARCOTICS": "Narcotics",
    "PUBLIC INDECENCY": "Narcotics",
    "PROSTITUTION": "Narcotics",

    # Quality-of-life / disorder
    "PUBLIC PEACE VIOLATION": "Disorder",
    "LIQUOR LAW VIOLATION": "Disorder",
    "INTERFERENCE WITH PUBLIC OFFICER": "Disorder",
    "OBSCENITY": "Disorder",
    "INTIMIDATION": "Disorder",
    "NON-CRIMINAL": "Disorder",
    "NON - CRIMINAL": "Disorder",
    "CONCEALED CARRY LICENSE VIOLATION": "Disorder",
    "OFFENSE INVOLVING CHILDREN": "Disorder",

    # Traffic / administrative
    "TRAFFIC VIOLATION": "Traffic",
    "OTHER NARCOTIC VIOLATION": "Traffic",
    "RITUALISM": "Traffic",
}


def categorize_primary_type(primary_type: Optional[str]) -> str:
    """Map a raw CPD primary_type to a broader analytical category."""
    if not primary_type:
        return "Unclassified"
    return CRIME_CATEGORY_MAP.get(primary_type.upper().strip(), "Unclassified")


def expand_categories(types: Iterable[str]) -> Dict[str, Iterable[str]]:
    """Group individual crime types by their assigned category."""
    grouped: Dict[str, set] = {}
    for t in types:
        category = categorize_primary_type(t)
        grouped.setdefault(category, set()).add(t)
    # Convert sets to sorted tuples for deterministic presentation
    return {cat: tuple(sorted(values)) for cat, values in grouped.items()}


__all__ = ["categorize_primary_type", "expand_categories", "CRIME_CATEGORY_MAP"]
