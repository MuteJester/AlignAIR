"""Profile sources and deterministic HMM banks for the amino-acid guard.

Existing imports remain available here. Source metadata lives in ``sources``,
HMM operations in ``bank``, and germline preparation in ``alignment``/``builder``.
Optional HMM tooling is loaded only when a build, load, or search is invoked.
"""
from .bank import ProfileBank, build_bank_fixture, build_profile, load_bank
from .bank import _amino, _serialize_deterministic
from .sources import (
    IMGT_CLEARANCE, PENDING, PROFILE_SOURCES, RECEPTOR_GROUPS, REQUIRED_PROFILES,
    HydrationResolution, NoPublishableProfileSource, ProductionBankResolution, ProfileSource,
    resolve_production_bank, source_matrix, synthetic_hydration_resolution,
)

__all__ = [
    "ProfileBank", "build_bank_fixture", "build_profile", "load_bank", "IMGT_CLEARANCE",
    "PENDING", "PROFILE_SOURCES", "RECEPTOR_GROUPS", "REQUIRED_PROFILES", "HydrationResolution",
    "NoPublishableProfileSource", "ProductionBankResolution", "ProfileSource",
    "resolve_production_bank", "source_matrix", "synthetic_hydration_resolution",
]
