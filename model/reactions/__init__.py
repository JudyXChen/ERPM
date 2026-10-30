"""Reaction modules for the SMART-based ER-PM Ca2+ model.
"""

from . import (
    ampar,
    buffers,
    ip3r,
    mglur_gq_plc,
    mglur_plc_ip3,
    nmdar,
    pmca_ncx,
    ryr,
    serca,
    soce,
    vscc,
)

_COMMON_REGISTRARS = (
    nmdar.register,
    ampar.register,
    vscc.register,
    pmca_ncx.register,
    serca.register,
    ryr.register,
    ip3r.register,
    buffers.register,
    mglur_plc_ip3.register,
)


def registrars_for(plc="new", include_soce=False):
    """Return one complete registrar set for the selected PLC implementation."""
    if plc not in {"old", "new"}:
        raise ValueError(f"plc must be 'old' or 'new', got {plc!r}")
    registrars = _COMMON_REGISTRARS
    if plc == "new":
        registrars = tuple(
            serca.register_gq if register is serca.register else
            mglur_gq_plc.register if register is mglur_plc_ip3.register else
            register
            for register in registrars
        )
    if include_soce:
        registrars += (soce.register,)
    return registrars


BASE_REGISTRARS = registrars_for()
SOCE_REGISTRARS = registrars_for(include_soce=True)
REGISTRARS = BASE_REGISTRARS

__all__ = [
    "ampar",
    "buffers",
    "ip3r",
    "mglur_gq_plc",
    "mglur_plc_ip3",
    "nmdar",
    "pmca_ncx",
    "ryr",
    "serca",
    "soce",
    "vscc",
    "BASE_REGISTRARS",
    "SOCE_REGISTRARS",
    "REGISTRARS",
    "registrars_for",
]
