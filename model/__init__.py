from .parameters import default_parameters, DEFAULTS

__all__ = [
    "build_model",
    "run_model",
    "species_names",
    "default_parameters",
    "DEFAULTS",
]

def __getattr__(name):
    if name in {"build_model", "run_model", "species_names"}:
        from .ode_system import build_model, run_model, species_names

        return {
            "build_model": build_model,
            "run_model": run_model,
            "species_names": species_names,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
