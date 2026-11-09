"""Regenerate supported figures from recorded CSV/JSON only; never run a model.

Writes results/plot_manifest.json, verifies PNG pairs, preserves numerical data.
Optional --archive-superseded moves replaced combined images to _legacy_plots/.
Images without recorded sources are reported and left untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/erpm_mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/erpm-cache")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from scripts import plot_style as style
from scripts.io_utils import read_columns


def data_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for pattern in ("*.csv", "*.json", "*.npz") for p in root.rglob(pattern)
            if p.name != "plot_manifest.json" and "_legacy_plots" not in p.parts
            and "archive" not in p.parts}


def replot(root, *, archive_superseded=False):
    from scripts import plot_ode0d, plc_gq_analysis, er_flux_decomp, ryr_conductance
    from scripts import run_conditions
    from sensitivity_analysis import plots
    root = Path(root).resolve()
    original_data = data_hashes(root)
    old_images = {p for p in root.rglob("*.png")
                  if "fig" not in p.parts and "_legacy_plots" not in p.parts
                  and "archive" not in p.parts}
    style.EXPORTED.clear()
    superseded = set()
    skipped = []

    for source in root.rglob("traces.csv"):
        col = read_columns(source)
        # Preserve the established 0–50 ms view; shorter records use their full extent.
        end_ms = min(50.0, float(col["t"][-1]) * 1000)
        plot_ode0d.main(source.parent, t_max_ms=end_ms, write_tables=False)
        superseded.add(source.parent / "stimulus.png")

    for source in root.rglob("recovery_trace.csv"):
        folder = source.parent
        pulse = folder / "stimulus_trace.csv"
        if pulse.exists():
            plc_gq_analysis.plot_trace_columns(folder, read_columns(source), read_columns(pulse))
            superseded.update(folder / name for name in ("recovery.png", "stimulus_response.png"))
    for source in root.rglob("*.csv"):
        col = read_columns(source)
        if "ER_recorded_change_ions" in col and "cyto_recorded_change_ions" in col:
            flux, counts = {}, {}
            for compartment, prefix in (("ER", "ER"), ("cytosol", "cyto")):
                flux[compartment] = {
                    key[len(prefix)+1:-len("_ions_s")].replace("_", " "): values
                    for key, values in col.items()
                    if key.startswith(prefix + "_") and key.endswith("_ions_s")}
                counts[compartment] = col[f"{prefix}_recorded_change_ions"]
            er_flux_decomp.plot_budget_counts(source.with_suffix(".png"), col["time_s"], counts, flux)
            superseded.add(source.with_suffix(".png"))
        elif source.name == "er_flux_trace.csv":
            er_flux_decomp.plot_er_flux(source.parent / "er_flux_balance.png", col["time_s"],
                {key[:-len("_ions_s")].replace("_", " "): values for key, values in col.items()
                 if key.endswith("_ions_s")}, marker_time=plc_gq_analysis.PERTURB_TIME)

    for source in root.rglob("ryr_conductance.csv"):
        col = read_columns(source)
        metadata_path = source.with_name("ryr_conductance_metadata.json")
        samples = json.loads(metadata_path.read_text()).get("sample_values", []) if metadata_path.exists() else []
        resting_po = next((r["P_open"] for r in samples if r["Ca_c_uM"] == 0.1), None)
        ryr_conductance.plot_conductance(source.with_suffix(".png"), col["Ca_c_uM"],
                                        col["P_open"], ca_er=col["Ca_ER_uM"][0],
                                        resting_po=resting_po)
    for source in root.rglob("ip3_budget_summary.json"):
        plc_gq_analysis.plot_ip3_budget(source.parent)

    for source in root.rglob("sobol_indices.json"):
        results = json.loads(source.read_text())
        first = next(iter(results.values()))
        if "groups" in first:
            plots.plot_grouped(results, source.parent)
            superseded.update(source.parent / name for name in (
                "grouped_sobol.png", "grouped_sobol_heatmap.png",
                "grouped_sobol_calcium.png", "grouped_sobol_flux.png",
                "grouped_sobol_heatmap_calcium.png", "grouped_sobol_heatmap_flux.png"))
        else:
            skipped.append({"source": str(source), "reason": "Within-group names require an explicit plot_within_sobol_heatmap call"})
    for source in root.rglob("morris_indices.json"):
        plots.plot_morris(json.loads(source.read_text()), source.parent)
        superseded.add(source.parent / "morris_mu_sigma.png")
    for source in (root / "stimulus").glob("*/summary.json"):
        tag = source.parent.name
        if all((source.parent / key / "observables.csv").exists()
               and (root / "Flux" / tag / key / "budget.csv").exists()
               for key in style.CONDITIONS):
            run_conditions.OUT, run_conditions.FLUX = source.parent, root / "Flux" / tag
            run_conditions.plot()

    for item in style.EXPORTED:
        with Image.open(item["panel"]) as panel:
            assert panel.size == (900, 900), item
        with Image.open(item["view"]) as view:
            view.verify()
        if item["legend"]:
            with Image.open(item["legend"]) as legend:
                assert legend.mode == "RGBA" and legend.getextrema()[3][0] == 0, item
    assert data_hashes(root) == original_data, "Recorded numerical data changed"
    exported = {Path(item["view"]).resolve() for item in style.EXPORTED}
    archived = []
    if archive_superseded:
        for path in sorted(superseded & old_images):
            if path in exported:
                continue
            destination = root / "_legacy_plots" / path.relative_to(root)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite historical image: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), destination)
            archived.append(str(destination.relative_to(root)))
    unconverted = sorted(str(p.relative_to(root)) for p in old_images - exported
                         if p.exists() and p not in superseded)
    manifest = {"recorded_data_only": True, "numerical_data_unchanged": True,
                "panels": style.EXPORTED, "archived": archived,
                "unconverted_images": unconverted, "skipped": skipped}
    (root / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(style.EXPORTED)} PNG pairs; archived {len(archived)} superseded images.")
    if unconverted:
        print("Historical images without supported saved sources: " + ", ".join(unconverted))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--archive-superseded", action="store_true")
    args = parser.parse_args()
    replot(args.results, archive_superseded=args.archive_superseded)
