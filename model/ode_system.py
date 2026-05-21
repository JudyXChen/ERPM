import pathlib

import dolfin as d
import sympy as sym

from smart import config, mesh, mesh_tools, model
from smart.model_assembly import (
    Compartment,
    CompartmentContainer,
    Parameter,
    ParameterContainer,
    Reaction,
    ReactionContainer,
    Species,
    SpeciesContainer,
)
from smart.units import unit

from .reactions import REGISTRARS

sec = unit.second
um = unit.micrometer
mV = unit.millivolt
dimensionless = unit.dimensionless

D_unit = um**2 / sec
conc_unit = dimensionless
rate_unit = dimensionless

CYTO_MARKER = 1
ER_MARKER = 2
PM_MARKER = 10
ERM_MARKER = 12

def build_compartments():
    Cyto = Compartment("Cyto", 3, um, CYTO_MARKER)
    ER = Compartment("ER", 3, um, ER_MARKER)
    PM = Compartment("PM", 2, um, PM_MARKER)
    ERm = Compartment("ERm", 2, um, ERM_MARKER)
    PM.specify_nonadjacency(["ER", "ERm"])
    ERm.specify_nonadjacency(["PM"])

    cc = CompartmentContainer()
    cc.add([Cyto, ER, PM, ERm])
    return cc


def build_shared_parameters(p, voltage_expr):
    if not isinstance(voltage_expr, sym.Expr):
        voltage_expr = sym.Float(float(voltage_expr))

    shared = [
        Parameter.from_expression(
            "V_m", voltage_expr, dimensionless, use_preintegration=False
        ),
        Parameter("V_spine", p.V_spine, dimensionless),
        Parameter("V_ER", p.V_ER, dimensionless),
        Parameter("V_spine_L", p.V_spine_L, dimensionless),
        Parameter("Ca_ext", p.Ca_ext, dimensionless),
        Parameter("F_const", p.F_const, dimensionless),
        Parameter("q_e", p.q_e, dimensionless),
        Parameter("N_A", p.N_A, dimensionless),
        Parameter("RT_F", p.RT_F, dimensionless),
        Parameter(
            "k_glu_clearance",
            float(getattr(p, "glu_clearance_rate", 0.0)),
            dimensionless,
        ),
    ]
    return shared

def build_mesh(
    outer_radius=0.25,
    inner_radius=0.1,
    h_edge=0.05,
    mesh_dir="mesh",
):
    mesh_folder = pathlib.Path(mesh_dir)
    mesh_folder.mkdir(exist_ok=True)
    mesh_path = mesh_folder / "ERPM_spheres.h5"

    domain, facet_markers, cell_markers = mesh_tools.create_spheres(
        outer_radius, inner_radius, hEdge=h_edge, verbose=False
    )
    mesh_tools.write_mesh(domain, facet_markers, cell_markers, filename=mesh_path)

    return mesh.ParentMesh(
        mesh_filename=str(mesh_path),
        mesh_filetype="hdf5",
        name="parent_mesh",
    )

def build_model(
    p,
    voltage_expr=-65.0,
    *,
    outer_radius=0.25,
    inner_radius=0.1,
    h_edge=0.05,
    mesh_dir="mesh",
    final_t=0.1,
    initial_dt=1e-4,
    time_precision=6,
):
    cc = build_compartments()
    pc = ParameterContainer()
    sc = SpeciesContainer()
    rc = ReactionContainer()

    pc.add(build_shared_parameters(p, voltage_expr))

    seen_params = {name for name, _ in pc.items}
    seen_species = set()
    seen_reactions = set()

    for register in REGISTRARS:
        species_list, params_list, reactions_list = register(p)
        for par in params_list:
            if par.name in seen_params:
                continue
            pc.add(par)
            seen_params.add(par.name)
        for sp in species_list:
            if sp.name in seen_species:
                continue
            sc.add(sp)
            seen_species.add(sp.name)
        for rx in reactions_list:
            if rx.name in seen_reactions:
                continue
            rc.add(rx)
            seen_reactions.add(rx.name)


    rc.add(
        Reaction(
            "glu_clearance",
            ["Glu"],
            [],
            param_map={"k": "k_glu_clearance"},
            eqn_f_str="k*Glu",
            species_map={"Glu": "Glu"},
        )
    )
    seen_reactions.add("glu_clearance")

    parent_mesh = build_mesh(
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        h_edge=h_edge,
        mesh_dir=mesh_dir,
    )

    cfg = config.Config()
    cfg.flags.update({"allow_unused_components": True})
    cfg.solver.update(
        {
            "final_t": final_t,
            "initial_dt": initial_dt,
            "time_precision": time_precision,
        }
    )

    m = model.Model(pc, sc, cc, rc, cfg, parent_mesh)
    m.initialize()

    return m, {"config": cfg, "parent_mesh": parent_mesh}


def species_names(p=None):
    """Ordered, de-duplicated list of state-variable names.

    Assembled from ``REGISTRARS`` without building the mesh/model, so it is
    cheap to call (e.g. for indexing result traces).
    """
    if p is None:
        from .parameters import default_parameters

        p = default_parameters()

    names = []
    seen = set()
    for register in REGISTRARS:
        species_list, _, _ = register(p)
        for sp in species_list:
            if sp.name not in seen:
                names.append(sp.name)
                seen.add(sp.name)
    return names


def run_model(
    m,
    run_info,
    *,
    output_dir="results",
):
    out = pathlib.Path(output_dir)
    out.mkdir(exist_ok=True)

    files = {}
    for species_name, _ in m.sc.items:
        f = d.XDMFFile(m.mpi_comm_world, str(out / f"{species_name}.xdmf"))
        f.parameters["flush_output"] = True
        f.write(m.sc[species_name].u["u"], m.t)
        files[species_name] = f

    while True:
        m.monolithic_solve()
        for species_name in files:
            files[species_name].write(m.sc[species_name].u["u"], m.t)
        if m.t >= m.final_t:
            break

    return files
