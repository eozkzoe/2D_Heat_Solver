"""
Main executable for CTFD project
"""

from heat import Diffusive, ConvectiveDiffusive
from utils import SourceSettings, FlowSettings, combine_meshes
from plottools import plot_tmap
from greens import AnalyticalTemp
from stokes import StokesSolver
from blocks import get_combined_boundary, get_block_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":

    source = SourceSettings(
        dimX=20,
        dimY=10,
        N=20,
        M=20,
        shape="linear",
        boundary=["R", "N", "N", "N"],
        TD=[0, 0, 0, 0],
        therm_con=1,
        Tinf=0,
        alpha=1,
        time_steps=1000,
        problem_type="unsteady",
        source_region=(8, 8, 12, 12),
        observer_region=(0, 0, 20, 20),
        bc_x=11,
        bc_y=12,
        G_N=5,
        G_M=5,
        mirrored=True,
    )

    source.setUpMesh(source.N, source.M)

    fvm_source = Diffusive(
        source.N,
        source.M,
        source.X,
        source.Y,
        source.D,
        source.dX,
        boundary=source.boundary,
        TD=source.TD,
        q=source.q,
        alpha=source.alpha,
        Tinf=source.Tinf,
    )

    fvm_source.update_AB("sparse", use_inner_at="N")

    # Get green's contribution from source
    temp_dist = AnalyticalTemp(
        source.dimX + 2,
        source.therm_con,
        source.G_N,
        source.G_M,
        source.bc_x,
        source.bc_y,
        source.source_field,
        source.observer_field,
        alphas=source.alphas,
    )
    greens_field = temp_dist.compute_contributions()

    """
    For individual source solver

    source_T_map = fvm_source.solve(
        problem_type=source.problem_type,
        solver_type="jacobi",
        matrix_type="sparse",
        theta=source.theta,
        t_steps=source.time_steps,
    )

    source_green_T_map = combine_finite_diff_and_green(source_T_map, greens_field)

    source.setUpMesh(source.N - 2, source.M - 2)

    plot_tmap(
        source_green_T_map,
        source.shape,
        source.N,
        source.M,
        source.X,
        source.Y,
        source.problem_type,
        timeline=fvm_source.timeline,
        t_steps=source.time_steps,
    )
    """

    fluid = FlowSettings(
        dimX=20,
        dimY=10,
        N=20,
        M=20,
        shape="rectangular",
        boundary=["N", "N", "D", "N"],
        therm_con=50,
        TD=[0, 0, 0, 0],
        Tinf=0,
        alpha=2,
        time_steps=3000,
        problem_type="unsteady",
        mu=5.0,  # Dynamic viscosity coefficient
        U_in=5,  # velocity at inlet
        U_out=0,  # velocity at outlet
        u_bc_type={"left": "D", "right": "N", "top": "D", "bottom": "D"},
        v_bc_type={"left": "D", "right": "N", "top": "D", "bottom": "D"},
        p_bc_type={"left": "N", "right": "D", "top": "N", "bottom": "N"},
    )

    stokes = StokesSolver(
        fluid.dimX,
        fluid.dimY,
        fluid.N,
        fluid.M,
        fluid.mu,
        fluid.U_in,
        fluid.U_out,
    )
    stokes.set_boundary_conditions(
        fluid.u_bc_type,
        fluid.v_bc_type,
        fluid.p_bc_type,
    )
    stokes.solve_stokes(
        fluid.u_bc_type,
        fluid.v_bc_type,
        fluid.p_bc_type,
        max_iter=10000,
        tol=1e-5,
    )

    # stokes.plot_results()

    fluid.setUpMesh(fluid.N, fluid.M)

    fvm_fluid = ConvectiveDiffusive(
        fluid.N,
        fluid.M,
        fluid.X,
        fluid.Y,
        fluid.D,
        fluid.dX,
        # 0.5,
        stokes.u.T.flatten(),
        boundary=fluid.boundary,
        TD=fluid.TD,
        q=fluid.q,
        alpha=fluid.alpha,
        Tinf=fluid.Tinf,
    )

    fvm_fluid.update_AB("sparse", use_inner_at="S")

    print("Starting combined solver")

    """
    # For individual fluid solver:

    fluid_T_map = fvm_fluid.solve(
        problem_type=fluid.problem_type,
        solver_type="jacobi",
        matrix_type="sparse",
        theta=fluid.theta,
        t_steps=fluid.time_steps,
    )

    fluid.setUpMesh(fluid.N - 2, fluid.M - 2)

    plot_tmap(
        fluid_T_map,
        fluid.shape,
        fluid.N,
        fluid.M,
        fluid.X,
        fluid.Y,
        fluid.problem_type,
        timeline=fvm_fluid.timeline,
        t_steps=fluid.time_steps,
    )
    """

    fvm_fluid.B = get_combined_boundary(
        solid_boundary=fvm_source.B,
        fluid_boundary=fvm_fluid.B,
        order="fs",
    )

    fvm_fluid.A = get_block_matrix(
        bottom_mesh=fvm_source.A.toarray(),
        top_mesh=fvm_fluid.A.toarray(),
        shape=(fvm_fluid.n, fvm_fluid.m),
    )

    fvm_fluid.sparse_or_dense(fvm_fluid.A, "sparse")

    fvm_fluid.n = fvm_fluid.n + fvm_source.n
    fvm_fluid.combine_boundaries(fluid.TD, source.TD)
    fvm_fluid.inspect_matrix()
    flow_T_map = fvm_fluid.solve(
        problem_type=fluid.problem_type,
        solver_type="jacobi",
        matrix_type="sparse",
        theta=fluid.theta,
        t_steps=fluid.time_steps,
        update=False,
        conjugate=True,
        greens=greens_field,
        green_limit=None,
    )

    combined_X, combined_Y = combine_meshes(
        bot_x=source.X, bot_y=source.Y, top_x=fluid.X, top_y=fluid.Y
    )

    plot_tmap(
        flow_T_map,
        fluid.shape,
        fluid.N + source.N,
        fluid.M,
        # fluid.X,
        # fluid.Y,
        combined_X,
        combined_Y,
        fluid.problem_type,
        timeline=fvm_fluid.timeline,
        t_steps=fluid.time_steps,
    )
