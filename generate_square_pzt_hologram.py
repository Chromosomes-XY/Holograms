from hologramGen import (
    acoustic_params,
    design_hologram_single_plane,
    export_hologram_stl,
    impedance_params,
    make_center_dot_target,
    make_grid,
    make_square_pzt,
    plot_hologram_outputs,
    project_hologram,
)


# Simulation grid
N = 512
SIMULATION_WIDTH_M = 0.045

# PZT and single point-focus target
PZT_SIDE_M = 0.04
TARGET_DOT_DIAMETER_M = 0.1e-3
FOCAL_DISTANCE_M = 20e-3
N_OPTIMIZATION_ITERATIONS = 300

# Thickness caps applied inside the designer before export
MIN_THICKNESS_M = 0.4e-3
MAX_THICKNESS_M = 3.0e-3

# Material/acoustic parameters
FREQUENCY_HZ = 2.05e6
C_MEDIUM = 1500.0
C_HOLOGRAM = 2550.0

PROJECTED_DOMAIN_MM = (-3, 3, -3, 3)

PLOT_OUTPUT = "square_pzt_hologram_output.png"
STL_OUTPUT = "square_pzt_hologram_output.stl"


def main(export_stl=True):
    _, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    p0_amp = make_square_pzt(xv, yv, PZT_SIDE_M)
    target_amp = make_center_dot_target(xv, yv, TARGET_DOT_DIAMETER_M)

    acoustic = acoustic_params(
        frequency_hz=FREQUENCY_HZ,
        c_medium=C_MEDIUM,
        c_hologram=C_HOLOGRAM,
    )
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)

    T, hist = design_hologram_single_plane(
        p0_amp=p0_amp,
        target_at_z=target_amp,
        z_target=FOCAL_DISTANCE_M,
        dx=dx,
        lam_m=acoustic["lam_m"],
        Zt=impedance["Zt"],
        Zh=impedance["Zh"],
        Zm=impedance["Zm"],
        kh=acoustic["kh"],
        km=acoustic["km"],
        n_iter=N_OPTIMIZATION_ITERATIONS,
        Tmin=MIN_THICKNESS_M,
        Tmax=MAX_THICKNESS_M,
    )

    p_img = project_hologram(
        p0_amp=p0_amp,
        T=T,
        dx=dx,
        lam_m=acoustic["lam_m"],
        z_target=FOCAL_DISTANCE_M,
        Zt=impedance["Zt"],
        Zh=impedance["Zh"],
        Zm=impedance["Zm"],
        kh=acoustic["kh"],
        km=acoustic["km"],
    )
    if export_stl:
        n_facets, nonmanifold_edges = export_hologram_stl(T, dx, STL_OUTPUT)
    plot_hologram_outputs(
        xv=xv,
        yv=yv,
        T=T,
        p_img=p_img,
        z_target=FOCAL_DISTANCE_M,
        output_path=PLOT_OUTPUT,
        projected_domain_mm=PROJECTED_DOMAIN_MM,
    )

    print("Final amp MSE:", hist["amp_mse"][-1])
    print("Final amp corr:", hist["amp_corr"][-1])
    print("Target focal distance [mm]:", FOCAL_DISTANCE_M * 1e3)
    print(f"Saved plot to {PLOT_OUTPUT}")
    if export_stl:
        print(f"Saved STL to {STL_OUTPUT} ({n_facets} facets)")
        print(f"STL nonmanifold edge count: {nonmanifold_edges}")


if __name__ == "__main__":
    main()
