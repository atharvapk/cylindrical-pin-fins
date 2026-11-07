import numpy as np
import matplotlib.pyplot as mp
from scipy.sparse import csr_array
import time

## Declare constants
D_pin = 1.2e-3
alpha = 0.75
num_h = 15
num_r = 10
k_values = [0.1, 0.621, 1, 10, 100, 250, 500, 750]

h_conv = 400       # W/(m^2 K)
T_inf = 10         # Celsius
q_flux = 10000     # W/m^2

## Discretization of computational domain
H_pin = alpha*D_pin
dh = H_pin/(num_h-1)
dr = ((D_pin/2)/(num_r-1))

num_nodes = num_h*num_r

# Integer node indices, ordered by height then radius.
neighbor_map = np.arange(num_nodes).reshape(num_h, num_r)

## Area functions
radius_list = np.linspace(0, D_pin/2, num_r)
height_list = np.linspace(0, H_pin, num_h)

A_l = lambda i: 2*np.pi*(radius_list[i] - dr/2)*dh
A_r = lambda i: 2*np.pi*(radius_list[i] + dr/2)*dh
A_h = lambda i: 2*np.pi*radius_list[i]*dr

A_left_edge_h = np.pi*((dr/2)**2)

def solve_fin(k):
    """Solve Design 1 for k; numerical efficiency uses area-averaged base T."""
    ## Create NxN matrix
    M = np.zeros([num_nodes, num_nodes])

    ## Create Nx1 Array
    b = np.zeros([num_nodes,1])

    ## Build Coefficient Matrix (M)

    ## Boundary nodes: insulated tip, heated base, and convecting outer side.
    ## Shared assembly below handles these node types:
    ## Centerline + insulated-tip corner (1): h == 0, r == 0
    ## Centerline + base-flux corner (2): h == num_h-1, r == 0
    ## Insulated-tip + convection corner (3): h == 0, r == num_r-1
    ## Base-flux + convection corner (4): h == num_h-1, r == num_r-1
    ## Outer convection edge (6): 0 < h < num_h-1, r == num_r-1
    ## Base heat-flux edge (8): h == num_h-1, 0 < r < num_r-1
    ## Insulated-tip edge (9): h == 0, 0 < r < num_r-1
    ## h = 0 is the tip; h = num_h-1 is the base.
    ## Clipped annuli give the correct axial areas at both radial edges.
    # Select boundary nodes, including corners.
    for h in range(num_h):
        for r in range(num_r):
            if h not in (0, num_h-1) and r != num_r-1:
                continue

            # Control-volume geometry.
            center_index = neighbor_map[h, r]
            r_inner = max(0, radius_list[r] - dr/2)
            r_outer = min(D_pin/2, radius_list[r] + dr/2)
            axial_area = np.pi*(r_outer**2 - r_inner**2)
            cv_height = dh/2 if h in (0, num_h-1) else dh
            den = 0.0

            # Axial conduction.
            # Only existing axial neighbors conduct; no heat crosses the tip.
            for axial_h in (h-1, h+1):
                if 0 <= axial_h < num_h:
                    conductance = k*axial_area/dh
                    M[center_index, neighbor_map[axial_h, r]] = conductance
                    den += conductance

            # Radial conduction toward the axis.
            if r > 0:
                conductance = k*2*np.pi*r_inner*cv_height/dr
                M[center_index, neighbor_map[h, r-1]] = conductance
                den += conductance
            # Outward conduction or surface convection.
            if r < num_r-1:
                conductance = k*2*np.pi*r_outer*cv_height/dr
                M[center_index, neighbor_map[h, r+1]] = conductance
                den += conductance
            else:
                convection = h_conv*2*np.pi*(D_pin/2)*cv_height
                den += convection
                b[center_index] += convection*T_inf

            # Base heat input and normalized coefficients.
            if h == num_h-1:
                b[center_index] += q_flux*axial_area

            M[center_index, :] /= den
            b[center_index] /= den

    ## Centerline symmetry edge (7)
    r = 0
    for h in range(1, num_h-1):
        center_index = neighbor_map[h,r]
        top_neigh_index, bottom_neigh_index = neighbor_map[h-1,r], neighbor_map[h+1,r]
        lateral_neigh_index = neighbor_map[h,r+1]

        # Use the full disk area for each axial face.
        axial_conductance = k*A_left_edge_h/dh
        radial_conductance = k*A_r(r)/dr
        den = 2*axial_conductance + radial_conductance
        M[center_index, top_neigh_index] = axial_conductance/den
        M[center_index, bottom_neigh_index] = axial_conductance/den
        M[center_index, lateral_neigh_index] = radial_conductance/den

    ## Interior node (5)
    for h in range(1, num_h-1):
        for r in range(1, num_r-1):
            center_index = neighbor_map[h, r]
            top_neigh_index, bottom_neigh_index = neighbor_map[h-1, r], neighbor_map[h+1, r]
            right_neigh_index, left_neigh_index = neighbor_map[h, r+1], neighbor_map[h, r-1]

            axial_conductance = k*A_h(r)/dh
            right_conductance = k*A_r(r)/dr
            left_conductance = k*A_l(r)/dr
            den = 2*axial_conductance + right_conductance + left_conductance
            M[center_index, top_neigh_index] = axial_conductance/den
            M[center_index, bottom_neigh_index] = axial_conductance/den
            M[center_index, right_neigh_index] = right_conductance/den
            M[center_index, left_neigh_index] = left_conductance/den

    M_sparse = csr_array(M)
    height_weights = np.full(num_h, dh)
    height_weights[[0, -1]] /= 2
    Q_in = q_flux*np.pi*(D_pin/2)**2
    side_area = 2*np.pi*(D_pin/2)*H_pin
    # Start near the solution to avoid slow uniform-temperature relaxation at high k.
    m_fin = np.sqrt(4*h_conv/(k*D_pin))
    initial_profile = T_inf + q_flux*np.cosh(m_fin*height_list)/(k*m_fin*np.sinh(m_fin*H_pin))
    old_temp = np.repeat(initial_profile, num_r).reshape(num_nodes, 1)
    start = time.time()

    # At large k, small iteration changes alone can hide a heat imbalance.
    for iteration in range(1, 5_000_001):
        new_temp = M_sparse@old_temp + b
        error = np.max(np.abs(new_temp - old_temp))
        old_temp = new_temp
        if error < 1e-8:
            surface_temp = new_temp.reshape(num_h, num_r)[:, -1]
            Q_out = h_conv*2*np.pi*(D_pin/2)*np.dot(
                height_weights, surface_temp - T_inf
            )
            relative_heat_imbalance = abs(Q_in - Q_out)/Q_in
            if relative_heat_imbalance < 1e-6:
                break
    else:
        raise RuntimeError(f"Solver did not converge for k={k}")

    temp_map1 = new_temp.reshape(num_h, num_r)
    # Base temperature is averaged over the exact annular control-volume areas.
    r_inner = np.maximum(0, radius_list - dr/2)
    r_outer = np.minimum(D_pin/2, radius_list + dr/2)
    base_areas = np.pi*(r_outer**2 - r_inner**2)
    base_temp = np.dot(base_areas, temp_map1[-1, :])/base_areas.sum()
    eta_numerical = Q_out/(h_conv*side_area*(base_temp - T_inf))
    m_fin = np.sqrt(4*h_conv/(k*D_pin))
    eta_analytical = np.tanh(m_fin*H_pin)/(m_fin*H_pin)

    return {
        "k": k,
        "eta_numerical": eta_numerical,
        "eta_analytical": eta_analytical,
        "base_temp": base_temp,
        "Q_out": Q_out,
        "relative_heat_imbalance": relative_heat_imbalance,
        "iterations": iteration,
        "seconds": time.time() - start,
    }


if __name__ == "__main__":
    print("Numerical efficiency uses the area-averaged base temperature.")
    print(f"{'k (W/m K)':>10} {'eta numerical':>15} {'eta analytical':>15} "
          f"{'heat imbalance':>16}", flush=True)
    results = []
    for k in k_values:
        result = solve_fin(k)
        results.append(result)
        print(f"{k:10g} {result['eta_numerical']:15.8f} "
              f"{result['eta_analytical']:15.8f} "
              f"{result['relative_heat_imbalance']:16.3e}", flush=True)

    conductivities = np.array([row["k"] for row in results])
    efficiency_error_percent = 100*np.abs(1 - np.array([
        row["eta_numerical"]/row["eta_analytical"] for row in results
    ]))
    closest_index = np.argmin(np.abs(efficiency_error_percent - 1.0))
    closest_k = conductivities[closest_index]
    closest_error = efficiency_error_percent[closest_index]
    print(f"Closest sampled point to 1%: "
          f"(k, error) = ({closest_k:.12g} W/(m K), {closest_error:.12g}%)")

    fig, ax = mp.subplots()
    ax.semilogx(conductivities, efficiency_error_percent, "o-",
                label="Calculated efficiency error")
    ax.axhline(1.0, color="gray", linestyle="--", label="1% threshold")
    ax.plot(closest_k, closest_error, marker="o", color="red",
            markersize=8, linestyle="none",
            label="Closest sampled point to 1%")
    ax.annotate(
        f"Closest sampled point\n"
        f"k = {closest_k:.12g} W/(m K)\n"
        f"error = {closest_error:.12g}%",
        xy=(closest_k, closest_error), xytext=(25, 55),
        textcoords="offset points", arrowprops={"arrowstyle": "->"},
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.9},
    )
    # Interpolate error linearly in log(k), matching the plotted line segments.
    crossing_intervals = np.flatnonzero(
        (efficiency_error_percent[:-1] - 1.0)
        * (efficiency_error_percent[1:] - 1.0) <= 0
    )
    if crossing_intervals.size:
        i = crossing_intervals[0]
        fraction = ((1.0 - efficiency_error_percent[i])
                    / (efficiency_error_percent[i+1] - efficiency_error_percent[i]))
        interpolated_k = np.exp(
            np.log(conductivities[i])
            + fraction*(np.log(conductivities[i+1]) - np.log(conductivities[i]))
        )
        print(f"Interpolated 1% crossing (linear in log(k)): "
              f"(k, error) = ({interpolated_k:.12g} W/(m K), 1%)")
        ax.plot(interpolated_k, 1.0, marker="x", color="red",
                markersize=12, markeredgewidth=2, linestyle="none",
                label="Interpolated 1% crossing (log k)")
        ax.annotate(
            f"Interpolated in log(k)\n"
            f"k = {interpolated_k:.8g} W/(m K)\nerror = 1%",
            xy=(interpolated_k, 1.0), xytext=(-35, 110),
            textcoords="offset points", ha="right",
            arrowprops={"arrowstyle": "->", "color": "red"},
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "alpha": 0.9},
        )
    else:
        print("No sampled interval brackets the 1% threshold.")

    ax.set_xticks(conductivities)
    ax.set_xticklabels([f"{value:g}" for value in conductivities], rotation=45)
    ax.set_xlabel("Thermal conductivity, k (W/(m K)) — logarithmic scale")
    ax.set_ylabel(r"Efficiency error, $100|1-\eta_{num}/\eta_{analytical}|$ (%)")
    ax.set_title("Design 1: efficiency error versus thermal conductivity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    mp.show()
