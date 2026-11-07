import numpy as np
import matplotlib.pyplot as mp
from scipy.sparse import csr_array
import time

## Declare constants
D_pin = 1.2e-3
alpha = 3
num_h = 60
num_r = 10
k = 0.621          # Rigid 10K Resin V1

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

## Create NxN matrix
M = np.zeros([num_nodes, num_nodes])

## Create Nx1 Array
b = np.zeros([num_nodes,1])

## Area functions
radius_list = np.linspace(0, D_pin/2, num_r)
height_list = np.linspace(0, H_pin, num_h)

A_l = lambda i: 2*np.pi*(radius_list[i] - dr/2)*dh
A_r = lambda i: 2*np.pi*(radius_list[i] + dr/2)*dh
A_h = lambda i: 2*np.pi*radius_list[i]*dr

A_left_edge_h = np.pi*((dr/2)**2)

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
error = 1
old_temp = np.full((num_nodes, 1), 50.0)
start = time.time()

while error > 1e-8:
    ## Solve System of Linear Equations (x_(i+1) = M*x_(i) + b)
    new_temp = M_sparse@old_temp + b

    error = np.max(np.abs(new_temp - old_temp))

    old_temp = new_temp

print(f"Solver time: {time.time() - start:.3f} s")

temp_soln = new_temp

# Global energy balance: trapezoidal average over the outer side surface.
surface_temp = temp_soln.reshape(num_h, num_r)[:, -1]
height_weights = np.full(num_h, dh)
height_weights[[0, -1]] /= 2
Q_in = q_flux*np.pi*(D_pin/2)**2
Q_out = h_conv*2*np.pi*(D_pin/2)*np.dot(height_weights, surface_temp - T_inf)
avg_surface_temp = np.dot(height_weights, surface_temp)/H_pin
expected_avg_surface_temp = T_inf + Q_in/(h_conv*2*np.pi*(D_pin/2)*H_pin)
relative_heat_imbalance = abs(Q_in - Q_out)/Q_in

print(f"Average side temperature: {avg_surface_temp:.8f} C")
print(f"Expected from energy balance: {expected_avg_surface_temp:.8f} C")
print(f"Heat input: {Q_in:.10f} W; convection loss: {Q_out:.10f} W")
print(f"Relative heat imbalance: {relative_heat_imbalance:.3e}")
print("Energy balance:", "PASS" if relative_heat_imbalance < 1e-5 else "CHECK")

# Original rows run from tip to base; flip only for the contour plot.
temp_map1 = temp_soln.reshape(num_h, num_r)
temp_map = np.flipud(temp_map1)

radius_list_nondim = radius_list/(D_pin/2)
height_list_nondim = height_list/(H_pin)

mp.contourf(radius_list_nondim, height_list_nondim, temp_map, levels=20)
cbar = mp.colorbar()
cbar.set_label(r'Temperature ($\degree C$)', size=12)
mp.ylabel(r'$h/H_{Pin}$', size=12)
mp.xlabel(r'$r/R_{Pin}$', size=12)

# Centerline comparison in original tip-to-base node order.
T_centerline = temp_map1[:, 0]
z_from_base = H_pin - height_list
m_fin = np.sqrt(4*h_conv/(k*D_pin))

# Analytical 1D fin: -k*dT/dz = q_flux at the base, dT/dz = 0 at the tip.
T_analytical = T_inf + (q_flux/(k*m_fin))*np.cosh(
    m_fin*(H_pin - z_from_base)
)/np.sinh(m_fin*H_pin)

# Assignment's relative RMS error, using temperatures in Celsius as specified.
relative_rmse = np.sqrt(np.mean((1 - T_centerline/T_analytical)**2))
print(f"Centerline relative RMSE (assignment definition): {relative_rmse:.8f} "
      f"({100*relative_rmse:.4f}%)")

mp.figure()
mp.plot(z_from_base/H_pin, T_centerline, "o-", label="Numerical centerline")
mp.plot(z_from_base/H_pin, T_analytical, "--", label="Analytical adiabatic tip")
mp.xlabel(r"$z/H_{Pin}$ (base = 0, tip = 1)")
mp.ylabel(r"Temperature ($\degree C$)")
mp.title(f"Design {1 if alpha == 0.75 else 2}: centerline comparison\n"
         f"Relative RMSE = {relative_rmse:.6f} ({100*relative_rmse:.3f}%)")
mp.legend()
mp.grid(True, alpha=0.3)
mp.tight_layout()

# Compare centerline and surface temperatures at each height.
# Their difference is a function of height, not a radial temperature profile.
T_surface = temp_map1[:, -1]
radial_difference = T_centerline - T_surface
Bi = h_conv*(D_pin/2)/(2*k)
max_difference_index = np.argmax(np.abs(radial_difference))

print(f"Radial Biot number (h*R/(2*k)): {Bi:.6f}")
print(f"Maximum absolute centerline-to-surface difference: "
      f"{abs(radial_difference[max_difference_index]):.6f} C "
      f"at z/H = {z_from_base[max_difference_index]/H_pin:.4f}")

mp.figure()
mp.plot(z_from_base/H_pin, radial_difference, "o-")
mp.xlabel(r"$z/H_{Pin}$ (base = 0, tip = 1)")
mp.ylabel(r"$T_{center}-T_{surface}$ ($\degree C$)")
mp.title(f"Design 2: radial temperature difference (Bi = {Bi:.3f})")
mp.grid(True, alpha=0.3)
mp.tight_layout()

# Fin efficiencies: numerical reference is the area-averaged base temperature.
r_inner = np.maximum(0, radius_list - dr/2)
r_outer = np.minimum(D_pin/2, radius_list + dr/2)
base_areas = np.pi*(r_outer**2 - r_inner**2)
base_temp = np.dot(base_areas, temp_map1[-1, :])/base_areas.sum()
side_area = 2*np.pi*(D_pin/2)*H_pin
eta_numerical = Q_out/(h_conv*side_area*(base_temp - T_inf))
eta_analytical = np.tanh(m_fin*H_pin)/(m_fin*H_pin)
print(f"Numerical fin efficiency: {eta_numerical:.8f} ({100*eta_numerical:.4f}%)")
print(f"Analytical fin efficiency: {eta_analytical:.8f} ({100*eta_analytical:.4f}%)")

# Standalone 2-by-2 table: method and efficiency in each row.
fig, ax = mp.subplots(figsize=(6, 2.5))
ax.axis("off")
efficiency_table = ax.table(
    cellText=[
        ["Numerical", f"{eta_numerical:.6f} ({100*eta_numerical:.3f}%)"],
        ["Analytical", f"{eta_analytical:.6f} ({100*eta_analytical:.3f}%)"],
    ],
    cellLoc="center", loc="center", colWidths=[0.4, 0.6],
)
efficiency_table.auto_set_font_size(False)
efficiency_table.set_fontsize(12)
efficiency_table.scale(1, 2)
ax.set_title("Design 2: fin efficiency")
fig.text(0.5, 0.08, "Numerical reference: area-averaged base temperature.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.15, 1, 1))

mp.show()
