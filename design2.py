import numpy as np
import matplotlib.pyplot as mp
from scipy.sparse import csr_array
import time

## Declare constants
D_pin = 1.2e-3
alpha = 3       # Design 1: 0.75, Design 2: 3
num_h = 60
num_r = 10
k = 0.621           # Rigid 10K Resin V1

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

# Reshape node temperatures and place the base at the bottom of the plot.
temp_map = np.flipud(temp_soln.reshape(num_h, num_r))

radius_list_nondim = radius_list/(D_pin/2)
height_list_nondim = height_list/(H_pin)

mp.contourf(radius_list_nondim, height_list_nondim, temp_map, levels=20)
cbar = mp.colorbar()
cbar.set_label(r'Temperature ($\degree C$)', size=12)
mp.ylabel(r'$h/H_{Pin}$', size=12)
mp.xlabel(r'$r/R_{Pin}$', size=12)

mp.show()