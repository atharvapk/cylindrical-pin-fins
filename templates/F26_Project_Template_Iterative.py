## Code Template courtesy of J. Gutierrez Plascencia, UC Davis
import numpy as np
import matplotlib.pyplot as mp
from scipy.sparse import csr_array
import time

## Declare constants
D_pin = 1.2e-3
alpha = 1
num_h = 50
num_r = 20
k = 50

## Discretization of computational domain
H_pin = alpha*D_pin
dh = H_pin/(num_h-1)
dr = ((D_pin/2)/(num_r-1))

num_nodes = num_h*num_r

## Create num_h x num_r Array - Node Index Holders
neighbor_map = np.zeros([num_h, num_r])

## Fill domain array with node indexes
count = 0
for h in range(num_h):
    for r in range(num_r):
        neighbor_map[h,r] = count
        count += 1

## Create NxN matrix
M = np.zeros([num_nodes, num_nodes])

## Create Nx1 Array
initial_guess = np.zeros([num_nodes,1]) + 50
b = np.zeros([num_nodes,1])

## Area functions
radius_list = np.linspace(0, D_pin/2, num_r)
height_list = np.linspace(0, H_pin, num_h)

A_l = lambda i: 2*np.pi*(radius_list[i] - dr/2)*dh
A_r = lambda i: 2*np.pi*(radius_list[i] + dr/2)*dh
A_h = lambda i: 2*np.pi*radius_list[i]*dr

A_left_edge_h = np.pi*((dr/2)**2)
A_right_edge_r = 2*np.pi*(D_pin/2)*dh

## Build Coefficient Matrix (M)

## Constant Temperature Top Edge
h = 0
for r in range(num_r):
    center_index = neighbor_map[h,r]
    b[int(center_index)] = -25

## Constant Temperature Bottom Edge
h = num_h-1
for r in range(num_r):
    center_index = neighbor_map[h,r]
    b[int(center_index)] = 150

## Constant Temperature Right Edge
r = num_r-1
for h in range(1,num_h-1):
    center_index = neighbor_map[h,r]
    b[int(center_index)] = 75

## Symmetric Left Edge
r = 0
for h in range(1, num_h-1):
    center_index = neighbor_map[h,r]
    top_neigh_index, bottom_neigh_index = neighbor_map[h+1,r], neighbor_map[h-1,r]
    lateral_neigh_index = neighbor_map[h,r+1]

    den = (((k*(A_left_edge_h/2))/dh) + ((k*(A_left_edge_h/2))/dh) + (k*A_r(r)/dr))
    M[int(center_index),int(top_neigh_index)] = ((k*(A_left_edge_h/2))/dh)/den
    M[int(center_index),int(bottom_neigh_index)] = ((k*(A_left_edge_h/2))/dh)/den
    M[int(center_index),int(lateral_neigh_index)] = (k*A_r(r)/dr)/den

## Inner Nodes
for h in range(1, num_h-1):
    for r in range(1, num_r-1):
        center_index = neighbor_map[h, r]
        top_neigh_index, bottom_neigh_index = neighbor_map[h+1, r], neighbor_map[h-1, r]
        right_neigh_index, left_neigh_index = neighbor_map[h, r+1], neighbor_map[h, r-1]

        den = ((k*A_h(r)/dh) + (k*A_h(r)/dh) + (k*A_r(r)/dr) + (k*A_l(r)/dr))
        M[int(center_index),int(top_neigh_index)] = (k*A_h(r)/dh)/den
        M[int(center_index),int(bottom_neigh_index)] = (k*A_h(r)/dh)/den
        M[int(center_index),int(right_neigh_index)] = (k*A_r(r)/dr)/den
        M[int(center_index),int(left_neigh_index)] = (k*A_l(r)/dr)/den

# row_scaling =  [1/np.linalg.norm(M[row,:],ord=np.inf) for row in range(num_nodes)]
row_scaling = 1/np.linalg.norm(M,axis=1,ord=np.inf)
D1 = np.diag(row_scaling)
M_eq = np.dot(D1,M)
b_eq = np.dot(D1,b)

M_sparse = csr_array(M)
error = 1
old_temp = initial_guess.copy()
start = time.time()

while error > 1e-8:
    ## Solve System of Linear Equations (x_(i+1) = M*x_(i) + b)
    new_temp = M_sparse@old_temp + b
    
    error = max(abs(new_temp - old_temp))
    print(error)
 
    old_temp = new_temp.copy()

print(time.time() - start)

temp_soln = new_temp

## Create 2D Temperature Map (temp_map)
temp_map1 = np.zeros([num_h, num_r])

for index in range(num_nodes):
    row, col = np.where(neighbor_map == index)

    temp_map1[row, col] = temp_soln[index]

temp_map = np.flipud(temp_map1)

radius_list_nondim = radius_list/(D_pin/2)
height_list_nondim = height_list/(H_pin)

mp.contourf(radius_list_nondim, height_list_nondim, temp_map, levels=20)
cbar = mp.colorbar()
cbar.set_label(r'Temperature ($\degree C$)', size=12)
mp.ylabel(r'$h/H_{Pin}$', size=12)
mp.xlabel(r'$r/R_{Pin}$', size=12)

mp.show()