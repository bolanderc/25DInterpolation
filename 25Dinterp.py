# import aeropy.CST_lib as cst
import aeropy.CST_3D as cst
import aeropy.CST_3D.mesh_tools as meshtools
import panairwrapper
from aeropy.filehandling.vtk import generate_surface
import surface_interpolation as si
import time
import numpy as np
import matplotlib.pyplot as plt


aoa = 3.375

# GENERATE INTERPOLATED WING SURFACES
# import surface points from file
dp_wu_0 = np.genfromtxt('./points/wing_upper_0.csv', skip_header=1,
                         delimiter=',')
dp_wu_1 = np.genfromtxt('./points/wing_upper_1.csv', skip_header=1,
                         delimiter=',')
dp_wl_0 = np.genfromtxt('./points/wing_lower_0.csv', skip_header=1,
                         delimiter=',')
dp_wl_1 = np.genfromtxt('./points/wing_lower_1.csv', skip_header=1,
                         delimiter=',')

# Import edge points. The edges provided are just on the sides of the section.
# The leading and trailing edges are created by drawing a straightline between
# the fordmost and rearmost points on each side.
dp_wu_e0 = np.genfromtxt('./points/wing_intersection_u.csv', skip_header=1,
                         delimiter=',')
dp_wu_e1 = np.genfromtxt('./points/wing_edge_1.csv', skip_header=1,
                         delimiter=',')
dp_wu_e2 = np.genfromtxt('./points/wing_edge_2.csv', skip_header=1,
                         delimiter=',')
dp_wl_e0 = np.genfromtxt('./points/wing_intersection_l.csv', skip_header=1,
                         delimiter=',')
dp_wl_e1 = np.genfromtxt('./points/wing_edge_1_l.csv', skip_header=1,
                         delimiter=',')
dp_wl_e2 = np.genfromtxt('./points/wing_edge_2_l.csv', skip_header=1,
                         delimiter=',')
# Generate interpolation surfaces. These have the same interface
# and similar behavior as the CST equations. Instead of a calculating
# the surface points using an equation, multivariate interpolation is
# used over the points that were pro0vided. Note that this gen_wing_surface
# function only works for wing section with linear leading and trailing
# edges.
surf_wu_0 = si.gen_wing_surface(dp_wu_0, dp_wu_e0, dp_wu_e1)
surf_wu_1 = si.gen_wing_surface(dp_wu_1, dp_wu_e1, dp_wu_e2)
surf_wl_0 = si.gen_wing_surface(dp_wl_0, dp_wl_e0, dp_wl_e1)
surf_wl_1 = si.gen_wing_surface(dp_wl_1, dp_wl_e1, dp_wl_e2)


# GENERATE INTERPOLATED FUSELAGE SURFACES
# Import surface points from file. Upper and lower fuselage surfaces are
# combined into one surface.
dp_fu = np.genfromtxt('./points/fuselage_upper.csv', skip_header=1, delimiter=',')
dp_fl = np.genfromtxt('./points/fuselage_lower.csv', skip_header=1, delimiter=',')
x, y, z = dp_fu.T
x1, y1, z1 = dp_fl.T
x = np.concatenate((x, x1))
y = np.concatenate((y, y1))
z = np.concatenate((z, z1))
dp_f = np.array([x, y, z])
# Get boundary points. In this case, all four boundaries of the surface are
# provided.
dp_f_eu = np.genfromtxt('./points/boundary_fuse_u.csv', skip_header=1, delimiter=',')[:, :3].T
dp_f_el = np.genfromtxt('./points/boundary_fuse_l.csv', skip_header=1, delimiter=',')[:, :3].T
dp_f_ef = np.genfromtxt('./points/fuselage_front_edge.csv', skip_header=1, delimiter=',').T
dp_f_eb = np.genfromtxt('./points/fuselage_tail_edge.csv', skip_header=1, delimiter=',').T
#plt.plot(dp_f_el[0], dp_f_el[2])
#plt.plot(dp_f_eu[0], dp_f_eu[2])
#plt.show()
#dp_f_eu[1] = 0.0
#dp_f_eu = np.delete(dp_f_eu, -2, axis=1)

# convert to cylindrical coordinates so that the interpolation works better
bound_f_l = si.Spline1D(dp_f_el[0], dp_f_el[1], dp_f_el[2], smooth=0.)
bound_f_u = si.Spline1D(dp_f_eu[0], dp_f_eu[1], dp_f_eu[2], smooth=0.)
# create a function that converts the fuselage points into cylindrical coordinates
fuse_length = np.max(dp_f[0])
xr = np.linspace(0, 1, num=1000)
convert_to_cyl = si.CartToCyl((bound_f_l, bound_f_u), fuse_length)
aa = bound_f_u(xr)
#plt.plot(aa[0], aa[2])
#plt.plot(dp_f_eu[0], dp_f_eu[2])
#plt.show()

# convert the edge points to cylindrical
dp_f_eu_cyl = convert_to_cyl(dp_f_eu[0], dp_f_eu[1], dp_f_eu[2])
dp_f_eu_cyl[1][0] *= -1  # fix error in conversion due to the very front having zero radius
#plt.plot(dp_f_eu_cyl[0], dp_f_eu_cyl[2])
#plt.show()
dp_f_eu_cyl[1][:] = np.pi/2.
dp_f_el_cyl = convert_to_cyl(dp_f_el[0], dp_f_el[1], dp_f_el[2])
dp_f_el_cyl[1][:] = -np.pi/2.
dp_f_ef_cyl = convert_to_cyl(dp_f_ef[0], dp_f_ef[1], dp_f_ef[2])
dp_f_ef_cyl[1][1] *= -1  # fix error in conversion due to the very front having zero radius
dp_f_eb_cyl = convert_to_cyl(dp_f_eb[0], dp_f_eb[1], dp_f_eb[2])
dp_f_eb_cyl[1][1] *= -1

# spline the boundaries
bound_f_u_cyl = si.Spline1D(dp_f_eu_cyl[0], dp_f_eu_cyl[1], dp_f_eu_cyl[2])
bound_f_l_cyl = si.Spline1D(dp_f_el_cyl[0], dp_f_el_cyl[1], dp_f_el_cyl[2])
bound_f_f_cyl = si.Linear1D(dp_f_ef_cyl[0], dp_f_ef_cyl[1], dp_f_ef_cyl[2])
bound_f_b_cyl = si.Linear1D(dp_f_eb_cyl[0], dp_f_eb_cyl[1], dp_f_eb_cyl[2])
aa = bound_f_l_cyl(xr)
#plt.plot(aa[0], aa[2])
#plt.show()
# convert the surface points to cylindrical
dp_f_cyl = convert_to_cyl(dp_f[0], dp_f[1], dp_f[2])
#plt.plot(dp_f_cyl[0], dp_f_cyl[2])
#plt.show()
dp_f_cyl[1][-1] = -np.pi/2.
test = np.argwhere(np.isnan(dp_f_cyl))
dp_f_cyl[1][16469] = -np.pi/2.
# create the interpolation surface
surf_f = si.SurfInterp(dp_f_cyl, (bound_f_f_cyl, bound_f_b_cyl),
                       (bound_f_l_cyl, bound_f_u_cyl))


# GENERATE MESH
N_chord = 33
N_span_0 = 5
N_span_1 = 17
#N_span_2 = 17
N_wingtip = 6
N_nose = 23
N_tail = 23
N_circ = 14

# create the meshes in the parameter space
# cos_space_half provide half of a cosine distribution since we just
# want points cosined clustered towards the wing tip and no where else.
cos_space_half = meshtools.cosine_spacing(period=0.5, offset=0.5)
psi_wu1, eta_wu1 = meshtools.meshparameterspace((N_chord, N_span_0),
                                                psi_spacing='cosine',
                                                eta_spacing='linear')
psi_wu2, eta_wu2 = meshtools.meshparameterspace((N_chord, N_span_1),
                                                psi_spacing='cosine',
                                                eta_spacing='user',
                                                user_spacing=(None, cos_space_half))

# create the wing meshes
mesh_wu1 = surf_wu_0(psi_wu1, eta_wu1)
mesh_wu2 = surf_wu_1(psi_wu2, eta_wu2)

# same process now for the lower surface
psi_wl1, eta_wl1 = meshtools.meshparameterspace((N_chord, N_span_0),
                                                psi_spacing='cosine',
                                                eta_spacing='linear')
psi_wl2, eta_wl2 = meshtools.meshparameterspace((N_chord, N_span_1),
                                                psi_spacing='cosine',
                                                eta_spacing='user',
                                                user_spacing=(None, cos_space_half))

mesh_wl1 = surf_wl_0(psi_wl1, eta_wl1)
mesh_wl2 = surf_wl_1(psi_wl2, eta_wl2)
#print(mesh_wl2)

# fuselage mesh
# import the intersection points for the upper and lower surfaces of the
# fuselage as well as the intersection points for the wing and fuselage.
dp_tail_intersect = np.genfromtxt('./points/tail_intersection_points.csv',
                                  skip_header=1, delimiter=',')
dp_nose_intersect = np.genfromtxt('./points/nose_intersection_points.csv',
                                  skip_header=1, delimiter=',')
dp_fw_intersect_u = np.genfromtxt('./points/wing_intersection_u.csv',
                                  skip_header=1, delimiter=',')
dp_fw_intersect_l = np.genfromtxt('./points/wing_intersection_l.csv',
                                  skip_header=1, delimiter=',')
# spline the fuselage intersection points
bound_fw_u = si.Spline1D(dp_fw_intersect_u[:, 0],
                         dp_fw_intersect_u[:, 1],
                         dp_fw_intersect_u[:, 2])
bound_fw_l = si.Spline1D(dp_fw_intersect_l[:, 0],
                         dp_fw_intersect_l[:, 1],
                         dp_fw_intersect_l[:, 2])
bound_f_nose = si.Spline1D(dp_nose_intersect[:, 0],
                           dp_nose_intersect[:, 1],
                           dp_nose_intersect[:, 2])
bound_f_tail = si.Spline1D(dp_tail_intersect[:, 0],
                           dp_tail_intersect[:, 1],
                           dp_tail_intersect[:, 2])
#aa = bound_fw_l(xr)
#plt.plot(aa[0], aa[2])
#aa = bound_fw_u(xr)
#plt.plot(aa[0], aa[2])
#plt.show()

#psi_nose = np.linspace(0., 20.82/38.6736, 1000)
#x_s_l, t_s_l, r_s_l = surf_f(psi_nose[:, None], np.full_like(psi_nose, 0.5)[:, None])
#
#x_new, r_new = meshtools.coarsen_axi(x_s_l, r_s_l, .016, .5)
#
#spacing_tail_edge = (x_new-x_new[0])/(x_new[-1]-x_new[0])
#spacing_tail_edge[0] = 0.
#spacing_tail_edge[-1] = 1.
#N_nose = len(spacing_tail_edge)
# get the axial spacing along the inner boundaries of the fuselage networks
cos_space = meshtools.cosine_spacing()
# cos_space_half_2 provides half a cosine distribution with the points
# clustered to the other side so we can cluster points towards the
# nose of the fuselage.
cos_space_half_2 = meshtools.cosine_spacing(period=0.5, offset=0.)
spacing_fw_edge = cos_space(0., 1., N_chord)
spacing_nose_edge = cos_space_half_2(0., 1., N_nose)
spacing_tail_edge = cos_space_half(0., 1., N_tail)

# here I get a little tricky. I generate a really fine spacing for the
# tail and then coarsen it using the same coarsening algorithm used in
# the AXIE case.
#psi_tail = np.linspace(34.0745/38.6736, 1., 1000)
#x_s_l, t_s_l, r_s_l = surf_f(psi_tail[:, None], np.full_like(psi_tail, 0.5)[:, None])
#
#x_new, r_new = meshtools.coarsen_axi(x_s_l, r_s_l, .016, .5)
#
#spacing_tail_edge = (x_new-x_new[0])/(x_new[-1]-x_new[0])
#spacing_tail_edge[0] = 0.
#spacing_tail_edge[-1] = 1.
#N_tail = len(spacing_tail_edge)

# get the boundary points for the inner boundaries of the fuselage networks
dp_fw_intersect_u = np.array(bound_fw_u(spacing_fw_edge)).T
dp_fw_intersect_l = np.array(bound_fw_l(spacing_fw_edge)).T
dp_nose_intersect = np.array(bound_f_nose(spacing_nose_edge)).T
dp_tail_intersect = np.array(bound_f_tail(spacing_tail_edge)).T
#print(dp_tail_intersect)
#plt.plot(dp_nose_intersect[:, 0], dp_nose_intersect[:, 2])
#plt.show()

dp_intersect_u = np.concatenate((dp_nose_intersect,
                                 dp_fw_intersect_u[1:-1],
                                 dp_tail_intersect)).T
#print(dp_intersect_u)
dp_intersect_l = np.concatenate((dp_nose_intersect,
                                 dp_fw_intersect_l[1:-1],
                                 dp_tail_intersect)).T

# convert the points to cylindrical
dp_intersect_u_cyl = convert_to_cyl(dp_intersect_u[0],
                                    dp_intersect_u[1],
                                    dp_intersect_u[2])
dp_intersect_l_cyl = convert_to_cyl(dp_intersect_l[0],
                                    dp_intersect_l[1],
                                    dp_intersect_l[2])


# find corresponding points in parameter space
dp_intersect_u_p = surf_f.inverse(dp_intersect_u_cyl[0],
                                  dp_intersect_u_cyl[1])
dp_intersect_u_p[1][0] = 0.5
dp_intersect_u_p[1][-1] = 0.5
dp_intersect_l_p = surf_f.inverse(dp_intersect_l_cyl[0],
                                  dp_intersect_l_cyl[1])
dp_intersect_l_p[1][0] = 0.5
dp_intersect_l_p[1][-1] = 0.5
# generate the meshes in the parameter space
eta_bound_u = np.array(dp_intersect_u_p).T
N_length_u = len(eta_bound_u)
eta_bound_u[0, 0] = 0.
eta_bound_u[-1, 0] = 1.
x_grid_u, t_grid_u = meshtools.meshparameterspace((N_length_u, N_circ),
                                                  psi_spacing='uniform',
                                                  eta_spacing='user',
                                                  eta_limits=(eta_bound_u, None),
                                                  user_spacing=(None, cos_space_half_2))
eta_bound_l = np.array(dp_intersect_l_p).T
N_length_l = len(eta_bound_l)
eta_bound_l[0, 0] = 0.
eta_bound_l[-1, 0] = 1.
x_grid_l, t_grid_l = meshtools.meshparameterspace((N_length_l, N_circ),
                                                  psi_spacing='uniform',
                                                  eta_spacing='user',
                                                  eta_limits=(None, eta_bound_l),
                                                  user_spacing=(None, cos_space_half))

# calculate mesh in x, theta, r
x_new_u, t_new_u, r_new_u = surf_f(x_grid_u, t_grid_u)
deltas = (r_new_u[6:1:-1, 13] - r_new_u[5:0:-1, 13])
for i in range(len(deltas)):
    r_new_u[5 - i, 12] = r_new_u[6 - i, 12] - deltas[i]
for i in range(2):
    r_new_u[2 - i, 11] = r_new_u[3 - i, 11] - deltas[i + 3]
r_new_u[1, 8:11] = r_new_u[2, 8:11] - deltas[-1]
deltas = (r_new_u[70:75, 13] - r_new_u[71:76, 13])
for i in range(len(deltas)):
    r_new_u[71 + i, 12] = r_new_u[70 + i, 12] - deltas[i]
r_new_u[74, 11] = r_new_u[73, 11] - deltas[-2]

r_new_u[75, :] = r_new_u[1, :]
#print(r_new_u)
# calculate mesh in x, y, z
x_new_u, y_new_u, z_new_u = convert_to_cyl.inverse(x_new_u, t_new_u, r_new_u)
# make sure intersection points are exact
x_new_u[:, 0] = dp_intersect_u[0]
y_new_u[:, 0] = dp_intersect_u[1]
z_new_u[:, 0] = dp_intersect_u[2]

mesh_fu = (x_new_u, y_new_u, z_new_u)

# repeat for lower fuselage network
x_new_l, t_new_l, r_new_l = surf_f(x_grid_l, t_grid_l)
r_new_l[1, :] = r_new_u[1, :]
r_new_l[72:, :] = r_new_l[4::-1, :]
x_new_l, y_new_l, z_new_l = convert_to_cyl.inverse(x_new_l, t_new_l, r_new_l)
x_new_l[:, -1] = dp_intersect_l[0]
y_new_l[:, -1] = dp_intersect_l[1]
z_new_l[:, -1] = dp_intersect_l[2]

mesh_fl = (x_new_l, y_new_l, z_new_l)

# format as networks
network_wu1 = np.dstack(mesh_wu1)
network_wu2 = np.dstack(mesh_wu2)
network_wl1 = np.dstack(mesh_wl1)
network_wl2 = np.dstack(mesh_wl2)
network_fu = np.dstack(mesh_fu)
network_fu = np.fliplr(np.rot90(network_fu, axes=(1, 0)))
network_fl = np.dstack(mesh_fl)
network_fl = np.fliplr(np.rot90(network_fl, axes=(1, 0)))

# make sure leading and trailing edge points of the upper and lower
# networks of the wing match
network_wu1[-1, :, :] = network_wl1[-1, :, :]
network_wu2[-1, :, :] = network_wl2[-1, :, :]
network_wu1[0, :, :] = network_wl1[0, :, :]
network_wu2[0, :, :] = network_wl2[0, :, :]

# make sure points along intersection of wing and fuselage match
# for both the wing and fuselage networks
network_wu1[:, 0, :] = network_fu[0, N_nose-1:(-N_tail+1), :]
network_wl1[:, 0, :] = network_fl[-1, N_nose-1:(-N_tail+1), :]

# generate  wing cap
network_wingcap = np.zeros((N_chord, N_wingtip, 3))
network_wingcap[:, 0, :] = network_wu2[:, -1, :]
network_wingcap[:, -1, :] = network_wl2[:, -1, :]
for i in range(0, N_chord):
    dx = [network_wingcap[i, 0, 0], network_wingcap[i, -1, 0]]
    dy = [network_wingcap[i, 0, 1], network_wingcap[i, -1, 1]]
    dz = [network_wingcap[i, 0, 2], network_wingcap[i, -1, 2]]
    interp = si.Linear1D(dx, dy, dz)
    x, y, z = interp(cos_space(0., 1., N_wingtip))
    network_wingcap[i, :, 0] = x
    network_wingcap[i, :, 1] = y
    network_wingcap[i, :, 2] = z

# generate tail cap
network_fusecap = np.zeros((2*N_circ-1, 2, 3))
network_fusecap[:, 1, :] = np.concatenate((network_fl[:-1, -1],
                                           network_fu[:, -1]))
network_fusecap[:, 0, 0] = network_fusecap[:, 1, 0]
network_fusecap[:, 0, 2] = np.linspace(network_fusecap[0, 1, 2],
                                       network_fusecap[-1, 1, 2],
                                       2*N_circ-1)

# calculate wakes
wing_trailing_edge = np.concatenate((network_wu1[-1, :-1, :],
                                     network_wu2[-1, :-1, :]))

fuselage_wake_boundary = network_fu[0, -N_tail:]
inner_endpoint = np.copy(fuselage_wake_boundary[-1])
n_wake_streamwise = len(fuselage_wake_boundary)+1
l_w = 0.5

body_wake_l = meshtools.generate_wake(network_fl[:, -1], inner_endpoint[0]+l_w,
                                      2, aoa)
body_wake_u = meshtools.generate_wake(network_fu[:, -1], inner_endpoint[0]+l_w,
                                      2, aoa)

wing_wake = meshtools.generate_wake(wing_trailing_edge, inner_endpoint[0]+l_w,
                                    n_wake_streamwise, aoa, user_spacing=cos_space_half)


wingbody_wake = np.zeros((n_wake_streamwise, 2, 3))
wingbody_wake[:-1, 0] = fuselage_wake_boundary
wingbody_wake[-1, 0] = body_wake_u[-1, 0]
wingbody_wake[:, 1] = wing_wake[:, 0]

# Generating vtk files
generate_surface(network_wu1, "wingupper1")
generate_surface(network_wu2, "wingupper2")
generate_surface(network_wl1, "winglower1")
generate_surface(network_wl2, "winglower2")
generate_surface(network_fu, "fuselageupper")
generate_surface(network_fl, "fuselagelower")
generate_surface(network_wingcap, "wingcap")
generate_surface(network_fusecap, "fusecap")
generate_surface(wing_wake, "wake")
generate_surface(body_wake_l, "body_wake_l")
generate_surface(body_wake_u, "body_wake_u")
generate_surface(wingbody_wake, "wingbody_wake")

N_cut = 6
# run in Panair
gamma = 1.4
MACH = 1.6
panair = panairwrapper.PanairWrapper('wingbody')
panair.set_aero_state(MACH, aoa)
panair.set_symmetry(1, 0)
panair.set_output_all(True)
panair.set_reference_data(32.8*2, 38.7, 38.7)
panair.add_network("wing_u1", np.flipud(network_wu1), 11.)
panair.add_network("wing_u2", np.flipud(network_wu2), 11.)
panair.add_network("wing_l1", network_wl1, 11.)
panair.add_network("wing_l2", network_wl2, 11.)
panair.add_network("fuselage_u_1", np.flipud(network_fu), 11.)
panair.add_network("fuselage_l_1", np.flipud(network_fl), 11.)
panair.add_network("wing_cap", np.flipud(network_wingcap), 11.)
#panair.add_network("fuse_cap", network_fusecap, 5.)
panair.add_network("wake", wing_wake, 18.)
panair.add_network("body_wake_l", body_wake_l, 18.)
panair.add_network("body_wake_u", body_wake_u, 18.)
panair.add_network("wingbody_wake", wingbody_wake, 20.)

panair.set_sensor(MACH, aoa, 5, 32.92, 2.)
t0 = time.time()
results = panair.run(overwrite=True)
t1 = time.time()
print(t1 - t0)
offbody_data = results.get_offbody_data()
distance_along_sensor = offbody_data[:, 2]
dp_over_p = 0.5*gamma*MACH**2*offbody_data[:, -2]
nearfield_sig = np.array([distance_along_sensor, dp_over_p]).T
np.savetxt('bump_NF', nearfield_sig)
plt.plot(nearfield_sig[:, 0], nearfield_sig[:, 1])
plt.title("nearfield signature")
plt.show()
results.write_vtk()
#F_M = results.get_forces_and_moments()
#print(F_M)
