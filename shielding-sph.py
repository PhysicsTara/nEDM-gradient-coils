import numpy as np
from mayavi import mlab
import trimesh
from bfieldtools.mesh_conductor import MeshConductor, StreamFunction
from bfieldtools.coil_optimize import optimize_streamfunctions
from bfieldtools.contour import scalar_contour
from bfieldtools.viz import plot_3d_current_loops, plot_data_on_vertices
from bfieldtools.utils import combine_meshes
from bfieldtools.sphtools import ylm, Wlm, Vlm
import pkg_resources


# Set unit, e.g. meter or millimeter.
# This doesn't matter, the problem is scale-invariant
scaling_factor = 1


# Load simple plane mesh that is centered on the origin
planemesh = trimesh.load(
    file_obj=pkg_resources.resource_filename(
        "bfieldtools", "example_meshes/10x10_plane_hires.obj"
    ),
    process=False,
)

planemesh.apply_scale(scaling_factor)

# Specify coil plane geometry
center_offset = np.array([0, 0, 0]) * scaling_factor
standoff = np.array([0, 5, 0]) * scaling_factor

# Create coil planes
coil_plus = trimesh.Trimesh(
    planemesh.vertices + center_offset + standoff, planemesh.faces, process=False
)

coil_minus = trimesh.Trimesh(
    planemesh.vertices + center_offset - standoff, planemesh.faces, process=False
)
coil_plus_r = trimesh.Trimesh(
    planemesh.vertices + center_offset + standoff, planemesh.faces, process=False
)

coil_minus_r = trimesh.Trimesh(
    planemesh.vertices + center_offset - standoff, planemesh.faces, process=False
)

coil_plus_r.apply_transform(([0, 1, 0 , 0], [-1, 0, 0 , 0], [0, 0, 1, 0], [0, 0, 0, 1]))
coil_minus_r.apply_transform(([0, 1, 0 , 0], [-1, 0, 0 , 0], [0, 0, 1, 0], [0, 0, 0, 1]))
joined_planes = combine_meshes((coil_plus, coil_minus, coil_plus_r, coil_minus_r))


# Create mesh class object
coil = MeshConductor(mesh_obj=joined_planes, fix_normals=True, basis_name="inner")

# Separate object for shield geometry
shieldmesh = trimesh.load(
    file_obj="C:\\Users\\waves\\OneDrive\\Documents\\inner-shielding-v3.stl",
    process=True,
)

shieldmesh.apply_scale(0.0044)

shield = MeshConductor(
    mesh_obj=shieldmesh, process=True, fix_normals=True, basis_name="vertex"
)
center = np.array([0, 0, 0]) * scaling_factor

sidelength = 3 * scaling_factor
n = 12
xx = np.linspace(-sidelength / 2, sidelength / 2, n)
yy = np.linspace(-sidelength / 2, sidelength / 2, n)
zz = np.linspace(-sidelength / 2, sidelength / 2, n)
X, Y, Z = np.meshgrid(xx, yy, zz, indexing="ij")

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

target_points = np.array([x, y, z]).T

# Turn cube into sphere by rejecting points "in the corners"
target_points = (
    target_points[np.linalg.norm(target_points, axis=1) < sidelength / 2] + center
)


# Plot coil, shield and target points

f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))

coil.plot_mesh(representation="surface", figure=f, opacity=0.5)
shield.plot_mesh(representation="surface", opacity=0.2, figure=f)
mlab.points3d(*target_points.T, color=(1, 1, 0))

f.scene.isometric_view()
f.scene.camera.zoom(1.1)

# The absolute target field amplitude is not of importance,
# and it is scaled to match the C matrix in the optimization function

def abel_Pi(r, l, m):
    x, y, z = r
    n= (l**2) + 3*l + m + 1
    Pi_x = np.array(
        [
            np.zeros_like(x),
            np.zeros_like(x),
            np.ones_like(x),
            y,
            np.zeros_like(x),
            (-1 / 2) * x,
            z,
            x,
            2 * x * y,
            2 * y * z,
            (-1 / 2) * x * y,
            -x * z,
            (-1 / 4) * (3 * x**2 + y**2 - 4 * z**2),
            2 * x * z,
            x**2 - y**2,
            3 * x**2 * y - y**3,
            6 * x * y * z,
            (-1 / 2) * (3 * x**2 * y + y**3 - 6 * y * z**2),
            (-3 / 2) * x * y * z,
            (3 / 8) * (x**3 + x * y**2 - 4 * x * z**2),
            (-1 / 4) * (9 * x**2 * z + 3 * y**2 * z - 4 * z**3),
            -(x**3) + 3 * x * z**2,
            3 * (x**2 * z - y**2 * z),
            x**3 - 3 * x * y**2,
        ]
    )
    Pi_y = np.array(
        [
            np.ones_like(y),  # 0 -1
            np.zeros_like(y),  # 0 0
            np.zeros_like(y),  # 0 1
            x,  # 1 -2
            z,  # 1 -1
            (-1 / 2) * y,  # 1 0
            np.zeros_like(y),  # 1 1
            (-1) * y,  # 1 2
            x**2 - y**2,  # 2 -3
            2 * x * z,  # 2 -2
            (-1 / 4) * (x**2 + 3 * y**2 - 4 * z**2),  # 2 -1
            -y * z,  # 2 0
            (-1 / 2) * x * y,  # 2 1
            -2 * y * z,  # 2 2
            -2 * x * y,  # 2 3
            x**3 - 3 * x * y**2,  # 3 -4
            3 * (x**2 * z - y**2 * z),  # 3 -3
            (-1 / 2) * (x**3 + 3 * x * y**2 - 6 * x * z**2),  # 3 -2
            (-1 / 4) * (3 * x**2 * z + 9 * y**2 * z - 4 * z**3),  # 3 -1
            (3 / 8) * (x**2 * y + y**3 - 4 * y * z**2),  # 3 0
            (-3 / 2) * (x * y * z),  # 3 1
            -3 * (y * z**2) + y**3,  # 3 2
            -6 * (x * y * z),  # 3 3
            -3 * x**2 * y + y**3,  # 3 4
        ]
    )
    Pi_z = np.array(
        [
            np.zeros_like(z),  # 0 -1
            np.ones_like(z),  # 0 0
            np.zeros_like(z),  # 0 1
            np.zeros_like(z),  # 1 -2
            y,  # 1 -1
            z,  # 1 0
            x,  # 1 1
            np.zeros_like(z),  # 1 2
            np.zeros_like(z),  # 2 -3
            2 * x * y,  # 2 -2
            2 * y * z,  # 2 -1
            z**2 - (1 / 2) * (x**2 + y**2),  # 2 0
            2 * x * z,  # 2 1
            x**2 - y**2,  # 2 2
            np.zeros_like(z),  # 2 3
            np.zeros_like(z),  # 3 -4
            3 * x**2 * y - y**3,  # 3 -3
            6 * x * y * z,  # 3 -2
            3 * y * z**2 - (3 / 4) * (x**2 * y + y**3),  # 3 -1
            z**3 - (3 / 2) * z * (x**2 + y**2),  # 3 0
            3 * x * z**2 - (3 / 4) * (x**3 + x * y**2),  # 3 1
            3 * (x**2 * z - y**2 * z),  # 3 2
            x**3 - 3 * x * y**2,  # 3 3
            np.zeros_like(z),  # 3 4
        ]
    )
    return np.array([Pi_x[n], Pi_y[n], Pi_z[n]])

target_field = np.zeros(target_points.shape)
i=0

#Target Field specifications
l=2
m=-2


for point in target_points:
    target_field[i] = abel_Pi(point, l, m)
    i+=1

target_abs_error = np.zeros_like(target_field)
target_abs_error[:, 0] += 0.005
target_abs_error[:, 1:3] += 0.005

target_spec = {
    "coupling": coil.B_coupling(target_points),
    "rel_error": 0,
    "abs_error": target_abs_error,
    "target": target_field,
}

import mosek

coil.s, coil.prob = optimize_streamfunctions(
    coil,
    [target_spec],
    objective="minimum_inductive_energy",
    solver="MOSEK",
    solver_opts={"mosek_params": {'MSK_IPAR_NUM_THREADS': 8}},
)

loops = scalar_contour(coil.mesh, coil.s.vert, N_contours=10)

f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
mlab.clf()

plot_3d_current_loops(loops, colors="auto", figure=f)

B_target = coil.B_coupling(target_points) @ coil.s

mlab.quiver3d(*target_points.T, *B_target.T, mode="arrow", scale_factor=0.5)

f.scene.isometric_view()
f.scene.camera.zoom(0.95)

