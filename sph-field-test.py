import numpy as np
from mayavi import mlab
from bfieldtools.sphtools import ylm, Wlm, Vlm

scaling_factor = 1
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

target_field = np.zeros(target_points.shape)

i=0
for point in target_points:
    a=point[0]
    b=point[1]
    c=point[2],
    radius=np.sqrt(((point[0])**2)+((point[1])**2)+((point[2])**2))
    theta=np.arccos(c/radius)
    
    if(a*b>0):
        if(a>0):
            phi=np.arctan(b/a)
        else:
            phi=np.arctan(b/a)+np.pi
    else:
        if(a>0):
            phi=np.arctan(b/a)+2*np.pi
        else:
            phi=np.arctan(b/a)+np.pi

    sph=ylm(3, 2, theta, phi) #ylm(l , m, theta, phi)
    r_hat=(point[0]/radius, point[1]/radius, point[2]/radius)
    target_field[i]=sph*r_hat
    #target_field[i]=Wlm(1, 0, theta, phi)
    i+=1
f = mlab.figure(None, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5), size=(800, 800))
mlab.quiver3d(*target_points.T, *target_field.T, mode="arrow", scale_factor=0.75)

f.scene.isometric_view()
f.scene.camera.zoom(0.95)
