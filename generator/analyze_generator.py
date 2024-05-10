from gen_dataset import get_random_scene_def, get_random_darts, get_random_sensors_rotations
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import mitsuba as mi

def show_darts_distrib():
    NUM = 150000
    print(f"Generating {NUM} darts...")
    pt = []
    for k in range(NUM // 3):
        rd = get_random_darts(3, spread=0.02 if random.random()<0.5 else None)
        pt.extend([[v["x"],v["y"]] for v in rd.values()])
    pt= np.array(pt).T
    print("Done", pt.shape)
    
    mini = np.min(pt,axis=0)
    maxi = np.max(pt,axis=0)
    print(mini,maxi)
    #print(json.dumps(rsd,default=  lambda x : str(x)))
    nx,ny = 100,100
    x_bins = np.linspace(-0.3,0.3,nx+1)
    y_bins = np.linspace(-0.3,0.3,ny+1)
    density, _, _ = np.histogram2d(pt[0,:],pt[1,:],[x_bins,y_bins])

    density = density.reshape(nx, ny)
    plt.imshow(density, extent=(-0.3,0.3,-0.3,0.3),
            cmap=cm.hot)#, norm=LogNorm())
    plt.colorbar()
    #plt.scatter(pt[0,:],pt[1,:], s=1)

    plt.show()


def show_cameras_distributions():
    NUM = 5000
    def get_sensor_transform(rotx, roty, fov, radius, distance_factor):
        d =  radius*distance_factor / np.tan(np.deg2rad(fov*0.5))
        # tr = mi.ScalarTransform4f.rotate([0,1,0],roty).rotate([1,0,0],rotx).translate([0,0,d])

        # #tr =  mi.ScalarTransform4f.look_at(origin= tr.translation().numpy(), target=[0, 0, 0],  up=[0, 1, 0])
        # return tr.translation().numpy()
        theta = 2 * np.pi * ((rotx/360)+0.5)
        #phi = np.pi * ((roty/180)+0.5)
        phi = np.arccos(1 - 2 * ((roty/140)+0.5))
        x = np.sin(phi) * np.cos(theta) * d
        y = np.sin(phi) * np.sin(theta) * d
        z = np.cos(phi) * d
        return [y,z,-x]

    ra = 0.9
    rb = 1.3
    vals = np.array([get_sensor_transform(r[0],r[1],37.5,0.2255,f) for r,f in zip(get_random_sensors_rotations(NUM,-40,40),  np.random.uniform(ra,rb,NUM))])
    mini = np.min(vals,axis=0)
    maxi = np.max(vals,axis=0)
    print(mini,maxi)
    vals = vals.T


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = vals[0,:]
    ys = vals[1,:]
    zs = vals[2,:]
    ax.scatter(xs, zs, ys, marker='o',s=0.1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    # ax2 = fig.add_subplot(2,1,2)
    # ax2.scatter()
    plt.show()

if __name__ == "__main__":
    #show_darts_distrib()
    show_cameras_distributions()
