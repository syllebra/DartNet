from gen_dataset import get_random_scene_def, get_random_darts
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

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

if __name__ == "__main__":
    show_darts_distrib()
