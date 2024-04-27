import subprocess

from obj_tools import *
import numpy as np
import os

def edit_flight_obj(filepath, name, out=None):
    pos, norms, uvs, faces = loadObjFile(filepath)

    for fpi,fp in enumerate(faces):
        for fi,f in enumerate(fp):
            vi = f[0]
            faces[fpi][fi][1] = vi

    verts = np.array(pos)
    mini = np.min(pos, axis = 0)
    maxi = np.max(pos, axis = 0)
    thickness_orig = maxi[2]-mini[2]
    thickness = 0.0001

    h = maxi[1]-mini[1]
    cx = (maxi[0]+mini[0])*0.5

    new_uvs = []
    for i,p in enumerate(verts):
        #scale uv from height and center on u
        v = (p[1]-mini[1]) / h
        u = ((p[0]-cx)/ h) + 0.5

        pos[i][2] = (pos[i][2]-mini[2]) / thickness_orig
        if(pos[i][2]>0.5):
             u = 1.0 - u

        pos[i][2] = pos[i][2]*thickness - thickness * 0.5
        new_uvs.append([u,v])

    out = filepath if out is None else out
    saveObjFile(out, name, pos, norms, new_uvs, faces)

    mini = np.min(pos, axis = 0)
    maxi = np.max(pos, axis = 0)
    return mini, maxi

    # stats = np.array(new_uvs)
    # print(np.min(stats,axis=0), np.max(stats,axis=0))

    # verts = np.array(pos)
    # print(np.min(verts,axis=0), np.max(verts,axis=0))

def edit_lathe_obj(filepath, name, out=None):
    pos, norms, uvs, faces = loadObjFile(filepath)
    # print("pos:", len(pos))
    # print("norms:", len(norms))
    # print("uvs:", len(uvs))
    # print("faces:", len(faces))
    # return
    for fpi,fp in enumerate(faces):
        for fi,f in enumerate(fp):
            vi = f[0]
            faces[fpi][fi][1] = vi

    verts = np.array(pos)
    mini = np.min(pos, axis = 0)
    maxi = np.max(pos, axis = 0)
    thickness_orig = maxi[2]-mini[2]
    thickness = 0.0001

    h = maxi[1]-mini[1]

    def unit_vector(vector):
        """ Returns the unit vector of the vector"""
        return vector / np.linalg.norm(vector)

    def angle(vector1, vector2):
        """ Returns the angle in radians between given vectors"""
        v1_u = unit_vector(vector1)
        v2_u = unit_vector(vector2)
        minor = np.linalg.det(
            np.stack((v1_u[-2:], v2_u[-2:]))
        )
        if minor == 0:
            raise NotImplementedError('Too odd vectors =(', v1_u, v2_u)
        return np.sign(minor) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    from numpy import arctan, pi, signbit
    from numpy.linalg import norm


    def angle_btw(v1, v2):
        nrm = norm(v1)
        if(nrm<0.000001):
            return 0
        u1 = v1 / nrm
        nrm = norm(v2)
        if(nrm<0.000001):
            return 0        
        u2 = v2 / nrm

        y = u1 - u2
        x = u1 + u2
        nrm = norm(x)
        if(nrm<0.000001):
            return 0
        a0 = 2 * arctan(norm(y) / nrm)

        if (not signbit(a0)) or signbit(pi - a0):
            return a0
        elif signbit(a0):
            return 0.0
        else:
            return pi

    new_uvs = []
    for i,p in enumerate(verts):
        #scale uv from height and center on u
        v = (p[1]-mini[1]) / h

        d = np.array([p[0],p[2]])
        u = angle_btw(d, np.array([0,1]))/np.pi

        #pos[i][2] = (pos[i][2]-mini[2]) / thickness_orig
        # if(pos[i][2]>0.5):
        #      u = 1.0 - u

        # pos[i][2] = pos[i][2]*thickness - thickness * 0.5
        new_uvs.append([u,v])

    out = filepath if out is None else out
    saveObjFile(out, name, pos, norms, new_uvs, faces)

    mini = np.min(pos, axis = 0)
    maxi = np.max(pos, axis = 0)
    return mini, maxi

    # stats = np.array(new_uvs)
    # print(np.min(stats,axis=0), np.max(stats,axis=0))

    # verts = np.array(pos)
    # print(np.min(verts,axis=0), np.max(verts,axis=0))

if __name__ == "__main__":
    import os, json

    exe  = os.path.join(os.getenv("BLENDER_DIR",""),'blender.exe')
    subprocess.call([exe,"--background", r"3D\Darts\darts.blend", "--python", "blender.py"])

    metadata = {}

    for col in ["FLIGHTS","BARRELS","TIPS","SHAFTS"]:
        dir = f"./tmp/{col}/"
        func = edit_flight_obj if col == "FLIGHTS" else edit_lathe_obj
        files = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
        for f in files:
            mini, maxi = func(f,os.path.basename(f).split('.')[0])#, "tst.obj")
            name = os.path.basename(f)
            metadata[name] = {"min":list(mini), "max":list(maxi)}
    
    with(open("./tmp/metadata.json","w") as f):
        json.dump(metadata,f, indent=4)