import re
import numpy as np

def loadObjFile(filename):
    pos = []
    norms = []
    uvs = []
    faces = []

    fin = open(filename, 'r')
    for line in fin:
        fields = line.split()
        if len(fields) == 0: #Blank line
            continue
        if fields[0][0] in ['#', '\0', ' '] or len(fields[0]) == 0:
            continue
        if fields[0] == "v":
            coords = [float(i) for i in fields[1:4]]
            pos.append([coords[0], coords[1], coords[2]])
        if fields[0] == "vn":
            coords = [float(i) for i in fields[1:4]]
            norms.append([coords[0], coords[1], coords[2]])
        if fields[0] == "vt":
            coords = [float(i) for i in fields[1:3]]
            uvs.append([coords[0], coords[1]])
        if fields[0] == "f":
            #Indices are numbered starting at 1 (so need to subtract that off)
            p = [[int(tok)-1 for tok in re.split("/",s)]  for s in fields[1:]]
            # indices = [[0])-1 for s in fields[1:]]
            # verts = [pos[i] for i in indices]
            faces.append(p)
    fin.close()

    # print("pos:\t",len(pos))
    # print("norms:\t",len(norms))
    # print("uvs:\t",len(uvs))
    # print("faces:\t",len(faces))

    return pos, norms, uvs, faces

def saveObjFile(filename, name, pos, norms, uvs, faces):
    file = open(filename, 'wt')
    file.write(f"o {name}\n")
    for p in pos:
        file.write(f"v {p[0]} {p[1]} {p[2]}\n")
    for n in norms:
        file.write(f"vn {n[0]} {n[1]} {n[2]}\n")
    for t in uvs:
        file.write(f"vt {t[0]} {t[1]}\n")
    for f in faces:
        line = "f"
        for fp in f:
            line += ' ' + '/'.join([str(v+1) for v in fp])
        file.write(line+"\n")
    file.close()

    return pos, norms, uvs, faces
