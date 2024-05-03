
import os
import random
import shutil
from tqdm import tqdm
import pathlib
import json

def get_data_file(image_file):
    file= os.path.splitext(image_file)[0]+".json"
    return file if os.path.isfile(file) else None

def move_sample(image_file, dest_dir):
    data_file = get_data_file(image_file)
    if(data_file is None):
        return
    shutil.move(image_file,dest_dir)
    shutil.move(data_file,dest_dir)

def reorganize_images_data(directory, validation_ratio, test_ratio):
    files =  [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and ".json" not in f and os.path.splitext(f)[1].lower() in [".jpg",".png"]]
    print(len(files))

    random.shuffle(files)

    os.makedirs(os.path.join(directory, "images"),exist_ok=True)
    num_test = int(len(files)*test_ratio)
    if(num_test>0):
        os.makedirs(os.path.join(directory, "images","test"),exist_ok=True)
    num_val = int(len(files)*validation_ratio)
    if(num_val>0):
        os.makedirs(os.path.join(directory, "images","val"),exist_ok=True)

    os.makedirs(os.path.join(directory, "images","train"),exist_ok=True)

    for i,f in tqdm(enumerate(files)):
        if(i<num_val):
            dst = "val"
        elif(i<num_val+num_test):
            dst = "test"
        else:
            dst = "train"
        move_sample(f,os.path.join(directory, "images",dst))

def get_image_size(file):
    from PIL import Image
    im = Image.open(file)
    return im.width, im.height
import os
import struct

def translate_annotations(directory, width=None, height=None):
    os.makedirs(os.path.join(directory, "labels"), exist_ok=True)
    tmp = pathlib.Path(directory)
    for p in tqdm(list(tmp.rglob("*.json")), leave=False):

        imgp = os.path.splitext(p)[0]+".jpg"
        if((width == None or height==None) and os.path.isfile(imgp)):
            width, height = get_image_size(imgp)

        with open(p,"r") as file:
            data = json.load(file)

        pdata = str(p).replace(f"images{os.sep}",f"labels{os.sep}").replace(".json", ".txt")
        os.makedirs(os.path.dirname(pdata), exist_ok=True)
        
        sz = 17 / 480
        with open(pdata,"w") as outfile:
            for k,v in data["kc"].items():
                if("dart" in k.lower()):
                    cl = 0
                elif ("cal" in k.lower()):
                    cl = int(k.replace("cal","").strip())
                outfile.write(f"{cl} {v[0]/width} {v[1]/height} {sz} {sz}\n")
            for k,v in data["bbox"].items():
                if("dart" in k.lower()):
                    cl = 5
                x = (v[0][0]+v[1][0])*0.5
                y = (v[0][1]+v[1][1])*0.5
                w = abs(v[1][0]-v[0][0])
                h = abs(v[1][1]-v[0][1])

                outfile.write(f"{cl} {x/width} {y/height} {w/width} {h/height}\n")

if __name__ == "__main__":
    directory = "_GENERATED"
    validation_ratio = 0.1
    test_ratio = 0.05
    reorganize_images_data(directory,validation_ratio, test_ratio)
    translate_annotations(directory, width=None, height=None)
    with open(os.path.join(directory,"data.yml"),"w") as outfile:
        outfile.write(f"path: {directory}\n")
        outfile.write(f"train: images/train\n")
        outfile.write(f"val: images/val\n")
        if(test_ratio>0):
            outfile.write(f"test: images/test\n")

        outfile.write(f"\n# Classes\n")
        outfile.write(f"names:\n")
        outfile.write(f" 0: tip\n")
        outfile.write(f" 1: cal1\n")
        outfile.write(f" 2: cal2\n")
        outfile.write(f" 3: cal3\n")
        outfile.write(f" 4: cal4\n")
        outfile.write(f" 5: dart\n")


    