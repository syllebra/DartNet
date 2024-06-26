
import os
import random
import shutil
from tqdm import tqdm
import pathlib
import json
import numpy as np
import glob
from add_cross_sections import get_cross_sections

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

def revert_initial_data(directory):
    # Get List of all images
    files = glob.glob(directory + '/**/*.jpg', recursive=True)

    for f in tqdm(files):
         if(os.path.abspath(os.path.dirname(f)) != os.path.abspath(directory)):
            move_sample(f,directory)

def get_image_size(file):
    from PIL import Image
    im = Image.open(file)
    return im.width, im.height
import os
import struct

def translate_annotations(directory, width=None, height=None, no_cals=False, add_cross_sections=False):
    os.makedirs(os.path.join(directory, "labels"), exist_ok=True)
    tmp = pathlib.Path(directory)
    for p in tqdm(list(tmp.rglob("*.json")), leave=False):

        imgp = os.path.splitext(p)[0]+".jpg"
        if((width == None or height==None) and os.path.isfile(imgp)):
            width, height = get_image_size(imgp)

        if((width == None or height==None)):
            continue
        
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
                if(not no_cals or cl <1):
                    if(v[0]>=0 and v[0]<width and v[1]>0 and v[1]<height):
                        outfile.write(f"{cl} {v[0]/width} {v[1]/height} {sz} {sz}\n")
            for k,v in data["bbox"].items():
                if("dart" in k.lower()):
                    cl = 1 if no_cals else 5

                if(v[0][0] <0 and v[1][0] <0): continue
                if(v[0][0] >=width and v[1][0] >=width): continue
                if(v[0][1] <0 and v[1][1] <0): continue
                if(v[0][1] >=height and v[1][1] >=height): continue

                v[0] = np.clip(np.array(v[0]),(0,0),(width-1,height-1))
                v[1] = np.clip(np.array(v[1]),(0,0),(width-1,height-1))
                x = (v[0][0]+v[1][0])*0.5
                y = (v[0][1]+v[1][1])*0.5
                w = abs(v[1][0]-v[0][0])
                h = abs(v[1][1]-v[0][1])

                outfile.write(f"{cl} {x/width} {y/height} {w/width} {h/height}\n")
            
            if(add_cross_sections):
                cl = 6
                sz = 17
                cs = get_cross_sections(p,sz, W=width, H=height)
                for p in  cs:
                    outfile.write(f"{cl} {p[0]/width} {p[1]/height} {sz/width} {sz/height}\n")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Generate a realistic random darts dataset.')
    parser.add_argument("-d", "--directory", type=str, default="_GENERATED", help="Destination directory")
    parser.add_argument('--revert', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--no_cals', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cross_sections', default=False, action=argparse.BooleanOptionalAction)
    parser.set_defaults(revert=False)

    args = parser.parse_args()
    
    
    if(args.revert):
        revert_initial_data(args.directory)
        if(os.path.exists(os.path.join(args.directory,"data.yml"))):
            os.remove(os.path.join(args.directory,"data.yml"))
        if(os.path.exists(os.path.join(args.directory,"images"))):
            shutil.rmtree(os.path.join(args.directory,"images"))
        if(os.path.exists(os.path.join(args.directory,"labels"))):
            shutil.rmtree(os.path.join(args.directory,"labels"))
    else:
        directory = args.directory
        validation_ratio = 0.1
        test_ratio = 0.05
        reorganize_images_data(directory,validation_ratio, test_ratio)
        translate_annotations(directory, width=None, height=None, no_cals=args.no_cals, add_cross_sections=args.cross_sections)
        with open(os.path.join(directory,"data.yml"),"w") as outfile:
            #outfile.write(f"path: {directory}\n")
            outfile.write(f"train: images/train\n")
            outfile.write(f"val: images/val\n")
            if(test_ratio>0):
                outfile.write(f"test: images/test\n")

            outfile.write(f"\n# Classes\n")
            outfile.write(f"names:\n")
            outfile.write(f" 0: tip\n")
            if(not args.no_cals):
                outfile.write(f" 1: cal1\n")
                outfile.write(f" 2: cal2\n")
                outfile.write(f" 3: cal3\n")
                outfile.write(f" 4: cal4\n")
                outfile.write(f" 5: dart\n")
                outfile.write(f" 6: cross\n")
            else:
                outfile.write(f" 1: dart\n")


    