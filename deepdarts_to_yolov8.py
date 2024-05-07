import pickle
import pandas as pd
import os
import cv2
from tqdm import tqdm
import json

def convert(src_dir="datasets/deepdarts", out_dir="_GENERATED", sz=640):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = f"{src_dir}/labels.pkl"

    with open(pkl_path,"rb") as f:
        df: pd.DataFrame = pickle.load(f)

    src_dir = f"{src_dir}/cropped_images/800"

    cols = [(0,215,255),(180, 105, 255),(112,255,202),(114,128,250)]
    for num,d in tqdm(enumerate(df.iterrows()), total=len(df)):
        img_path = os.path.join(src_dir,d[1]["img_folder"],d[1]["img_name"])
        if(os.path.exists(img_path)):
            im = cv2.imread(img_path)
            im = cv2.resize(im,(sz,sz))
            out_img_path = os.path.join(out_dir,f"{num}.jpg")
            out_meta_path = os.path.join(out_dir,f"{num}.json")
            cv2.imwrite(out_img_path, im)

            dname = f'DeepDarts_{d[1]["img_folder"].split("_")[0]}'
            dt = {"kc":{},"board_file": f"3D/Boards/{dname}.json"}
            for i,ptr in enumerate(d[1]["xy"]):
                x = int(ptr[0]*im.shape[1])
                y = int(ptr[1]*im.shape[0])
                kc_name = f"cal{i+1}" if i<4 else f"dart{i-4}"
                dt["kc"][kc_name] = [x,y]
            with open(out_meta_path,"w") as outf:
                json.dump(dt, outf)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert dataset from DeepDarts to DartNet.')
    parser.add_argument("-d", "--directory", type=str, default="cropped_images", help="DeepDarts source directory")
    parser.add_argument("-o", "--output_dir", type=str, default="_GENERATED", help="DartNet destination directory")
    parser.add_argument("-s",'--render_size', type=int, default=640, help="Render size is sxs")

    args = parser.parse_args()
    convert(args.directory, args.output_dir, args.render_size)
