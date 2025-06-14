
import numpy as np
from render import render_and_save, render, save_metadata
import hashlib
import json
import random
import os
from tqdm import tqdm
import mitsuba as mi
import cv2

def get_rnd_x_axis(min=-10, max = 50):
    if(np.random.rand() < 0.05): # 5% under 0
         return np.clip(-abs(np.random.exponential(1)),min,0)
    return np.clip(abs(np.random.normal(0, max/4)) ,0,max)

def get_rnd_y_axis():
    return np.random.normal(0, 2)

def get_rnd_z_axis():
    return np.random.uniform(-180,180)

def get_random_axis():
    return [get_rnd_x_axis(),get_rnd_y_axis(),get_rnd_z_axis()]

def get_variations_around_random_axis(number,amplitudes=[5,3,180]):
    ret = []
    for i in range(number):
        axis = get_random_axis()
        vars  = [np.random.normal(0,a/4) for a in amplitudes]
        ret.append(np.array(axis)+np.array(vars))
    return ret

# custom choice function
def choice(l, p = None, num=1):
    prob = p.copy() if p is not None else [None]*len(l)
    if(isinstance(p,dict)):
        prob = [None]*len(l)
        for k,v in p.items():
            for i,name in enumerate(l):
                if(k in name):
                    prob[i] = v
    nb_none = len([v for v in prob if v is None])
    total = sum([v for v in prob if v is not None])
    if(nb_none>0):
        val = (1.0-total)/nb_none
        prob = [val if p is None else p for p in prob]
    ret = random.choices(l, weights = prob, k=num )
    return ret[0] if(num==1) else ret

#print(choice(["poulet","chat","chien","poisson"],p=[None,0.3,None,0.6],num=1))
#print(choice(["poulet","chat","chien","poisson"],p={"poisson":0.7,"chat":0},num=20))

def get_dominant_colors(img_file, palette_size=16, num_colors=10):
    from PIL import Image
    
    # Resize image to speed up processing
    with Image.open(img_file) as img:
        img.thumbnail((100, 100))

        # Reduce colors (uses k-means internally)
        paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)

    dominant_colors = []
    for i in range(num_colors):
      palette_index = color_counts[i][1]
      dominant_colors.append(palette[palette_index*3:palette_index*3+3])

    return dominant_colors


def get_random_sensors_rotations(number = 1, min=-40,max=40):
    ret = np.random.uniform(min,max,(number,2))
    return ret


def get_dir_list(relevant_path, included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif'], full =True, with_ext=True):
    file_names = [fn for fn in os.listdir(relevant_path)
                if any(fn.endswith(ext) for ext in included_extensions)]
    if(full):
        file_names = [os.path.join(relevant_path,f) for f in file_names]
    if(not with_ext):
        file_names = [os.path.splitext(f)[0] for f in file_names]
    return file_names


def test_func(func):
    s = [func() for _ in range(300000)]
    import matplotlib.pyplot as plt
    count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
    plt.axis('tight')
    plt.show()


tips_list = get_dir_list("3D/Darts/_gen/TIPS",['obj'],False, False)
barrels_list = get_dir_list("3D/Darts/_gen/BARRELS",['obj'],False, False)
shafts_list = get_dir_list("3D/Darts/_gen/SHAFTS",['obj'],False, False)
flights_list = get_dir_list("3D/Darts/_gen/FLIGHTS",['obj'],False, False)
boards_list = get_dir_list("3D/Boards/")
hdri_list =  get_dir_list("3D/hdri/",['hdr','exr'])

flights_flags_textures_list =  get_dir_list("3D/Flights/country-flags/png100px/")
flights_arts_textures_list =  get_dir_list("3D/Flights/arts/")

def get_random_coherent_shaft_cols_and_texture():
    flight_texture_file = choice(flights_arts_textures_list if random.random()>0.3 else flights_flags_textures_list)
    shaft_col = [0,0,0]
    if(random.random()>0.2):
        try:
            shaft_col = np.array(get_dominant_colors(flight_texture_file,10)[0])/255.0
        except Exception as e:
            pass
    return flight_texture_file, shaft_col

def get_random_darts(num=3, spread=None):
    angles = get_variations_around_random_axis(num,amplitudes=[5,3,180])
    a = np.random.uniform(0,2*np.pi,num)
    r = (np.random.uniform(0,0.22,num) ** 0.5) * 0.5
    x = np.cos(a) * r
    y = np.sin(a) * r

    if(spread != None):
        x[1:] = np.random.uniform(x[0]-spread,x[0]+spread,num-1)
        y[1:] = np.random.uniform(y[0]-spread,y[0]+spread,num-1)

    ret = {f"dart{i}": {"x":x[i],"y":y[i], "x_angle":angles[i][0], "y_angle":angles[i][1], "roll_angle":angles[i][2], "penetration": np.random.uniform(0.005,0.018)} for i in range(num)}
    return ret

def get_random_scene_def(sensors_num=1, darts_num = None):
    flight_texture_file, shaft_col = get_random_coherent_shaft_cols_and_texture ()

    params = {}#"render_size":480, "spp":35}
    params["dart_def"] = {
                    "tip":choice(tips_list), "tip_mat":"iron", "tip_length" : np.random.uniform(0.032,0.041),
                    "barrel" : choice(barrels_list), "barrel_mat":choice(["brass","tungstene","aluminium"],[0.3,0.65,0.05]), "grip_normal_path":None, 
                    "shaft":choice(shafts_list), "shaft_mat" : "smooth_plastic", "shaft_color" : shaft_col,#[0,0.0,0.5],
                    "flight":choice(flights_list, {"standard":0.5}), "flight_in" : 0.01,"flight_texture_file" : flight_texture_file,
                }
    params["lights_def"] = {"hdri_file": choice(hdri_list), "hdri_rotation" : np.random.uniform(-50,50), "scale":np.random.uniform(1,1.3) }
    params["board_def"] = {"board_image_path":choice(boards_list,{"canaveral":0.2}), "board_thickness" : 0.0381}
    
    params["sensors_def"] = {"rotations":get_random_sensors_rotations(sensors_num), "fov":37.4, "distance_factors": np.random.uniform(0.9,1.3,sensors_num)}
    

    nb_darts = darts_num if darts_num is not None else choice([1,2,3],[0.2,0.3,0.5])
    params["darts"] = get_random_darts(nb_darts, spread=0.02 if random.random()<0.5 else None)

    # params["darts"] = {
    #         "dart1": {"x":0.0,"y":0.0, "x_angle":10.0, "y_angle":0.5, "roll_angle":10.6, "penetration": 0.013},
    #         "dart2": {"x":0.07,"y":0.1, "x_angle":12.0, "y_angle":-0.5, "roll_angle":60.6, "penetration": 0.013}
    #     }
    return params

def render_random(render_size=480, spp=35, out_dir="_GENERATED", total=60, sensors_num = 1):
    for i in tqdm(range(total//sensors_num)):
        params = get_random_scene_def(sensors_num)
        hash = hashlib.sha256(json.dumps(params, sort_keys=True,default=lambda e : str(e)).encode('utf-8')).hexdigest()

        render_and_save(render_size=render_size, spp=spp, out_dir=out_dir, base_name=hash, **params)


def render_random_diff(render_size=480, spp=35, out_dir="_GENERATED", total=60, sensors_num = 1,angle_jitter = 0.0,distance_jitter=0.0):
    os.makedirs(out_dir,exist_ok=True)
    for i in tqdm(range(total//sensors_num)):
        params = get_random_scene_def(sensors_num,3)
        hash = hashlib.sha256(json.dumps(params, sort_keys=True,default=lambda e : str(e)).encode('utf-8')).hexdigest()

        params_mods = params.copy()

        all = []
        for drt in range(4):
            params_mods = params.copy()
            params_mods["darts"] = {} if drt == 0 else {k:params["darts"][k] for k in list(params["darts"].keys())[:drt]}
            if(angle_jitter>0):
                params_mods["sensors_def"]["rotations"] =  params["sensors_def"]["rotations"]+np.random.normal(-angle_jitter,angle_jitter,params["sensors_def"]["rotations"].shape)
            if(distance_jitter>0):
                params_mods["sensors_def"]["distance_factors"] =  params["sensors_def"]["distance_factors"]+np.random.normal(-distance_jitter,distance_jitter,params["sensors_def"]["distance_factors"].shape)
                
            images, projs_cals, projs_darts, projs_darts_bbox, board_metadata = render(render_size=render_size, spp=spp, **params_mods)
            all.append((images, projs_cals, projs_darts, projs_darts_bbox, board_metadata))

        # hash = hashlib.sha256(json.dumps(params, sort_keys=True,default=lambda e : str(e)).encode('utf-8')).hexdigest()

        # render_and_save(render_size=render_size, spp=spp, out_dir=out_dir, base_name=hash, **params)
        print("sensors:",len(all[drt][0]))
        base_file = hash
        for ii in range(sensors_num):
            empty = None
            for drt in range(len(all)):
                # # Denoise the rendered image
                # denoiser = mi.OptixDenoiser(input_size=image.shape[:2], albedo=False, normals=False, temporal=False)
                # image = denoiser(image)
                
                img = (np.clip(all[drt][0][ii].numpy()** (1.0 / 2.2),0,1.0)* 255).astype('uint8')
                #mi.Bitmap(all[drt][0][ii]).convert(pixel_format=mi.Bitmap.PixelFormat.RGB, component_format=mi.Struct.Type.UInt8, srgb_gamma=True).write(img_save_path)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #img = cv2.blur(img,(5,5))
                if(empty is None):
                    empty = img
                else:
                    diff = cv2.absdiff(img,empty)
                    #blur_sz = 3
                    #kernel = np.ones((blur_sz, blur_sz), np.uint8)
                    #diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
                    #diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
                    #diff = cv2.erode(diff,(3,3), iterations = 3)
                    
                    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    # cv2.imshow("dart",diff)
                    # cv2.imshow("prev",empty)
                    # cv2.imshow("img",img)
                    # cv2.waitKey()
                    empty = img
                    tip = all[drt][2][ii][-1]
                    box = all[drt][3][ii][-1]
                    all[drt][2][ii] = [tip]
                    all[drt][3][ii] = [box]
                    img_save_path = os.path.join(out_dir,f'{base_file}_{drt}_{ii}.jpg')
                    cv2.imwrite(img_save_path,diff)
                    save_metadata(f"{os.path.splitext(img_save_path)[0]}.json",all[drt][4], all[drt][1][ii], all[drt][2][ii], all[drt][3][ii])

if __name__ == "__main__":

#print(get_variations_around_random_axis(3,amplitudes=[5,3,180]))
#test_func(get_rnd_x_axis)
    
    import argparse

    parser = argparse.ArgumentParser(description='Generate a realistic random darts dataset.')
    parser.add_argument('number', metavar='N', type=int, help='number of samples to be generated')
    parser.add_argument("-s",'--render_size', type=int, default=640, help="Render size is sxs")
    parser.add_argument("-v",'--view_points', type=int, default=3, help="Number of view points renders for each sample")
    parser.add_argument("-q",'--quality', type=int, default=35, help="Number of sample per rays")
    parser.add_argument("-d", "--directory", type=str, default="_GENERATED", help="Destination directory")
    parser.add_argument("-t", "--type", type=str, default= "normal", choices=['normal', 'temporal'], help="Dataset type")
    
    args = parser.parse_args()

    from mitsuba import Thread, LogLevel
    Thread.thread().logger().set_log_level(LogLevel.Error)

    if(args.type == "temporal"):
        render_random_diff(render_size=args.render_size,spp=args.quality,sensors_num=args.view_points, out_dir=args.directory, total=args.number, angle_jitter = 0.1, distance_jitter = 0)
    else:
        render_random(render_size=args.render_size,spp=args.quality,sensors_num=args.view_points, out_dir=args.directory, total=args.number)
    
