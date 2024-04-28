import mitsuba as mi
import json
import os
from obj_tools import *
import hashlib

mi.set_variant("cuda_ad_rgb")
#mi.set_variant('scalar_rgb')

darts_metadata = {}
with open("./3D/Darts/_gen/metadata.json","r") as f:
    darts_metadata = json.load(f)

with open('3D/materials.json','r') as f:
    materials = json.load(f)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    
    Takes module, class, list, dictionary, or string."""
    methodList = [method for method in dir(object) if callable(getattr(object, method))]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                    (method.ljust(spacing),
                    processFunc(str(getattr(object, method).__doc__)))
                    for method in methodList]))

def make_dart_def(tip="A", tip_mat="iron", tip_length = 0.036,
                  barrel = "01", barrel_mat="tungstene", grip_normal_path=None,
                  shaft="78262", shaft_mat = "smooth_plastic", shaft_color = [0,0,0],
                  flight="standard", flight_in = 0.01, flight_texture_file =  '3D/Flights/country-flags/png100px/sh.jpg',
                  **params):

    def _obj_def(file, tr=None, mat_ref="smooth_plastic"):
        ret = {"type": "obj",
                "filename": file,
                "face_normals": False,
                'material':{
                    "type":"ref",
                    "id":mat_ref
                    } 
            }
        if(tr is not None):
            ret["to_world"] = tr
        return ret

    y = 0
    dart_def = {"type": "shapegroup"}
    dart_def["tip"] = _obj_def(f"3D/Darts/_gen/TIPS/{tip}.obj", mat_ref = tip_mat)
    y += tip_length#darts_metadata[f"{tip}.obj"]["max"][1]
    dart_def["barrel"] = _obj_def(f"3D/Darts/_gen/BARRELS/{barrel}.obj", tr=mi.ScalarTransform4f.translate([0.0,y,0]), mat_ref = barrel_mat)
    y += darts_metadata[f"{barrel}.obj"]["max"][1]
    dart_def["shaft"] = _obj_def(f"3D/Darts/_gen/SHAFTS/{shaft}.obj", tr=mi.ScalarTransform4f.translate([0.0,y,0]), mat_ref = shaft_mat)
    y += darts_metadata[f"{shaft}.obj"]["max"][1] - flight_in
    dart_def["flightA"] = _obj_def(f"3D/Darts/_gen/FLIGHTS/{flight}.obj", tr=mi.ScalarTransform4f.translate([0.0,y,0]), mat_ref = "flights_mat")
    dart_def["flightB"] = _obj_def(f"3D/Darts/_gen/FLIGHTS/{flight}.obj", tr=mi.ScalarTransform4f.rotate([0,1,0],90).translate([0.0,y,0]), mat_ref = "flights_mat")
    
    if(materials is not None):
        shaft_mat_def = materials[shaft_mat].copy()
        if(shaft_mat_def["type"] in ["plastic"]):
            if("diffuse_reflectance" not in shaft_mat_def):
                shaft_mat_def["diffuse_reflectance"] = {"type": "rgb"}
            shaft_mat_def["diffuse_reflectance"]["value"] = shaft_color
            dart_def["shaft"]["material"] = shaft_mat_def

        flights_mat = materials['smooth_plastic'].copy()
        flights_mat['diffuse_reflectance'] = {'type': 'bitmap','filename': flight_texture_file}
        materials["flights_mat"] = flights_mat

    if(grip_normal_path is not None):
        dart_def["barrel"]["material"] = {
                'type': 'normalmap',
            'normalmap': {
                'type': 'bitmap',
                'raw': True,
                'filename': grip_normal_path
            }}
        dart_def["barrel"]["material"]["bsdf"] = {"type":"ref", "id":barrel_mat}

    #dart_def["barrel"]["material"]={"type":"ref", "id":"uv_check"}

    return dart_def

def make_lights_def(hdri_file = '3D/hdri/kiara_interior_1k.hdr', hdri_rotation = 0, scale=1.0,**params):
    return {
            # 'type': 'constant',
            # 'radiance': {
            #     'type': 'rgb',
            #     'value': 1.0
            # }
            # NOTE: For better results comment out the constant emitter above
            # and uncomment out the lines below changing the filename to an HDRI
            # envmap you have.
            'type': 'envmap',
            'filename': hdri_file,
            #'filename': '3D/hdri/yoga_room_2k.hdr',
            'scale': scale,
            'to_world': mi.ScalarTransform4f.rotate([0,1,0],angle=hdri_rotation)
        }

def make_board_def(board_image_path='3D/Boards/unicorn-striker.jpg', board_thickness = 0., **params):

    metadata_path = os.path.splitext(board_image_path)[0]+".json"
    board_metadata = {"board": {"r_board":0.2255}}
    if(os.path.exists(metadata_path)):
        with open(metadata_path,"r") as f:
            board_metadata = json.load(f)
            board_metadata["board"]["file"] = metadata_path

    board_radius = board_metadata["board"]["r_board"]
    

    board_def = {'board_sides': {
            'type': 'cylinder',
            'p0': [0,0,0],
            'p1': [0,0,board_thickness],
            'radius': board_radius,
            'material': {
                'type': 'diffuse'
            }
        },
        "board_face": {
            "type": "obj",
            "filename": "3D/Boards/boardface.obj",
            "face_normals": False,
            "to_world": mi.ScalarTransform4f.translate([0,0,board_thickness]).rotate([1, 0, 0], angle=-90).scale(board_radius+0.002),
             'material': {
                'type': 'principled',
                'base_color': {'type': 'bitmap','filename': board_image_path},
                'metallic': 0.2,
                'specular': 0.1,
                'roughness': 0.8,
                'spec_tint': 0.4,
                'anisotropic': 0.5,
                # 'sheen': 0.3,
                # 'sheen_tint': 0.2,
                #'clearcoat': 0.4,
                #'clearcoat_gloss': 0.3,
                # 'spec_trans': 0.4,
                'flatness':0.15
                ,#,'wrap_mode': 'mirror'}
            }
        }
    }
    return board_def, board_metadata

def load_sensor(render_size, rotx, roty, board_radius, board_thickness, fov=75.4, distance_factor = 1.2,**params):
        def get_sensor_transform(rotx, roty, fov, radius):
            d =  radius*distance_factor / np.tan(np.deg2rad(fov*0.5))
            tr = mi.ScalarTransform4f.rotate([1,0,0],rotx).rotate([0,1,0],roty).translate([0,0,d])

            return mi.ScalarTransform4f.look_at(origin= tr.translation().numpy()+[0,0,board_thickness], target=[0, 0, board_thickness],  up=[0, 1, 0])

        return mi.load_dict(
            
        {
            'type':
                'perspective',
            # 'focal_length':
            #     '28mm',
            'fov': fov,
            'to_world':  get_sensor_transform(rotx,roty,fov,board_radius),
            'film': {
                'type': 'hdrfilm',
                'width': render_size,
                'height': render_size,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
            'sampler': {
                'type': 'multijitter',
                'sample_count': 35,
            }
        })

def render(render_size = 800, spp=35, **params ):
    fov = params["sensors_def"].get("fov",37.4)
    scene_base = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'light': make_lights_def( **params["lights_def"]),
        'wall': {
            'type': 'rectangle',
            'material': {
                'type': 'diffuse'
            }            
        },
    }

    board_thickness = params["board_def"].get("board_thickness",0.0381)
    board_def, board_metadata = make_board_def(**params["board_def"])

    scene_base["dart"] = make_dart_def(**params["dart_def"])
        # tip="A",barrel="03", shaft="78281", flight="standard", tip_mat="iron", barrel_mat="brass",
        #                                grip_normal_path='3D/Darts/textures/grip/grip_01_nm.png',
        #                                shaft_mat="aluminium", shaft_color=[0.0,0,1.0],
        #                                materials = materials)

    def add_dart(dic, name, x=0.0,y=0.0, x_angle=0.0, y_angle = 0.0, roll_angle=0.0, penetration = 0.013, ref='dart'):
        transform = mi.ScalarTransform4f.translate([x,y,board_thickness]).rotate([1, 0, 0], angle=90-x_angle).rotate([0, 0, 1], angle=y_angle).rotate([0, 1, 0], angle=roll_angle).translate([0,-penetration,0])
        dic[name]= {
            "type": "instance",
            "to_world": transform,
            'shapegroup': {
                'type': 'ref',
                'id': ref
            }            
        }

    darts = {}
    darts_pos = []# [0.1,0.2,board_thickness], [0.04,0.15,board_thickness], [-0.02,0.05,board_thickness]]
    for k,v in params["darts"].items():
        add_dart(darts, k, **v)
        darts_pos.append([v.get('x',0),v.get('y',0),board_thickness])

    scene = mi.load_dict(materials|scene_base|board_def|darts)

    def backproject(pts, sensor):
        #sensor = get_by_name("sensor", "sensor")
        film = sensor.film()
        prj = mi.perspective_projection(film.size(),film.crop_size(),film.crop_offset(),fov,sensor.near_clip(),sensor.far_clip())
        projs = []
        for p in pts:
            if(p is not None):
                proj = (prj @ sensor.world_transform().inverse() @ p)*render_size
                proj = proj.numpy()[0][0:2]
                projs.append([float(p) for p in proj])
            else:
                proj.append(None)
        return projs
    
    def backproject_uvs(pts_uvs, sensor):
        def get_by_name(name, type="shape"):
            l = scene.sensors() if type=="sensor" else scene.shapes()
            for s in l:
                if(s.id() == name):
                    return s

        bf = get_by_name("board_face")
        d1 = get_by_name("d1")

        # print("sensor shape:", sensor.shape())
        projs = []
        to_backproj = []
        for p in pts_uvs:
            inter = bf.eval_parameterization(uv=mi.Point2f(p[0]/(board_metadata["board"]["width"]-1),p[1]/(board_metadata["board"]["height"]-1)), ray_flags=mi.RayFlags.All, active=True)
            if(inter.is_valid()[0]):
                #pt = inter.p.numpy()[0]
                # pt[2] -= 0.003 # penetration
                # #params["d1.to_world"] = mi.ScalarTransform4f.translate(pt).rotate([1, 0, 0], angle=10).rotate([0, 0, 1], angle=45)
                to_backproj.append(inter.p)
            else:
                print("Invalid")
                to_backproj.append(None)
        return backproject(to_backproj, sensor)
    
    sensors_rot = params["sensors_def"].get("rotations",[[0,0]])
    factors = params["sensors_def"].get("distance_factors",[1.1]*len(sensors_rot))

    sensors = [load_sensor(render_size,sr[0],sr[1],board_metadata["board"]["r_board"],board_thickness,fov, factors[i]) for i,sr in enumerate(sensors_rot)]

    projs_cal = [backproject_uvs([p for k,p in board_metadata["kc"].items() if k in ["cal1","cal2","cal3","cal4"]], sensor=s) for s in sensors]
    projs_darts = [backproject(darts_pos, sensor=s) for s in sensors]
    # params = mi.traverse(scene)
    # params.update()
    
    images = [mi.render(scene, spp=spp, sensor=s) for s in sensors]

    return images, projs_cal, projs_darts, board_metadata

def render_and_save(out_dir="_GENERATED", render_size = 800, spp=35, **params):
    images, projs_cals, projs_darts, board_metadata = render(render_size=render_size, spp=spp, **params)
    os.makedirs(out_dir,exist_ok=True)

    base_file = params.get("base_name","tmp")

    for ii, image in enumerate(images):
        # # Denoise the rendered image
        # denoiser = mi.OptixDenoiser(input_size=image.shape[:2], albedo=False, normals=False, temporal=False)
        # image = denoiser(image)
        img_save_path = os.path.join(out_dir,f'{base_file}_{ii}.jpg')

        #img = (np.clip(image.numpy()** (1.0 / 2.2),0,1.0)* 255).astype('uint8')
        mi.Bitmap(image).convert(pixel_format=mi.Bitmap.PixelFormat.RGB, component_format=mi.Struct.Type.UInt8, srgb_gamma=True).write(img_save_path)

        md = {"kc":{},"board_file":board_metadata["board"]["file"]}
        for i,k in enumerate(["cal1","cal2","cal3","cal4"]):
            md["kc"][k] = projs_cals[ii][i]
        for i in range(len(projs_darts[ii])):
            md["kc"][f"dart{i}"] = projs_darts[ii][i]            

        data_save_path = f"{os.path.splitext(img_save_path)[0]}.json"
        with open(data_save_path, "w") as f:
            json.dump(md, f, indent=4)

def debug_file(file=r"C:\Users\csyllebran\Documents\PERSONNEL\words\d\tmp_0.jpg"):
    from PIL import Image
    from viewer import Application
    import tkinter as tk
    import threading

    app = Application(master= tk.Tk())

    def cb():
        app.set_image(file)
        #app.set_pil_image(Image.fromarray(img))
    timer = threading.Timer(0.4, cb)
    timer.start()
    app.mainloop()

if __name__ == "__main__":
    import numpy as np
    params = {}#"render_size":480, "spp":35}
    params["dart_def"] = {
                    "tip":"A", "tip_mat":"iron", "tip_length" : 0.036, 
                    "barrel" : "02", "barrel_mat":"tungstene", "grip_normal_path":"3D/Darts/textures/grip/grip_01_nm.png", 
                    "shaft":"78262", "shaft_mat" : "smooth_plastic", "shaft_color" : [1,1.0,1.0],
                    "flight":"standard", "flight_in" : 0.01,"flight_texture_file" :  '3D/Flights/arts/ComfyUI_temp_pmzjp_00095_.jpg',
                  }
    params["lights_def"] = {"hdri_file": '3D/hdri/pump_house_1k.hdr', "hdri_rotation" : 50, "scale":0.95 }
    params["board_def"] = {"board_image_path":'3D/Boards/target_wc.jpg', "board_thickness" : 0.0381}
    params["sensors_def"] = {"rotations":[[-5,-20]], "fov":37.4, "distance_factors": [2.2,1.0,0.9]}
    
    params["darts"] = {
            "dart1": {"x":-0.028,"y":0.06, "x_angle":10.0, "y_angle":0.5, "roll_angle":10.6, "penetration": 0.015},
            "dart2": {"x":0.04,"y":0.13, "x_angle":12.0, "y_angle":-0.5, "roll_angle":60.6, "penetration": 0.013},
            "dart3": {"x":0.01,"y":0.03, "x_angle":8.0, "y_angle":-0.2, "roll_angle":124.6, "penetration": 0.013}
        }

    hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()

    render_and_save(render_size=1920, spp=256, out_dir=".", base_name="render", **params)
    debug_file(file=r"./render_0.jpg")



