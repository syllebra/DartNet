# This script will be executed inside blender and run by blender_run.property
# It will use the blender's python version and modules...

import bpy
import mathutils

import os

collection_names = ["FLIGHTS","BARRELS","TIPS","SHAFTS"]

# If collection_names is an empty list, use all collections,
# otherwise use only collections whose names are in collection_names
collections = bpy.data.collections if not collection_names else [col for col in bpy.data.collections if col.name in collection_names]

for col in collections:
    os.makedirs(f"./tmp/{col.name}", exist_ok=True)
    for obj in col.all_objects:
        if("_REF" in obj.name):
            continue

        for ob in bpy.context.selected_objects:
            ob.select_set(False)
        obj.select_set(True)

        # move to scened origin
        obj.location = mathutils.Vector((0,0,0))

        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(
            filepath=f"./tmp/{col.name}/{obj.name}.obj",
            check_existing=False,
            filter_blender=False,
            filter_backup=False,
            filter_image=False,
            filter_movie=False,
            filter_python=False,
            filter_font=False,
            filter_sound=False,
            filter_text=False,
            filter_archive=False,
            filter_btx=False,
            filter_collada=False,
            filter_alembic=False,
            filter_usd=False,
            filter_obj=False,
            filter_volume=False,
            filter_folder=True,
            filter_blenlib=False,
            filemode=8,
            display_type='DEFAULT',
            sort_method='DEFAULT',
            export_animation=False,
            start_frame=-2147483648,
            end_frame=2147483647,
            forward_axis='NEGATIVE_Z',
            up_axis='Y',
            global_scale=0.001,
            apply_modifiers=True,
            export_eval_mode='DAG_EVAL_VIEWPORT',
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_colors=False,
            export_materials=False,
            export_pbr_extensions=False,
            path_mode='AUTO',
            export_triangulated_mesh=False,
            export_curves_as_nurbs=False,
            export_object_groups=False,
            export_material_groups=False,
            export_vertex_groups=False,
            export_smooth_groups=False,
            smooth_group_bitflags=False,
            filter_glob='*.obj;*.mtl'
        )
        
        obj.select_set(False)