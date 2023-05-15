#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
Script for loading a GFX replay file into Blender. This allows playing back a trajectory from Habitat in Blender.

See the following tutorial for more information on GFX: https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/replay_tutorial.ipynb

Example Usage:
```
blender --python scripts/blender_load_replay.py -- --replay data/replays/episode8362.replay.json --root-dir ./
```
- Replace `/Applications/Blender.app/Contents/MacOS/Blender` with your path to blender.
- Also specify `--settings-path settings.json` for dataset specific blender settings (to avoid manually changing the settings in blender for every replay).
Example `settings` json with all supported fields:
```
{
  "lights": [
    {
      "type": "SUN",
      "name": "my-light",
      "energy": 3.0,
      "location": [1.40667, -2.66486, 6.1511]
    }
  ]
}
```

"""


try:
    import bpy
except ImportError:
    raise ImportError(
        "Failed to import Blender modules. This script can't run "
        "standalone. Run `blender --python path/to/blender_load_replay.py ...`. Watch the terminal for "
        "debug/error output."
    )

import argparse
import json
import os
import mathutils
import bmesh
from dataclasses import dataclass
from typing import Dict

_ignore_prefix_list = ["capsule3DSolid",     "capsule3DWireframe", "coneSolid",
                       "coneWireframe",      "cubeSolid",          "cubeWireframe",
                       "cylinderSolid",      "cylinderWireframe",  "icosphereSolid",
                       "icosphereWireframe", "uvSphereSolid",      "uvSphereWireframe",
                       "bellow"]

def _is_creation_valid(filepath: str) -> bool:
    for ignore_prefix in _ignore_prefix_list:
        if ignore_prefix in filepath:
            return False
    return True

@dataclass
class ImportItem:
    filepath: str
    do_join_all: bool = True
    force_color: list = None

class Rig:
    armature = None
    bone_map: dict = {}
    point_cloud: dict = {}

def import_scene_helper(raw_filepath):
    """
    Import a single asset into Blender
    """

    # These fixups will be tried one at a time, in order, until a matching, existing file is found
    model_filepath_fixups = [
        # use uncompressed versions when available
        ("/stages/", "/stages_uncompressed/"),
        ("/urdf/", "/urdf_uncompressed/"),
        None,
    ]

    filepath = None
    for fixup in model_filepath_fixups:
        raw_filepath = raw_filepath.split("?")[0]
        if fixup is None:
            filepath = raw_filepath
        else:
            if raw_filepath.find(fixup[0]) != -1:
                filepath = raw_filepath.replace(fixup[0], fixup[1])
            else:
                continue
        if os.path.exists(filepath):
            break
    else:
        raise RuntimeError("can't find file " + raw_filepath, filepath)
    

    if not os.path.isfile(filepath):
        print("ERROR: Cannot find file: " + filepath)

    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".dae":
        # Convert to GLB.
        # TODO: This adds a dependency to assimp
        orig_filepath = filepath
        filepath = filepath.replace(".dae", ".glb")
        raw_filepath = raw_filepath.replace(".dae", ".glb")
        if not os.path.exists(filepath):
            os.system(f"assimp export {orig_filepath} {filepath}")
        ext = ".glb"

    if ext == ".glb" or ext == ".gltf":
        if "ycb" in filepath:
            filepath += ".orig"

        filename = os.path.basename(filepath)
        try:
            bpy.ops.import_scene.gltf(
                filepath=filepath, files=[{"name": filename}],  bone_heuristic="BLENDER"
            )
        except:
            print("FAILED: " + filepath)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=filepath)
    else:
        raise RuntimeError("no importer found for " + filepath)

    return filepath


def import_item(item):
    import_scene_helper(item.filepath)

    bpy.ops.object.select_all(action="SELECT")
    if len(bpy.context.selected_objects) == 0:
        raise ValueError("No objects found in scene")
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    if "Stage" in item.filepath:

        def should_ignore(x):
            ignore_words = ["Light", "ceiling"]
            return any(w in x for w in ignore_words)

        remove_objs = [
            x for x in bpy.context.selected_objects if should_ignore(x.name)
        ]
        bpy.ops.object.delete({"selected_objects": remove_objs})

    childless_empties = [
        e
        for e in bpy.context.selected_objects
        if e.type.startswith("EMPTY") and not e.children
    ]
    if len(childless_empties):
        print(
            "removing {} childless EMPTY nodes".format(len(childless_empties))
        )
        while childless_empties:
            bpy.data.objects.remove(childless_empties.pop())
        bpy.ops.object.select_all(action="SELECT")
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

    if item.do_join_all:
        if len(bpy.context.selected_objects) > 1:
            try:
                bpy.ops.object.join()
                bpy.ops.object.select_all(action="SELECT")
            except BaseException:
                pass
        o = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.transform_apply(
            location=True, rotation=True, scale=True
        )

    # Currently unused for here for reference, in case we add color override to gfx-replay
    if item.force_color:
        o = bpy.context.selected_objects[0]
        mtrl = o.data.materials[0]
        mtrl.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            item.force_color[0],
            item.force_color[1],
            item.force_color[2],
            1,
        )

    retval = bpy.context.selected_objects[0]

    for o in bpy.context.selected_objects:
        o.hide_set(True)

    return retval


def import_gfx_replay(replay_filepath, settings):
    # Remove default assets (e.g. cube) from scene
    for object in bpy.context.scene.objects:
        object.select_set(True)
    bpy.ops.object.delete()
    
    with open(replay_filepath, "r") as f:
        json_root = json.load(f)
        assert "keyframes" in json_root
        keyframes = json_root["keyframes"]
        assert len(keyframes) > 0

    render_asset_map = {}
    asset_info_by_filepath = {}
    asset_info_by_key = {}
    rig_map: Dict(int, Rig) = {}

    do_add_anim_keyframes = len(keyframes) > 1
    for keyframe_index, keyframe in enumerate(keyframes):

        if "loads" in keyframe:
            for asset_info in keyframe["loads"]:
                filepath = asset_info["filepath"]
                asset_info_by_filepath[filepath] = asset_info
        
        if "creations" in keyframe:
            last_scene_idx = 0
            for i, x in enumerate(keyframe["creations"]):
                fpath = x["creation"]["filepath"].split("/")[-1]
                if "Stage" in fpath:
                    last_scene_idx = i

            all_c = keyframe["creations"][last_scene_idx:]
            for creation_dict in all_c:
                filepath = creation_dict["creation"]["filepath"]
                if not _is_creation_valid(filepath):
                    print("Ignoring {}".format(filepath))
                    continue
                obj = import_item(ImportItem(filepath))
                if "scale" in creation_dict["creation"]:
                    obj.scale = creation_dict["creation"]["scale"]
                instance_key = creation_dict["instanceKey"]
                render_asset_map[instance_key] = obj

                filepath = filepath.split("?")[0]
                asset_info_by_key[instance_key] = asset_info_by_filepath[
                    filepath
                ]

                # Check if the object has an armature
                if "rigId" in creation_dict["creation"]:
                    rig = Rig()
                    rig.armature = obj
                    rig_map[creation_dict["creation"]["rigId"]] = rig

        if "boneCreations" in keyframe:
            for bone_creation_dict in keyframe["boneCreations"]:
                rig_id = bone_creation_dict["rigId"]
                bone_id = bone_creation_dict["id"]
                boneName = bone_creation_dict["name"]

                rig = rig_map[rig_id]

                for bone in rig.armature.pose.bones:
                    if bone.name == boneName:
                        rig.bone_map[bone_id] = bone
                        #TODO: Fails if multiple creations

                        # Create animation point cloud.
                        mesh = bpy.data.meshes.new('_debug_geom_' + str(rig_id) + '_' + bone.name)
                        obj = bpy.data.objects.new('_source_' + str(rig_id) + '_' + str(bone.name), mesh)
                        rig.point_cloud[bone_id] = obj
                        bpy.context.collection.objects.link(obj)
                        
                        # Uncomment to visualize point cloud
                        #bpy.context.view_layer.objects.active = obj
                        #obj.select_set(True)
                        #bm = bmesh.new()
                        #bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=8, radius=0.025)
                        #bm.to_mesh(mesh)
                        #bm.free()

                        # This is required for keyframing quaternion rotations.
                        obj.rotation_mode = "QUATERNION"

                        # Add constraint to bone so that it follows its associated point in the point cloud.
                        constraint = bone.constraints.new('COPY_TRANSFORMS')
                        constraint.target = obj
                        bpy.context.view_layer.update()
                        break
        
        bpy.context.view_layer.update()

        if "stateUpdates" in keyframe:
            for update_dict in keyframe["stateUpdates"]:
                instance_key = update_dict["instanceKey"]
                translation = update_dict["state"]["absTransform"][
                    "translation"
                ]
                rotation = update_dict["state"]["absTransform"]["rotation"]
                if instance_key not in render_asset_map:
                    continue
                obj = render_asset_map[instance_key]

                obj.rotation_mode = "QUATERNION"

                asset_info = asset_info_by_key[instance_key]

                # note coordinate convention change for Blender
                obj.location = (
                    translation[0],
                    -translation[2],
                    translation[1],
                )
                obj.rotation_quaternion = (
                    rotation[0],
                    rotation[1],
                    -rotation[3],
                    rotation[2],
                )

                frame = asset_info["frame"]
                if frame["up"] == [0.0, 1.0, 0.0]:
                    pass
                elif frame["up"] == [0.0, 0.0, 1.0]:
                    obj.rotation_mode = "XYZ"
                    obj.rotation_euler[0] -= 1.5708
                else:
                    raise NotImplementedError(
                        "unexpected coordinate frame " + frame
                    )

        # Update scene.
        bpy.context.view_layer.update()

        # Grounding translation
        # TODO: The model isn't grounded. For now, this empirical correction is used.
        rig_grounding_matrix = mathutils.Matrix.Translation([0.0, 0.0, -0.125]) # -0.41

        # Bone orientation correction matrix
        # TODO: The model bone orientation is wrong. For now, this empirical correction is used.
        bone_correction_matrix = mathutils.Matrix(((1.0, 0.0, 0.0, 0.0),
                                                   (0.0, 0.0,-1.0, 0.0),
                                                   (0.0, 1.0, 0.0, 0.0),
                                                   (0.0, 0.0, 0.0, 1.0)))

        if "boneUpdates" in keyframe:
            for bone_update_dict in keyframe["boneUpdates"]:
                rig_id = bone_update_dict["rigId"]
                bone_id = bone_update_dict["boneId"]
                rig = rig_map[rig_id]
                bone = rig.bone_map[bone_id]

                bone_world_translation = bone_update_dict["absTransform"]["translation"]
                bone_world_translation = (
                    bone_world_translation[0],
                    -bone_world_translation[2],
                    bone_world_translation[1],
                )
                bone_world_translation = mathutils.Matrix.Translation(bone_world_translation)

                bone_world_rotation = bone_update_dict["absTransform"]["rotation"]
                bone_world_rotation = (
                    bone_world_rotation[0],
                    bone_world_rotation[1],
                    -bone_world_rotation[3],
                    bone_world_rotation[2],
                )
                bone_world_rotation = mathutils.Quaternion(bone_world_rotation)
                bone_world_rotation = bone_world_rotation.to_matrix()
                bone_world_rotation.resize_4x4()

                bone_world_matrix = rig_grounding_matrix @ bone_world_translation @ bone_world_rotation @ bone_correction_matrix

                # The bones are animated by constraining them to their associated point in the point cloud.
                # This animation method is used to avoid animation jittering caused by directly assigning bone matrices.
                #print(str(bone_id))
                #print(rig.point_cloud[bone_id].name)
                rig.point_cloud[bone_id].matrix_world = bone_world_matrix

        if do_add_anim_keyframes:
            for instance_key in render_asset_map:
                obj = render_asset_map[instance_key]
                obj.keyframe_insert(data_path="location", frame=keyframe_index)
                obj.keyframe_insert(
                    data_path="rotation_quaternion", frame=keyframe_index
                )
            for rig_id, rig in rig_map.items():
                rig.armature.keyframe_insert(data_path="location", frame=keyframe_index)
                rig.armature.keyframe_insert(data_path="rotation_quaternion", frame=keyframe_index)
                for bone_id, bone in rig.bone_map.items():
                    rig.point_cloud[bone_id].keyframe_insert(data_path="location", frame=keyframe_index)
                    rig.point_cloud[bone_id].keyframe_insert(data_path="rotation_quaternion", frame=keyframe_index)

    for o in bpy.context.scene.objects:
        o.hide_set(False)
    # To fix import issue where some robot models would have transparent links.
    for m in bpy.data.materials:
        m.show_transparent_back = False

    add_lights = settings.get("lights", [])
    for light in add_lights:
        # Create light datablock
        light_data = bpy.data.lights.new(
            name=light["name"], type=light["type"]
        )
        light_data.energy = light["energy"]

        # Create new object, pass the light data
        light_object = bpy.data.objects.new(
            name=light["name"], object_data=light_data
        )

        # Link object to collection in context
        bpy.context.collection.objects.link(light_object)

        # Change light position
        light_object.location = light["location"]

    print("")
    if len(keyframes) > 1:
        print(
            "Success! Imported {} with {} render instances and {} animation keyframes.".format(
                replay_filepath, len(render_asset_map), len(keyframes)
            )
        )
    else:
        print(
            "Success! Imported {} with {} render instances (no animation found)".format(
                replay_filepath, len(render_asset_map)
            )
        )
    print("")
    print(
        "Explore the Blender GUI window to visualize your replay, then close it when done."
    )


if __name__ == "__main__":

    import sys

    argv = sys.argv
    argv = argv[argv.index("--") + 1 :]  # get all args after "--"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        type=str,
        required=True,
        help="Path to the replay file relative to the `root-dir`.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="The root directory for the assets and replay file.",
    )
    parser.add_argument(
        "--settings-path",
        type=str,
        default=None,
        help="Optional. Path to a yaml file describing additional scene settings. See doc string at top of this file for more info.",
    )
    args = parser.parse_args(argv)

    if args.settings_path is not None:
        with open(args.settings_path, "r") as f:
            settings = json.load(f)
    else:
        settings = {}

    os.chdir(
        args.root_dir
    )  # todo: get working directory from the replay, itself
    import_gfx_replay(args.replay, settings)
