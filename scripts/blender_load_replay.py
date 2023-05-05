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
    id_to_bone_map: dict = {}
    debug_point_cloud: dict = {}
    inverse_root_transform = None

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

    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".dae":
        # Convert to GLB.
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
        bpy.ops.import_scene.gltf(
            filepath=filepath, files=[{"name": filename}],  bone_heuristic="BLENDER"
        )
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
            last_scene_idx = 0 #TODO: ??
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
                rigId = bone_creation_dict["rigId"]
                boneId = bone_creation_dict["id"]
                boneName = bone_creation_dict["name"]

                rig = rig_map[rigId]

                for bone in rig.armature.pose.bones:
                    if bone.name == boneName:
                        rig.id_to_bone_map[boneId] = bone

                        # debug point cloud
                        mesh = bpy.data.meshes.new('_debug_' + bone.name)
                        obj = bpy.data.objects.new('_debug_' + str(bone.name), mesh)
                        bpy.context.collection.objects.link(obj)
                        bpy.context.view_layer.objects.active = obj
                        obj.select_set(True)
                        bm = bmesh.new()
                        #bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=8, radius=0.025)
                        bmesh.ops.create_cone(bm, segments=8, radius1=0.01, radius2=0, depth=0.1)
                        bm.to_mesh(mesh)
                        bm.free()
                        rig.debug_point_cloud[boneId] = obj
                        break

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
                
                # TODO: Model is rotated during import. Figure out where this comes from.
                #if obj.name == "SMPLX-female":
                #    obj.rotation_mode = "XYZ"
                #    obj.rotation_euler[0] += 1.5708

        # Update scene.
        # Without this step, some matrices, like `Object.matrix_world`, won't be updated by Blender after the previous step.
        bpy.context.view_layer.update()
        
        if "rigUpdates" in keyframe:
            for rig_update_dict in keyframe["rigUpdates"]:
                rigId = rig_update_dict["rigId"]

                rig_world_translation = rig_update_dict["rootTransform"]["translation"]
                rig_world_translation = (
                    rig_world_translation[0],
                    -rig_world_translation[2],
                    rig_world_translation[1],
                )
                rig_world_translation = mathutils.Matrix.Translation(rig_world_translation)

                rig_world_rotation = rig_update_dict["rootTransform"]["rotation"]
                rig_world_rotation = (
                    rig_world_rotation[0],
                    rig_world_rotation[1],
                    -rig_world_rotation[3],
                    rig_world_rotation[2],
                )
                rig_world_rotation = mathutils.Quaternion(rig_world_rotation)
                rig_world_rotation = rig_world_rotation.to_matrix()
                rig_world_rotation.resize_4x4()

                rig_rotation_correction = mathutils.Matrix.Rotation(-1.5708, 4, 'X')

                rig_world_matrix = rig_world_translation @ rig_world_rotation @ rig_rotation_correction

                rig.inverse_root_transform = rig_world_matrix.inverted()

        if "boneUpdates" in keyframe:
            
            for bone_update_dict in keyframe["boneUpdates"]:
                rigId = bone_update_dict["rigId"]
                boneId = bone_update_dict["boneId"]
                rig = rig_map[rigId]
                bone = rig.id_to_bone_map[boneId]

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

                correction_matrix = mathutils.Matrix(((1.0, 0.0, 0.0, 0.0),
                                                      (0.0, 0.0, -1.0, 0.0),
                                                      (0.0, 1.0, 0.0, 0.0),
                                                      (0.0, 0.0, 0.0, 1.0)))
                bone_world_matrix = bone_world_translation @ bone_world_rotation# @ correction_matrix

                bone_rotation_correction = mathutils.Matrix.Rotation(1.5708, 4, 'X')
                # Hack: Offset correction. Some transform is not correctly applied. Until it is found, this works fine for the demo model:
                root_rotation_correction = mathutils.Matrix.Translation([0,0.42,0]) @ mathutils.Matrix.Rotation(1, 4, 'Y') @ mathutils.Matrix.Rotation(1.5708, 4, 'X')
                #root_rotation_correction = mathutils.Matrix.Translation([0,0.42,0]) @ mathutils.Matrix.Rotation(1.5708, 4, 'Y') @ mathutils.Matrix.Rotation(1.5708, 4, 'X')
                
                # Obscure Hack: Bone matrix manipulation is less jittery in 'POSE' mode.
                # Select armature and set editor mode to POSE
                #bpy.context.view_layer.objects.active = rig.armature
                #rig.armature.select_set(True)
                #bpy.ops.object.mode_set(mode="POSE")

                #rig.armature.matrix_world.inverted() vs rig.inverse_root_transform


                root_bone = None
                for armature_bone in rig.armature.pose.bones:
                    if armature_bone.parent == None:
                        root_bone = armature_bone
                        break
                root_bone_world_matrix = rig.armature.matrix_world# @ root_bone.matrix
                root_bone_world_translation = mathutils.Matrix.Translation(root_bone_world_matrix.decompose()[0])

                bone_pose_matrix = root_rotation_correction @ root_bone_world_translation.inverted() @ bone_world_matrix @ bone_rotation_correction
                bone.matrix = bone_pose_matrix

                rig.debug_point_cloud[boneId].matrix_world = bone_world_matrix

                # matrix_local = matrix_parent_inverse * matrix_basis, and matrix_world = parent.matrix_world * matrix_local
                #NOTE:
                #Habitat calculation from world space coords:
                #  jointTransformations_[i] =
                #    invRootTransform *
                #    jointNodeIt->second->absoluteTransformationMatrix() *
                #    skin->inverseBindMatrices()[i];
                #NOTE:
                #https://blender.stackexchange.com/questions/44637/how-can-i-manually-calculate-bpy-types-posebone-matrix-using-blenders-python-ap
                #https://blender.stackexchange.com/questions/109815/how-can-i-move-a-posebone-to-a-specific-world-space-position

        if do_add_anim_keyframes:
            for instance_key in render_asset_map:
                obj = render_asset_map[instance_key]
                obj.keyframe_insert(data_path="location", frame=keyframe_index)
                obj.keyframe_insert(
                    data_path="rotation_quaternion", frame=keyframe_index
                )
            for rigId, rig in rig_map.items():
                for boneId, bone in rig.id_to_bone_map.items():
                    bone.keyframe_insert(data_path="location", frame=keyframe_index)
                    bone.keyframe_insert(data_path="rotation_quaternion", frame=keyframe_index)
                    rig.debug_point_cloud[boneId].keyframe_insert(data_path="location", frame=keyframe_index)
                    rig.debug_point_cloud[boneId].keyframe_insert(data_path="rotation_quaternion", frame=keyframe_index)

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
