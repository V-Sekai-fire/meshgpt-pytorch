import os
import subprocess
from multiprocessing import Pool
import trimesh
import numpy as np


def run_command(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    print(f"Command: {cmd}")
    print(f"Stdout: {stdout.decode('utf-8')}")
    print(f"Stderr: {stderr.decode('utf-8')}")
    return stdout, stderr

import argparse

def process_glb_file(glb_path, output_glb_path):
    obj_path = os.path.normpath(
        "temporary/" + glb_path.replace(".glb", "") + "_converted.obj"
    )
    output_base = os.path.normpath(obj_path.replace(".obj", ""))
    remeshed_p0_obj_path = os.path.normpath(output_base + "_rem_p0.obj")
    target_quad_count = 1000
    remeshed_quadrangulation_smooth_obj_path = os.path.normpath(
        output_base + f"_rem_p0_{target_quad_count}_quadrangulation_smooth.obj"
    )
    mesh = trimesh.load(glb_path, force="mesh")
    mesh.merge_vertices(4)
    mesh.vertex_normals
    mesh.export(obj_path, file_type="obj")
    commands = [
        os.path.normpath(f"thirdparty\quadwild_windows\quadwild.exe {obj_path} 2 thirdparty/quadwild_windows/config/prep_config/basic_setup_Mechanical.txt"),
        os.path.normpath(f"thirdparty\quadwild_windows\quad_from_patches.exe {remeshed_p0_obj_path} {target_quad_count} thirdparty/quadwild_windows/config/main_config/flow.txt {output_base}.json"),
    ]
    with Pool(os.cpu_count()) as p:
        results = p.map(run_command, commands)

    mesh = trimesh.load(remeshed_quadrangulation_smooth_obj_path, force="mesh")
    mesh.merge_vertices(4)
    mesh.vertex_normals
    mesh.export(output_glb_path, file_type="glb")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GLB file.")
    parser.add_argument("input", type=str, help="Input GLB file path")
    args = parser.parse_args()

    glb_path = os.path.normpath(args.input)
    output_glb_path = os.path.normpath("./temporary/" + os.path.basename(glb_path).replace(
        ".glb", "_output.glb"
    ))
    process_glb_file(glb_path, output_glb_path)
