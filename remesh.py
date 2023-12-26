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
    output_basename = os.path.splitext(os.path.basename(glb_path))[0]
    mesh = trimesh.load(glb_path, force="mesh")
    mesh.merge_vertices(4)
    mesh.vertex_normals
    source_file = os.path.normpath('temporary/' + output_basename + '.obj')
    mesh.export(source_file, file_type="obj")
    target_quad_count = 1000
    source_file_second_stage = os.path.normpath("./temporary/" + output_basename + "_rem_p0.obj")
    commands = [
        os.path.normpath(f"./thirdparty/quadwild_windows/quadwild {source_file} 2 thirdparty/quadwild_windows/config/prep_config/basic_setup_Mechanical.txt"),
        os.path.normpath(f"./thirdparty/quadwild_windows/quad_from_patches {source_file_second_stage} {target_quad_count} thirdparty/quadwild_windows/config/main_config/flow.txt"),
    ]
    with Pool(os.cpu_count()) as p:
        results = p.map(run_command, commands)

    remeshed_quadrangulation_smooth_obj_path = "temporary/" + output_basename + f"_rem_p0_{target_quad_count}_quadrangulation_smooth.obj"
    mesh = trimesh.load(remeshed_quadrangulation_smooth_obj_path, force="mesh")
    mesh.merge_vertices(4)
    mesh.vertex_normals
    mesh.export(output_glb_path, file_type="glb")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GLB file.")
    parser.add_argument("input", type=str, help="Input GLB file path")
    parser.add_argument("output", type=str, help="Input GLB file path")
    args = parser.parse_args()

    glb_path = os.path.normpath(args.input)
    output_glb_path = os.path.basename(args.output)
    process_glb_file(glb_path, output_glb_path)
