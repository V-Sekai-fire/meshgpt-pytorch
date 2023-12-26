import argparse
import os
from multiprocessing import Pool


def run_command(cmd):
    import subprocess

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    print(f"Command: {cmd}")
    print(f"Stdout: {stdout.decode('utf-8')}")
    print(f"Stderr: {stderr.decode('utf-8')}")
    return stdout, stderr


def load_mesh_process_export(file_path, output_path, file_type):
    import trimesh

    mesh = trimesh.load(file_path, force="mesh")
    mesh.merge_vertices(4)
    mesh.vertex_normals
    mesh.export(output_path, file_type=file_type)


def process_glb_file(glb_path, output_glb_path):
    import numpy as np

    output_basename = os.path.splitext(os.path.basename(glb_path))[0]
    temp_dir = "temporary"
    source_file = os.path.normpath(os.path.join(temp_dir, f"{output_basename}.obj"))
    load_mesh_process_export(glb_path, source_file, "obj")

    target_quad_count = 1000
    source_file_second_stage = os.path.normpath(
        os.path.join(temp_dir, f"{output_basename}_rem_p0.obj")
    )
    commands = [
        os.path.normpath(
            f"./thirdparty/quadwild_windows/quadwild {source_file} 2 thirdparty/quadwild_windows/config/prep_config/basic_setup_Mechanical.txt"
        ),
        os.path.normpath(
            f"./thirdparty/quadwild_windows/quad_from_patches {source_file_second_stage} {target_quad_count} thirdparty/quadwild_windows/config/main_config/flow.txt"
        ),
    ]
    with Pool(os.cpu_count()) as p:
        results = p.map(run_command, commands)

    remeshed_quadrangulation_smooth_obj_path = os.path.join(
        temp_dir,
        f"{output_basename}_rem_p0_{target_quad_count}_quadrangulation_smooth.obj",
    )
    load_mesh_process_export(
        remeshed_quadrangulation_smooth_obj_path, output_glb_path, "glb"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GLB file.")
    parser.add_argument("input", type=str, help="Input GLB file path")
    parser.add_argument("output", type=str, help="Output GLB file path")
    args = parser.parse_args()

    glb_path = os.path.normpath(args.input)
    output_glb_path = os.path.normpath(args.output)
    process_glb_file(glb_path, output_glb_path)
