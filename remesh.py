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

import trimesh
import numpy as np

def chamfer_distance(arr1, arr2):
    distance_1_to_2 = 0
    distance_2_to_1 = 0

    points1 = np.column_stack((arr1['X'], arr1['Y'], arr1['Z']))
    points2 = np.column_stack((arr2['X'], arr2['Y'], arr2['Z']))

    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(arr1) + len(arr2))

def compute_chamfer_distance(mesh1_path, mesh2_path):
    mesh1 = trimesh.load(mesh1_path)
    mesh2 = trimesh.load(mesh2_path)

    if isinstance(mesh1, trimesh.Scene):
        mesh1 = mesh1.dump(concatenate=True)
    if isinstance(mesh2, trimesh.Scene):
        mesh2 = mesh2.dump(concatenate=True)

    points1 = mesh1.sample(10000)
    points2 = mesh2.sample(10000)

    arr1 = np.core.records.fromarrays(points1.transpose(), names='X,Y,Z')
    arr2 = np.core.records.fromarrays(points2.transpose(), names='X,Y,Z')

    calculated_chamfer_distance = chamfer_distance(arr1, arr2)

    return calculated_chamfer_distance


def process_glb_file(glb_path, output_glb_path, setup_type):
    import numpy as np

    output_basename = os.path.splitext(os.path.basename(glb_path))[0]
    temp_dir = "temporary"
    source_file = os.path.normpath(os.path.join(temp_dir, f"{output_basename}.obj"))
    load_mesh_process_export(glb_path, source_file, "obj")

    target_quad_count = 1000
    source_file_second_stage = os.path.normpath(
        os.path.join(temp_dir, f"{output_basename}_rem_p0.obj")
    )

    config_file = (
        "basic_setup_Mechanical.txt"
        if setup_type == "mechanical"
        else "basic_setup_Organic.txt"
    )

    commands = [
        os.path.normpath(
            f"./thirdparty/quadwild_windows/quadwild {source_file} 2 thirdparty/quadwild_windows/config/prep_config/{config_file}"
        ),
        os.path.normpath(
            f"./thirdparty/quadwild_windows/quad_from_patches {source_file_second_stage} {target_quad_count} thirdparty/quadwild_windows/config/main_config/flow.txt"
        ),
    ]
    with Pool(os.cpu_count()) as p:
        for cmd in commands:
            p.apply(run_command, (cmd,))

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
    parser.add_argument(
        "setup_type",
        type=str,
        choices=["mechanical", "organic"],
        help="Type of setup (mechanical or organic)",
    )
    args = parser.parse_args()

    glb_path = os.path.normpath(args.input)
    output_glb_path = os.path.normpath(args.output)
    process_glb_file(glb_path, output_glb_path, args.setup_type)
    chamfer_distance = compute_chamfer_distance(glb_path, output_glb_path)
    print(f"Chamfer distance (Lower is better): {chamfer_distance}")