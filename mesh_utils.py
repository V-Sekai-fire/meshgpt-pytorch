import argparse
import torch
import wandb
import os
from meshgpt_pytorch import MeshAutoencoder, MeshAutoencoderTrainer
from datetime import datetime
import re
import json
import numpy as np
import wandb
import functools
from meshgpt_pytorch import MeshTransformer
import trimesh
import math
from sklearn.cluster import KMeans
import unittest
from scipy.spatial import KDTree
from meshgpt_pytorch.data import MeshDataset


def convert_to_glb(json_data, output_file_path):
    scene = trimesh.Scene()
    vertices = np.array(json_data[0])
    faces = np.array(json_data[1])
    if faces.max() >= len(vertices):
        raise ValueError(
            f"Face index {faces.max()} exceeds number of vertices {len(vertices)}"
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.vertex_normals
    scene.add_geometry(mesh)
    with open(output_file_path, "wb") as f:
        f.write(scene.export(file_type="glb"))


def convert_to_obj(json_data, output_file_path):
    scene = trimesh.Scene()
    vertices = np.array(json_data[0])
    faces = np.array(json_data[1])
    if faces.max() >= len(vertices):
        raise ValueError(
            f"Face index {faces.max()} exceeds number of vertices {len(vertices)}"
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.vertex_normals
    scene.add_geometry(mesh)
    with open(output_file_path, "w") as f:
        f.write(scene.export(file_type="obj"))


def compare_json(json_data1, json_data2):
    if len(json_data1) != len(json_data2):
        return False

    if not np.array_equal(json_data1[0], json_data2[0]):
        return False

    if not np.array_equal(json_data1[1], json_data2[1]):
        return False

    return True


def snake_to_sentence_case(snake_str):
    components = snake_str.split("_")
    return " ".join(word.capitalize() for word in components)


def compare_vertices(vertex_a, vertex_b):
    # glTF uses right-handed coordinate system (Y-Z-X).
    # Y is up and is different from the meshgpt paper.
    for i in [1, 2, 0]:  # Compare Y, then Z, then X
        if vertex_a[i] < vertex_b[i]:
            return -1
        elif vertex_a[i] > vertex_b[i]:
            return 1
    return 0  # If all coordinates are equal


def compare_faces(face_a, face_b, vertices):
    for i in range(3):
        if face_a[i] >= len(vertices) or face_b[i] >= len(vertices):
            raise IndexError("Face index out of range")

        vertex_comparison = compare_vertices(vertices[face_a[i]], vertices[face_b[i]])
        if vertex_comparison != 0:
            return vertex_comparison

    return 0


def load_and_process_scene(file_idx, files, folder_path, max_face_count):
    file_path = os.path.join(folder_path, files[file_idx])
    _, file_extension = os.path.splitext(file_path)
    scene = trimesh.load(file_path, force="scene")

    all_triangles = []
    all_faces = []
    all_vertices = []

    for mesh_idx, (name, geometry) in enumerate(scene.geometry.items()):
        vertex_indices = {}

        try:
            geometry.apply_transform(scene.graph.get(name)[0])
        except Exception as e:
            pass

        vertices = geometry.vertices
        vertices = [tuple(v) for v in vertices]
        vertex_indices.update({v: i for i, v in enumerate(vertices)})

        geometry.vertices = np.array(vertices)

        offset = len(all_vertices)

        faces = [
            [
                vertex_indices[tuple(geometry.vertices[vertex])] + offset
                for vertex in face
            ]
            for face in geometry.faces
        ]

        faces = [[vertex for vertex in face] for face in faces]

        all_faces.extend(faces)
        all_vertices.extend(vertices)

    all_faces.sort(
        key=functools.cmp_to_key(lambda a, b: compare_faces(a, b, all_vertices))
    )

    total_faces_in_file = len(all_faces)
    num_chunks = math.ceil(total_faces_in_file / max_face_count)

    return all_faces, all_vertices, num_chunks


def create_new_vertices_and_faces(all_faces, all_vertices):
    new_vertices = []
    new_faces = []
    vertex_map = {}

    import math

    def calculate_angle(point, center):
        return math.atan2(point[1] - center[1], point[0] - center[0])

    def sort_vertices_ccw(vertices):
        center = [
            sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(2)
        ]

        return sorted(vertices, key=lambda vertex: -calculate_angle(vertex, center))

    def calculate_normal(face_vertices):
        v1 = np.array(face_vertices[1]) - np.array(face_vertices[0])
        v2 = np.array(face_vertices[2]) - np.array(face_vertices[0])
        return np.cross(v1, v2)

    for face in all_faces:
        new_face = []
        for vertex_index in face:
            if vertex_index not in vertex_map:
                new_vertex = all_vertices[vertex_index]
                new_vertices.append(new_vertex)
                vertex_map[vertex_index] = len(new_vertices) - 1
            new_face.append(vertex_map[vertex_index])

        new_face_vertices = [new_vertices[i] for i in new_face]

        original_normal = calculate_normal(new_face_vertices)

        sorted_vertices = sort_vertices_ccw(new_face_vertices)

        new_normal = calculate_normal(sorted_vertices)

        if np.dot(original_normal, new_normal) < 0:
            sorted_vertices = list(reversed(sorted_vertices))

        sorted_indices = [
            new_face[new_face_vertices.index(vertex)] for vertex in sorted_vertices
        ]

        new_faces.append(sorted_indices)

    return new_vertices, new_faces


def extract_mesh_with_max_number_of_faces(
    kdtree, random_point, vertices_np, all_faces, max_faces
):
    num_neighbours = min(max_faces, len(vertices_np))

    distances, indices = kdtree.query(random_point, k=num_neighbours)

    selected_vertex_indices = set(indices.flatten())
    selected_faces = [
        face
        for face in all_faces
        if any(vertex in selected_vertex_indices for vertex in face)
    ]

    return np.array(selected_faces)


def center_mesh(base_mesh):
    vertices = base_mesh[0]

    centroid = [sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3)]

    translated_vertices = [[v[i] - centroid[i] for i in range(3)] for v in vertices]

    min_vals = np.min(translated_vertices, axis=0)
    max_vals = np.max(translated_vertices, axis=0)

    max_abs_val = max(np.max(np.abs(min_vals)), np.max(np.abs(max_vals)))

    scale_factor = 1 / max_abs_val if max_abs_val != 0 else 1

    final_vertices = [
        [component * scale_factor for component in v] for v in translated_vertices
    ]

    return (
        torch.from_numpy(np.array(final_vertices, dtype=np.float32)),
        torch.from_numpy(np.array(base_mesh[1], dtype=np.int64)),
    )


def filter_files(file_list, supported_formats):
    filtered_list = [file for file in file_list if file.endswith(supported_formats)]
    return filtered_list


def log_mesh_details(file_name, total_faces_in_file, max_face_count):
    wandb.log(
        {
            "file_name": file_name,
            "total_faces_in_file": total_faces_in_file,
            "max_faces_allowed": max_face_count,
        }
    )


def load_and_process_files(folder_path, supported_formats, max_faces_allowed):
    file_list = os.listdir(folder_path)
    files = filter_files(file_list, supported_formats)
    idx_to_file_idx = []

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        scene = trimesh.load(file_path, force="scene")
        total_faces_in_file = process_scene(scene)

        num_chunks = math.ceil(total_faces_in_file / max_faces_allowed)
        file_idx = files.index(file_name)
        idx_to_file_idx.extend([file_idx] * num_chunks)

    return idx_to_file_idx


def process_scene(scene):
    total_faces_in_file = 0
    for _, geometry in scene.geometry.items():
        try:
            geometry.apply_transform(scene.graph.get(_)[0])
        except Exception as e:
            pass

        num_faces = len(geometry.faces)
        total_faces_in_file += num_faces

    return total_faces_in_file


def generate_mesh_data(idx, idx_to_file_idx, files, folder_path, max_faces_allowed):
    file_idx = idx_to_file_idx[idx]
    file_name = files[file_idx]
    all_faces, all_vertices, num_chunks = load_and_process_scene(
        file_idx, files, folder_path, max_faces_allowed
    )

    file_name_without_ext = os.path.splitext(file_name)[0]
    text = snake_to_sentence_case(file_name_without_ext)

    centroids = []
    for face in all_faces:
        face_vertices = [all_vertices[vertex_idx] for vertex_idx in face]
        centroid = np.mean(face_vertices, axis=0)
        centroids.append(centroid)

    # Ensure the number of clusters is less than both num_chunks and max_faces_allowed
    num_clusters = min(num_chunks, max_faces_allowed - 1)

    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
    kmeans.fit(centroids)
    centroids = kmeans.cluster_centers_

    centroid_idx = idx % len(centroids)
    centroid = centroids[centroid_idx]

    vertices_np = np.array(all_vertices)
    faces_np = np.array(all_faces)

    kdtree = KDTree(vertices_np)
    selected_faces = extract_mesh_with_max_number_of_faces(
        kdtree, centroid, vertices_np, all_faces, max_faces_allowed
    )

    new_vertices, new_faces = create_new_vertices_and_faces(
        selected_faces, all_vertices
    )
    faces = torch.from_numpy(np.array(new_faces))

    vertices, faces = center_mesh(
        (
            torch.tensor(new_vertices, dtype=torch.float),
            faces,
        )
    )
    return {
        "vertices": vertices,
        "faces": faces,
        "text": text,
    }


def train_autoencoder(run, dataset, autoencoder):
    trainer = MeshAutoencoderTrainer(
        autoencoder,
        num_train_steps=2000,
        dataset=dataset,
        batch_size=run.config.batch_size,
        grad_accum_every=run.config.grad_accum_every,
        learning_rate=run.config.autoencoder_learning_rate,
        use_wandb_tracking=True,
    )
    trainer.train()

    current_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trainer.save(
        f"checkpoints/mesh_autoencoder_{os.path.basename(run.config.dataset_directory)}_{current_time}.pt"
    )

    return autoencoder


from datetime import datetime
from meshgpt_pytorch import MeshTransformer, MeshTransformerTrainer


def train_transformer(autoencoder, run, dataset, device, seq_len):
    transformer = MeshTransformer(
        autoencoder,
        max_seq_len=seq_len,
        dim=run.config.dim,
        condition_on_text=True,
    ).to(device)

    transformer_trainer = MeshTransformerTrainer(
        transformer,
        num_train_steps=2000,
        dataset=dataset,
        batch_size=wandb.config.batch_size,
        grad_accum_every=wandb.config.grad_accum_every,
        warmup_steps=500,
        learning_rate=wandb.config.transformer_learning_rate,
        use_wandb_tracking=True,
    )
    transformer_trainer.train()

    current_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    transformer_trainer.save(
        f"checkpoints/mesh_transformer_{os.path.basename(run.config.dataset_directory)}_{current_time}.pt"
    )
    return transformer


def process_mesh_data(run, device, transformer, texts):
    codes = transformer.generate(return_codes=True, texts=texts)

    transformer.autoencoder.eval()

    continuous_coors = transformer.autoencoder.decode_from_codes_to_faces(codes)[0]

    continuous_coors_list = [np_array.tolist() for np_array in continuous_coors]

    flat_list = [item for sublist in continuous_coors_list for item in sublist]

    vertices = [vertex for sublist in flat_list for vertex in sublist]

    faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]

    convert_to_glb((vertices, faces), "output.glb")
    convert_to_obj((vertices, faces), "output.obj")

    def encode_to_pua(codes):
        flat_codes = [
            item for sublist in codes for subsublist in sublist for item in subsublist
        ]
        return "".join(chr(code + 0xF0000) for code in flat_codes)

    encoded_codes = encode_to_pua(codes.cpu().tolist())

    with open("output.obj", "r") as file:
        obj_contents = file.read()

    new_data = [
        [
            {
                "role": "system",
                "content": "This assistant can understand 3D models using the meshgpt-pytorch Unicode plane 15 codebook for 16384 triangles and the .obj 3d format.",
            },
            {"role": "user", "content": f"{encoded_codes}"},
            {"role": "assistant", "content": f"{obj_contents}"},
        ]
    ]

    data = []
    try:
        with open("chatml.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("The file 'chatml.jsonl' does not exist.")

    data = new_data + data

    with open("chatml.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
