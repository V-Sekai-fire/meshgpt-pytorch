import torch
from torch.utils.data import Dataset
from meshgpt_pytorch.data import derive_face_edges_from_faces
import os
import json
import trimesh
import numpy as np
import sys
import functools
import wandb

from functools import lru_cache

from abc import abstractmethod
import os
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

import math


class MeshDataset(Dataset):
    def __init__(self, folder_path):
        self.files = []
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.supported_formats = (".glb", ".gltf")
        self.seed = 42
        self.total_augments = 0
        self.max_faces = 1365
        self.idx_to_file_idx = []

        chunk_counter = 0  # Keep track of total number of chunks
        self.files = self.filter_files()
        for file_name in self.files:
            file_path = os.path.join(self.folder_path, file_name)
            scene = trimesh.load(file_path, force="scene")
            total_faces_in_file = 0
            for _, geometry in scene.geometry.items():
                try:
                    geometry.apply_transform(scene.graph.get(_)[0])
                except Exception as e:
                    pass

                num_faces = len(geometry.faces)
                total_faces_in_file += num_faces

            # Calculate the number of chunks for this file
            num_chunks = math.ceil(total_faces_in_file / self.max_faces)
            self.total_augments += num_chunks
            self.log_mesh_details(file_name, total_faces_in_file)

            file_idx = self.filter_files().index(file_name)
            # Add entries to the lookup table
            for i in range(num_chunks):
                self.idx_to_file_idx.append(file_idx)
            chunk_counter += 1

    def __len__(self):
        return self.total_augments

    def filter_files(self):
        filtered_list = [
            file for file in self.file_list if file.endswith(self.supported_formats)
        ]
        return filtered_list

    def log_mesh_details(self, file_name, total_faces_in_file):
        wandb.log(
            {
                "file_name": file_name,
                "total_faces_in_file": total_faces_in_file,
                "max_faces_allowed": self.get_max_face_count(),
            }
        )

    def __len__(self):
        return self.total_augments

    def get_max_face_count(self):
        max_faces = 0
        files = self.filter_files()
        files = sorted(self.filter_files())
        for file in files:
            file_path = os.path.join(self.folder_path, file)
            scene = trimesh.load(file_path, force="scene")
            total_faces_in_file = 0
            for _, geometry in scene.geometry.items():
                try:
                    geometry.apply_transform(scene.graph.get(_)[0])
                except Exception as e:
                    pass

                num_faces = len(geometry.faces)
                total_faces_in_file += num_faces

            if total_faces_in_file > max_faces:
                max_faces = total_faces_in_file

        return max_faces

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def compare_json(json_data1, json_data2):
        if len(json_data1) != len(json_data2):
            return False

        if not np.array_equal(json_data1[0], json_data2[0]):
            return False

        if not np.array_equal(json_data1[1], json_data2[1]):
            return False

        return True

    @staticmethod
    def snake_to_sentence_case(snake_str):
        components = snake_str.split("_")
        return " ".join(word.capitalize() for word in components)

    @staticmethod
    def compare_vertices(vertex_a, vertex_b):
        # glTF uses right-handed coordinate system (Y-Z-X).
        # Y is up and is different from the meshgpt paper.
        for i in [1, 2, 0]:  # Compare Y, then Z, then X
            if vertex_a[i] < vertex_b[i]:
                return -1
            elif vertex_a[i] > vertex_b[i]:
                return 1
        return 0  # If all coordinates are equal

    @staticmethod
    def compare_faces(face_a, face_b, vertices):
        for i in range(3):
            # Check if face indices are within the range of vertices list
            if face_a[i] >= len(vertices) or face_b[i] >= len(vertices):
                raise IndexError("Face index out of range")

            vertex_comparison = MeshDataset.compare_vertices(
                vertices[face_a[i]], vertices[face_b[i]]
            )
            if vertex_comparison != 0:
                return vertex_comparison

        return 0

    def __getitem__(self, idx):
        file_idx = self.idx_to_file_idx[idx]

        all_faces, all_vertices, num_chunks = self.load_and_process_scene(file_idx)

        file_name = self.files[file_idx]
        file_name_without_ext = os.path.splitext(file_name)[0]
        text = MeshDataset.snake_to_sentence_case(file_name_without_ext)

        all_vertices_np = np.array(all_vertices)
        centroids = self.generate_face_centroids(all_faces, all_vertices, num_chunks)

        centroid_idx = idx % len(centroids)
        centroid = centroids[centroid_idx]

        vertices_np = np.array(all_vertices)
        faces_np = np.array(all_faces)

        kdtree = KDTree(vertices_np)

        selected_faces = self.extract_mesh_with_max_number_of_faces(
            kdtree, centroid, vertices_np, all_faces
        )

        new_vertices, new_faces = self.create_new_vertices_and_faces(
            selected_faces, all_vertices
        )

        faces = torch.from_numpy(np.array(new_faces))

        vertices, faces = self.center_mesh(
            (
                torch.tensor(new_vertices, dtype=torch.float),
                faces,
            ),
        )

        face_edges = derive_face_edges_from_faces(faces)

        return vertices, faces, face_edges, text

    def load_and_process_scene(self, file_idx):
        file_path = os.path.join(self.folder_path, self.files[file_idx])
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
            key=functools.cmp_to_key(
                lambda a, b: MeshDataset.compare_faces(a, b, all_vertices)
            )
        )

        total_faces_in_file = len(all_faces)
        num_chunks = math.ceil(total_faces_in_file / self.max_faces)

        return all_faces, all_vertices, num_chunks

    def create_new_vertices_and_faces(self, all_faces, all_vertices):
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

    def generate_face_centroids(self, all_faces, all_vertices, num_chunk):
        """
        Generate a list of centroids for each face in the mesh and find the furthest away points.

        Parameters:
        all_faces (list): The faces of the mesh.
        all_vertices (list): The vertices of the mesh.
        num_chunk (int): The number of centroids to generate.

        Returns:
        ndarray: An array of centroids that are furthest apart.
        """

        # Generate the centroids by averaging the vertices of each face
        centroids = []
        for face in all_faces:
            face_vertices = [all_vertices[vertex_idx] for vertex_idx in face]
            centroid = np.mean(face_vertices, axis=0)
            centroids.append(centroid)

        # Use K-means clustering to find the furthest away points
        kmeans = KMeans(n_clusters=num_chunk)
        kmeans.fit(centroids)
        furthest_points = kmeans.cluster_centers_

        return furthest_points

    def extract_mesh_with_max_number_of_faces(
        self, kdtree, random_point, vertices_np, all_faces
    ):
        num_neighbours = min(self.max_faces, len(vertices_np))

        distances, indices = kdtree.query(random_point, k=num_neighbours)

        selected_vertex_indices = set(indices.flatten())
        selected_faces = [
            face
            for face in all_faces
            if any(vertex in selected_vertex_indices for vertex in face)
        ]

        return np.array(selected_faces)

    def center_mesh(self, base_mesh):
        vertices = base_mesh[0]

        # Calculate the centroid of the object
        centroid = [
            sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3)
        ]

        # Translate the vertices so that the centroid is at the origin
        translated_vertices = [[v[i] - centroid[i] for i in range(3)] for v in vertices]

        # Normalize uniformly to fill [-1, 1]
        min_vals = np.min(translated_vertices, axis=0)
        max_vals = np.max(translated_vertices, axis=0)

        # Calculate the maximum absolute value among all vertices
        max_abs_val = max(np.max(np.abs(min_vals)), np.max(np.abs(max_vals)))

        # Calculate the scale factor as the reciprocal of the maximum absolute value
        scale_factor = 1 / max_abs_val if max_abs_val != 0 else 1

        # Apply the normalization
        final_vertices = [
            [component * scale_factor for component in v] for v in translated_vertices
        ]

        return (
            torch.from_numpy(np.array(final_vertices, dtype=np.float32)),
            torch.from_numpy(np.array(base_mesh[1], dtype=np.int64)),
        )


import unittest
import json


class TestMeshDataset(unittest.TestCase):
    def setUp(self):
        self.augments = 20
        self.dataset = MeshDataset("blockmesh_test")

    def test_mesh_augmentation(self):
        for i in range(self.augments):
            item = self.dataset.__getitem__(i)
            # Check if the item is a tuple of (tensor, tensor, tensor, string)
            if isinstance(item, tuple) and len(item) == 4:
                tensor1, tensor2, tensor3, str_item = item
                if all(
                    isinstance(tensor, (torch.Tensor, np.ndarray))
                    for tensor in [tensor1, tensor2, tensor3]
                ):
                    vertices = tensor1.tolist()
                    faces = tensor2.tolist()
                    face_edges = tensor3.tolist()
                    with open(f"unit_augment/mesh_{str(i).zfill(2)}.json", "wb") as f:
                        f.write(json.dumps((vertices, faces, str_item)).encode())
                    self.dataset.convert_to_glb(
                        (vertices, faces), f"unit_augment/mesh_{str(i).zfill(2)}.glb"
                    )
                else:
                    print(f"Item {i} in the dataset does not contain valid tensors.")
            else:
                print(f"Item {i} in the dataset is not a valid tuple.")


if __name__ == "__main__":
    wandb.init(project="meshgpt-pytorch", config={})
    unittest.main()
