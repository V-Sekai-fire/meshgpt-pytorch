import torch
from torch.utils.data import Dataset
import os
import json
import trimesh
import numpy as np
import sys
import functools
import wandb

from abc import abstractmethod
import os
import random
from scipy.spatial.transform import Rotation as R


class MeshDataset(Dataset):
    def __init__(self, folder_path, augments_per_item):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.supported_formats = (".glb", ".gltf")
        self.augments_per_item = augments_per_item

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

    def filter_files(self):
        filtered_list = [
            file for file in self.file_list if file.endswith(self.supported_formats)
        ]
        return filtered_list

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

    def __len__(self):
        return len(self.filter_files()) * self.augments_per_item

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
        files = self.filter_files()
        file_idx = idx // self.augments_per_item
        augment_idx = idx % self.augments_per_item
        file_path = os.path.join(self.folder_path, files[file_idx])

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

        new_vertices = []
        new_faces = []
        vertex_map = {}

        import math

        def calculate_angle(point, center):
            # Calculate the angle between the point and the horizontal line through the center
            return math.atan2(point[1] - center[1], point[0] - center[0])

        def sort_vertices_ccw(vertices):
            # Calculate the center of the vertices
            center = [
                sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(2)
            ]

            # Sort the vertices based on the angle each makes with the center
            return sorted(vertices, key=lambda vertex: -calculate_angle(vertex, center))

        def calculate_normal(face_vertices):
            # Calculate the normal of a face given its vertices
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

            # Convert indices to actual vertices
            new_face_vertices = [new_vertices[i] for i in new_face]

            # Calculate the original normal
            original_normal = calculate_normal(new_face_vertices)

            # Sort vertices in counter-clockwise order
            sorted_vertices = sort_vertices_ccw(new_face_vertices)

            # Calculate the new normal
            new_normal = calculate_normal(sorted_vertices)

            # If the direction of the normal has changed, reverse the order of the sorted vertices
            if np.dot(original_normal, new_normal) < 0:
                sorted_vertices = list(reversed(sorted_vertices))

            # Map sorted vertices back to their corresponding indices
            sorted_indices = [
                new_face[new_face_vertices.index(vertex)] for vertex in sorted_vertices
            ]

            new_faces.append(sorted_indices)

        return self.augment_mesh(
            (
                torch.tensor(new_vertices, dtype=torch.float),
                torch.tensor(new_faces, dtype=torch.long),
            ),
            self.augments_per_item,
            augment_idx,
        )


    def augment_mesh(self, base_mesh, augment_count, augment_idx):
        vertices = base_mesh[0]

        # Calculate the centroid of the object
        centroid = [
            sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3)
        ]

        # Translate the vertices so that the centroid is at the origin
        translated_vertices = [[v[i] - centroid[i] for i in range(3)] for v in vertices]

        # Calculate the maximum absolute value across all dimensions
        max_value = max(max(abs(v[i]) for v in translated_vertices) for i in range(3))

        # Generate a scale factor based on the maximum value
        scale = 1 / max_value

        # Scale the translated vertices
        scaled_vertices = [
            [v[i] * scale for i in range(3)] for v in translated_vertices
        ]

        # Generate a random rotation matrix
        rotation = R.from_euler("y", random.uniform(-180, 180), degrees=True)

        # Apply the transformations to each vertex of the object
        new_vertices = [
            (np.dot(rotation.as_matrix(), np.array(v))).tolist()
            for v in scaled_vertices
        ]

        # Translate the vertices back so that the centroid is at its original position
        final_vertices = [[v[i] + centroid[i] for i in range(3)] for v in new_vertices]

        return (
            torch.from_numpy(np.array(final_vertices, dtype=np.float32)),
            base_mesh[1],
        )



import unittest
import json


class TestMeshDataset(unittest.TestCase):
    def setUp(self):
        self.augments = 3
        self.dataset = MeshDataset("unit_test", self.augments)
        self.mesh_00 = [tensor.tolist() for tensor in self.dataset.__getitem__(0)]

    def test_mesh_augmentation(self):
        for i in range(self.augments):
            mesh = [tensor.tolist() for tensor in self.dataset.__getitem__(i)]
            with open(f"unit_augment/mesh_{str(i).zfill(2)}.json", "wb") as f:
                f.write(json.dumps(mesh).encode())
            self.dataset.convert_to_glb(
                mesh, f"unit_augment/mesh_{str(i).zfill(2)}.glb"
            )

    def test_json_comparison(self):
        i = 0
        mesh = [tensor.tolist() for tensor in self.dataset.__getitem__(i)]
        self.assertEqual(
            MeshDataset.compare_json(self.mesh_00, mesh),
            True,
            f"JSON data 00 and {str(i).zfill(2)} are different.",
        )


if __name__ == "__main__":
    unittest.main()
