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


class MeshDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    @staticmethod
    def compare_faces(face_a, face_b, axis_order, vertices):
        for i in range(3):
            # Check if face indices are within the range of vertices list
            if face_a[i] >= len(vertices) or face_b[i] >= len(vertices):
                raise IndexError("Face index out of range")

            vertex_comparison = MeshDataset.compare_vertices(
                vertices[face_a[i]], vertices[face_b[i]], axis_order
            )
            if vertex_comparison != 0:
                return vertex_comparison

        for i in range(3):
            if face_a[i] < face_b[i]:
                return -1
            elif face_a[i] > face_b[i]:
                return 1

        return 0

    def filter_files(self):
        filtered_list = [
            file
            for file in self.file_list
            if file.endswith((".glb", ".gltf", ".ply", ".obj", ".stl"))
        ]
        return filtered_list

    def __len__(self):
        return len(self.filter_files())

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
    def compare_json(json_data1, json_data2):
        if len(json_data1) != len(json_data2):
            return False

        if not np.array_equal(json_data1[0], json_data2[0]):
            return False

        if not np.array_equal(json_data1[1], json_data2[1]):
            return False

        return True

    @staticmethod
    def compare_vertices(vertex_a, vertex_b, axis_order):
        # Compare vertices according to the defined axis order
        for axis in axis_order:
            if vertex_a[axis] < vertex_b[axis]:
                return -1
            elif vertex_a[axis] > vertex_b[axis]:
                return 1

        return 0


    def filter_files(self):
        filtered_list = [
            file
            for file in self.file_list
            if file.endswith((".glb", ".gltf", ".ply", ".obj", ".stl"))
        ]
        return filtered_list

    def __len__(self):
        return len(self.filter_files())

    def __getitem__(self, idx):
        files = self.filter_files()
        file_path = os.path.join(self.folder_path, files[idx])
        _, file_extension = os.path.splitext(file_path)

        axis_orders = {
            '.ply': [1, 2, 0],  # Y-Z-X
            '.stl': [0, 1, 2],  # X-Y-Z
            '.obj': [0, 2, 1],  # X-Z-Y
            '.glb': [1, 2, 0],  # Y-Z-X
            '.gltf': [1, 2, 0]  # Y-Z-X
        }

        # Get the axis order for the current file type
        axis_order = axis_orders.get(file_extension, [0, 1, 2])

        scene = trimesh.load(file_path, force="scene")
        vertex_indices = {}

        all_triangles = []
        all_faces = []
        all_vertices = []

        for mesh_idx, (name, geometry) in enumerate(scene.geometry.items()):
            vertex_indices = {}

            try:
                geometry.apply_transform(scene.graph.get(name)[0])
            except Exception as e:
                pass

            vertices = [tuple(vertex) for vertex in geometry.vertices]
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
            geometry_data = (vertices, faces)

            triangles = [tuple(vertex) for vertex in geometry.vertices]
            all_triangles.extend(triangles)
            all_faces.extend(faces)
            all_vertices.extend(vertices)

        all_faces.sort(
            key=functools.cmp_to_key(
                lambda a, b: MeshDataset.compare_faces(a, b, axis_order, all_vertices)
            )
        )

        new_vertices = []
        new_faces = []

        for face in all_faces:
            new_face = []
            for vertex_index in face:
                if all_vertices[vertex_index] not in new_vertices:
                    new_vertices.append(all_vertices[vertex_index])
                new_face.append(new_vertices.index(all_vertices[vertex_index]))
            new_faces.append(new_face)

        return (
            torch.tensor(new_vertices, dtype=torch.float),
            torch.tensor(new_faces, dtype=torch.long),
        )


if __name__ == "__main__":
    dataset = MeshDataset("unit_test")

    mesh_00 = [tensor.tolist() for tensor in dataset.__getitem__(0)]
    
    with open("unit_test/mesh_00.json", "wb") as f:
        f.write(json.dumps(mesh_00).encode())

    dataset.convert_to_glb(mesh_00, "unit_test/box_test_01.glb")

    for i in range(1, 5):
        mesh = [tensor.tolist() for tensor in dataset.__getitem__(i)]

        with open(f"unit_test/mesh_{str(i).zfill(2)}.json", "wb") as f:
            f.write(json.dumps(mesh).encode())

        if MeshDataset.compare_json(mesh_00, mesh):
            print(f"JSON data 00 and {str(i).zfill(2)} are the same.")
        else:
            print(f"JSON data 00 and {str(i).zfill(2)} are different.")
