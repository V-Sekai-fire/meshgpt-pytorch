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
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)
        self.supported_formats = (".glb", ".gltf")
        self.augments_per_item = 200
        self.seed = 42

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

    def filter_files(self):
        filtered_list = [
            file
            for file in self.file_list
            if file.endswith(self.supported_formats)
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
    def compare_json(json_data1, json_data2):
        if len(json_data1) != len(json_data2):
            return False

        if not np.array_equal(json_data1[0], json_data2[0]):
            return False

        if not np.array_equal(json_data1[1], json_data2[1]):
            return False

        return True

    @staticmethod
    def compare_vertices(vertex_a, vertex_b):
        # glTF uses right-handed coordinate system (Y-Z-X).
        # Y is up and is different from the meshgpt paper.
        if vertex_a[1] < vertex_b[1]:
            return -1
        elif vertex_a[1] > vertex_b[1]:
            return 1
        elif vertex_a[2] < vertex_b[2]:
            return -1
        elif vertex_a[2] > vertex_b[2]:
            return 1
        elif vertex_a[0] < vertex_b[0]:
            return -1
        elif vertex_a[0] > vertex_b[0]:
            return 1
        return 0

    def filter_files(self):
        filtered_list = [
            file
            for file in self.file_list
            if file.endswith(self.supported_formats)
        ]
        return filtered_list

    def __len__(self):
        return len(self.filter_files()) * self.augments_per_item

    def __getitem__(self, idx):
        files = self.filter_files()
        file_idx = idx // self.augments_per_item
        augment_idx = idx % self.augments_per_item
        file_path = os.path.join(self.folder_path, files[file_idx])

        print(f"File Path: {file_path} Augmentation Index: {augment_idx} Index: {idx}")

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

        for face in all_faces:
            new_face = []
            for vertex_index in face:
                if vertex_index not in vertex_map:
                    new_vertex = all_vertices[vertex_index]
                    new_vertices.append(new_vertex)
                    vertex_map[vertex_index] = len(new_vertices) - 1
                new_face.append(vertex_map[vertex_index])
            new_faces.append(new_face)

        random.seed(self.seed + idx * self.augments_per_item + idx)
        scale = random.uniform(0.8, 1.2)  # Uniform scaling
        rotation = R.from_euler('y', random.uniform(-180, 180), degrees=True)  # Random rotation around y-axis
        translation = np.array([random.uniform(-0.5, 0.5) for _ in [0, 2]])  # Random translation in x and z directions
        translation = np.pad(translation, (0, 1), 'constant', constant_values=(0,))

        new_vertices = [(np.dot(rotation.as_matrix(), np.array(v)*scale) + translation).tolist() for v in new_vertices]


        return (
            torch.tensor(new_vertices, dtype=torch.float),
            torch.tensor(new_faces, dtype=torch.long),
        )

if __name__ == "__main__":

    dataset = MeshDataset("unit_test")

    mesh_00 = [tensor.tolist() for tensor in dataset.__getitem__(0)]

    with open("unit_test/mesh_00.json", "wb") as f:
        f.write(json.dumps(mesh_00).encode())

    # for i in range(1, 10):
    #     mesh = [tensor.tolist() for tensor in dataset.__getitem__(i)]
    #     dataset.convert_to_glb(mesh, f"unit_test/mesh_{str(i).zfill(2)}.glb")

    for i in range(1, 2):
        mesh = [tensor.tolist() for tensor in dataset.__getitem__(i)]

        with open(f"unit_test/mesh_{str(i).zfill(2)}.json", "wb") as f:
            f.write(json.dumps(mesh).encode())

        if MeshDataset.compare_json(mesh_00, mesh):
            print(f"JSON data 00 and {str(i).zfill(2)} are the same.")
        else:
            print(f"JSON data 00 and {str(i).zfill(2)} are different.")