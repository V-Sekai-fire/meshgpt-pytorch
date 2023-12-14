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
        self.augments_per_item = 4000
        self.seed = 42

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
    def compare_json(json_data1, json_data2):
        if len(json_data1) != len(json_data2):
            return False

        if not np.array_equal(json_data1[0], json_data2[0]):
            return False

        if not np.array_equal(json_data1[1], json_data2[1]):
            return False

        return True

    def filter_files(self):
        filtered_list = [
            file for file in self.file_list if file.endswith(self.supported_formats)
        ]
        return filtered_list

    def __len__(self):
        return len(self.filter_files()) * self.augments_per_item

    def augment_mesh(self, base_mesh, augment_count, augment_idx):
        # Set the random seed for reproducibility
        random.seed(self.seed + augment_count * augment_idx + augment_idx)

        # Generate a random scale factor
        scale = random.uniform(0.8, 1.2)

        # Generate a random translation vector with small values within [-1, 1] bounds
        translation = [random.uniform(-0.03, 0.03) for _ in range(3)]

        vertices = base_mesh[0]

        # Calculate the centroid of the object
        centroid = [
            sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3)
        ]

        # Translate the vertices so that the centroid is at the origin
        translated_vertices = [[v[i] - centroid[i] for i in range(3)] for v in vertices]

        # Scale the translated vertices
        scaled_vertices = [
            [v[i] * scale for i in range(3)] for v in translated_vertices
        ]

        # Apply the random jitter (translation) to the scaled vertices
        jittered_vertices = [
            [v[i] + translation[i] for i in range(3)] for v in scaled_vertices
        ]

        # Translate the jittered vertices back so that the centroid is at its approximate original position
        final_vertices = [[v[i] + centroid[i] for i in range(3)] for v in jittered_vertices]

        # Normalize uniformly to fill [-1, 1]
        min_vals = np.min(final_vertices, axis=0)
        max_vals = np.max(final_vertices, axis=0)
        max_range = np.max(max_vals - min_vals) / 2
        final_vertices = [[(component - c) / max_range for component, c in zip(v, centroid)] for v in final_vertices]

        return (
            torch.from_numpy(np.array(final_vertices, dtype=np.float32)),
            base_mesh[1],
        )

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

        # Sort all the vertices based on their y-coordinate, then z-coordinate, then x-coordinate
        sorted_vertices = sorted(range(len(all_vertices)), key=lambda k: (all_vertices[k][1], all_vertices[k][2], all_vertices[k][0]))

        # Create a map from old vertex index to new one
        vertex_map = {old: new for new, old in enumerate(sorted_vertices)}

        # Now create new_faces with updated vertex indices
        new_faces = []
        for face in all_faces:
            new_face = [vertex_map[vertex_index] for vertex_index in face]
            
            # Cyclically permute indices to place the lowest index first
            min_index_position = new_face.index(min(new_face))
            new_face = new_face[min_index_position:] + new_face[:min_index_position]
            
            new_faces.append(new_face)

        return self.augment_mesh(
            (
                torch.tensor([all_vertices[i] for i in sorted_vertices], dtype=torch.float),
                torch.tensor(new_faces, dtype=torch.long),
            ),
            self.augments_per_item,
            augment_idx,
        )



if __name__ == "__main__":
    dataset = MeshDataset("unit_test")

    mesh_00 = [tensor.tolist() for tensor in dataset.__getitem__(0)]

    with open("unit_test/mesh_00.json", "wb") as f:
        f.write(json.dumps(mesh_00).encode())

    for i in range(0, 10):
        mesh = [tensor.tolist() for tensor in dataset.__getitem__(i)]
        with open(f"unit_augment/mesh_{str(i).zfill(2)}.json", "wb") as f:
            f.write(json.dumps(mesh).encode())
        dataset.convert_to_glb(mesh, f"unit_augment/mesh_{str(i).zfill(2)}.glb")

    for i in range(1, 2):
        mesh = [tensor.tolist() for tensor in dataset.__getitem__(i)]

        with open(f"unit_test/mesh_{str(i).zfill(2)}.json", "wb") as f:
            f.write(json.dumps(mesh).encode())

        if MeshDataset.compare_json(mesh_00, mesh):
            print(f"JSON data 00 and {str(i).zfill(2)} are the same.")
        else:
            print(f"JSON data 00 and {str(i).zfill(2)} are different.")
