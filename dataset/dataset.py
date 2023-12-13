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
            if file.endswith((".glb", ".gltf", ".ply", ".obj", ".stl"))
        ]
        return filtered_list

    def __len__(self):
        return len(self.filter_files())

    def transform_axis_order(self, data, src_format, dest_format):
        axis_orders_to_gltf = {
            ".ply": [2, 1, 0],  # Assuming Y-Z-X
            ".stl": [0, 2, 1],  # Assuming X-Z-Y
            ".obj": [0, 2, 1],  # X-Y-Z
            ".glb": [0, 1, 2],  # X-Y-Z
            ".gltf": [0, 1, 2],  # X-Y-Z
        }
        src_order = axis_orders_to_gltf[src_format]

        return [data[i] for i in src_order]

    def convert_to_dest_axis(self, source_extension, vertex_position):
        forward_axis_conversion = {
            ".ply": 1,
            ".stl": -1,
            ".obj": 1,
            ".glb": 1,
            ".gltf": 1,
        }

        # Transform the axis order from source to destination
        dest_converted_vertex_position = self.transform_axis_order(np.array(vertex_position), source_extension, ".gltf")
        
        # Perform the operation on both lists
        dest_converted_vertex_position[2] *= forward_axis_conversion[source_extension]

        return dest_converted_vertex_position

    def __getitem__(self, idx):
        files = self.filter_files()
        file_path = os.path.join(self.folder_path, files[idx])
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

            vertices = []
            for vertex in geometry.vertices:
                converted_vertex = self.convert_to_dest_axis(file_extension, vertex)
                vertices.append(converted_vertex)
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
          
            # Calculate the normal of the first face
            v1 = np.array(vertices[1]) - np.array(vertices[0])
            v2 = np.array(vertices[2]) - np.array(vertices[0])
            normal = np.cross(v1, v2)

            # If the z-coordinate of the normal is positive, the winding order is CCW
            # Otherwise, it's CW
            if normal[2] > 0:
                print("Clockwise")
                faces = [face[::-1] for face in faces]
            else:
                print("Counter-clockwise")
            
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

        return (
            torch.tensor(new_vertices, dtype=torch.float),
            torch.tensor(new_faces, dtype=torch.long),
        )

import unittest

class TestMain(unittest.TestCase):

    def setUp(self):
        self.main = MeshDataset("unit_test")  # Assuming the class name is Main

    def test_convert_to_dest_axis(self):
        # Test case for .ply extension
        result = self.main.convert_to_dest_axis(".ply", [1, 2, 3])
        self.assertEqual(result, [3, 2, 1])

        # Test case for .stl extension
        result = self.main.convert_to_dest_axis(".stl", [1, 2, 3])
        self.assertEqual(result, [1, 3, -2])

        # Test case for .obj extension
        result = self.main.convert_to_dest_axis(".obj", [1, 2, 3])
        self.assertEqual(result, [1, 3, 2])

        # Test case for .glb extension
        result = self.main.convert_to_dest_axis(".glb", [1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

        # Test case for .gltf extension
        result = self.main.convert_to_dest_axis(".gltf", [1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

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
    unittest.main()
