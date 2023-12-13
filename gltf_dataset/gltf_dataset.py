import torch
from torch.utils.data import Dataset
import os
import json
import trimesh
import numpy as np
import sys
import functools
import wandb


class GLTFMeshDataset(Dataset):
    def __init__(self, folder_path, use_wandb_tracking = False):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

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

    @staticmethod
    def compare_faces(face_a, face_b, vertices):
        for i in range(3):
            # Check if face indices are within the range of vertices list
            if face_a[i] >= len(vertices) or face_b[i] >= len(vertices):
                raise IndexError("Face index out of range")

            vertex_comparison = GLTFMeshDataset.compare_vertices(
                vertices[face_a[i]], vertices[face_b[i]]
            )
            if vertex_comparison != 0:
                return vertex_comparison

        for i in range(3):
            if face_a[i] < face_b[i]:
                return -1
            elif face_a[i] > face_b[i]:
                return 1

        return 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        scene = trimesh.load(file_path, force="Scene")
        vertex_indices = {}

        all_triangles = []
        all_faces = []
        all_vertices = []
        all_items = []

        for mesh_idx, (name, geometry) in enumerate(scene.geometry.items()):
            vertex_indices = {}

            try:
                geometry.apply_transform(scene.graph.get(name)[0])
            except Exception as e:
                pass

            vertices = [tuple(vertex) for vertex in geometry.vertices]
            vertex_indices.update({v: i for i, v in enumerate(vertices)})

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
            all_vertices.extend(vertices)  # Update all_vertices here
            all_items.append((idx, mesh_idx, name, len(vertices), len(faces)))

        all_faces.sort(
            key=functools.cmp_to_key(
                lambda a, b: GLTFMeshDataset.compare_faces(a, b, all_vertices)
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

        table = wandb.Table(data=all_items, columns=["idx", "index", "name", "vertex_count", "face_count"])

        wandb.log({
            "num_vertices": len(new_vertices),
            "num_faces": len(new_faces),
            "item_index": idx,
            "item_table": table,
        })

        return (torch.tensor(new_vertices, dtype=torch.float), torch.tensor(new_faces, dtype=torch.long))

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

        if not np.array_equal(json_data1[0], json_data1[0]):
            return False

        if not np.array_equal(json_data2[1], json_data2[1]):
            return False

        return True


if __name__ == "__main__":
    dataset = GLTFMeshDataset("unit_test")

    mesh_00 = dataset.__getitem__(0)
    with open("mesh_00.json", "wb") as f:
        f.write(json.dumps(mesh_00).encode())

    mesh_01 = dataset.__getitem__(1)
    with open("mesh_01.json", "wb") as f:
        f.write(json.dumps(mesh_01).encode())

    if GLTFMeshDataset.compare_json(mesh_00, mesh_01):
        print("JSON data 00 and 01 are the same.")
    else:
        print("JSON data 00 and 01 are different.")

    dataset.convert_to_glb(mesh_00, "unit_test/box_test_02.glb")
    dataset.convert_to_glb(mesh_01, "unit_test/box_test_03.glb")

    mesh_02 = dataset.__getitem__(2)
    with open("mesh_02.json", "wb") as f:
        f.write(json.dumps(mesh_02).encode())

    mesh_03 = dataset.__getitem__(3)
    with open("mesh_03.json", "wb") as f:
        f.write(json.dumps(mesh_03).encode())

    if GLTFMeshDataset.compare_json(mesh_02, mesh_03):
        print("JSON data 02 and 03 are the same.")
    else:
        print("JSON data 02 and 03 are different.")
