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
from run import load_and_process_files, generate_mesh_data, convert_to_glb
from meshgpt_pytorch.data import MeshDataset

class TestMeshDataset(unittest.TestCase):
    def setUp(self):
        self.augments = 3
        folder_path = "dataset/blockmesh_test"
        files = os.listdir(folder_path)
        supported_formats = (".glb", ".gltf")
        files = [
            file
            for file in os.listdir(folder_path)
            if os.path.splitext(file)[1] in supported_formats
        ]
        files = sorted(files)
        max_faces_allowed = 1365
        idx_to_file_idx = load_and_process_files(
            folder_path, supported_formats, max_faces_allowed
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = [
            generate_mesh_data(idx, idx_to_file_idx, files, folder_path, max_faces_allowed)
            for idx in range(len(idx_to_file_idx))
        ]
        dataset = MeshDataset(data)
        # dataset = MeshDataset.load("mesh_dataset.npz")
        dataset.generate_face_edges()
        self.dataset = dataset

    def test_mesh_augmentation(self):
        for i in range(self.augments):
            item = self.dataset.__getitem__(i)
            tensor1 = item["vertices"]
            tensor2 = item["faces"]
            tensor3 = item["face_edges"]
            str_item = item["text"]
            if all(
                isinstance(tensor, (torch.Tensor, np.ndarray))
                for tensor in [tensor1, tensor2, tensor3]
            ):
                vertices = tensor1.tolist()
                faces = tensor2.tolist()
                face_edges = tensor3.tolist()
                with open(
                    f"dataset/unit_augment/mesh_{str(i).zfill(2)}.json", "wb"
                ) as f:
                    f.write(json.dumps((vertices, faces, str_item)).encode())
                convert_to_glb(
                    (vertices, faces),
                    f"dataset/unit_augment/mesh_{str(i).zfill(2)}.glb",
                )
            else:
                print(f"Item {i} in the dataset does not contain valid tensors.")


if __name__ == "__main__":
    wandb.init(project="meshgpt-pytorch", config={})
    unittest.main()
