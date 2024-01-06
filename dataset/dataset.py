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

from torch.utils.data import Dataset
import numpy as np
from meshgpt_pytorch import MeshAutoencoder, MeshTransformer

from meshgpt_pytorch.data import derive_face_edges_from_faces


# From Marcus.
class MeshDataset(Dataset):
    """
    A PyTorch Dataset to load and process mesh data.
    The `MeshDataset` provides functions to load mesh data from a file, embed text information, generate face edges, and generate codes.

    Attributes:
        data (list): A list of mesh data entries. Each entry is a dictionary containing the following keys:
            vertices (torch.Tensor): A tensor of vertices with shape (num_vertices, 3).
            faces (torch.Tensor): A tensor of faces with shape (num_faces, 3).
            text (str): A string containing the associated text information for the mesh.
            text_embeds (torch.Tensor): A tensor of text embeddings for the mesh.
            face_edges (torch.Tensor): A tensor of face edges with shape (num_faces, num_edges).
            codes (torch.Tensor): A tensor of codes generated from the mesh data.

    Example usage:

    ```
    data = [
        {'vertices': torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), 'faces': torch.tensor([[0, 1, 2]], dtype=torch.long), 'text': 'table'},
        {'vertices': torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32), 'faces': torch.tensor([[1, 2, 0]], dtype=torch.long), "text": "chair"},
    ]

    # Create a MeshDataset instance
    mesh_dataset = MeshDataset(data)

    # Save the MeshDataset to disk
    mesh_dataset.save('mesh_dataset.npz')

    # Load the MeshDataset from disk
    loaded_mesh_dataset = MeshDataset.load('mesh_dataset.npz')

    # Generate face edges so it doesn't need to be done every time during training
    dataset.generate_face_edges()
    ```
    """

    def __init__(self, data):
        self.data = data
        print(f"[MeshDataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def save(self, path):
        np.savez_compressed(path, self.data, allow_pickle=True)
        print(f"[MeshDataset] Saved {len(self.data)} entries at {path}")

    @classmethod
    def load(cls, path):
        loaded_data = np.load(path, allow_pickle=True)
        data = []
        for item in loaded_data["arr_0"]:
            data.append(item)
        print(f"[MeshDataset] Loaded {len(data)} entries")
        return cls(data)

    def generate_face_edges(self):
        for i in range(0, len(self.data)):
            item = self.data[i]
            item["face_edges"] = derive_face_edges_from_faces(item["faces"])

        desired_order = ["vertices", "faces", "face_edges", "texts"]
        self.data = [{key: d[key] for key in desired_order} for d in self.data]
        print(f"[MeshDataset] Generated face_edges for {len(self.data)} entries")

    def generate_codes(self, autoencoder: MeshAutoencoder):
        for i in range(0, len(self.data)):
            item = self.data[i]

            codes = autoencoder.tokenize(
                vertices=item["vertices"],
                faces=item["faces"],
                face_edges=item["face_edges"],
            )
            item["codes"] = codes

        print(f"[MeshDataset] Generated codes for {len(self.data)} entries")

    def embed_texts(self, transformer: MeshTransformer):
        unique_texts = set(item["texts"] for item in self.data)

        text_embeddings = transformer.embed_texts(list(unique_texts))
        print(f"[MeshDataset] Generated {len(text_embeddings)} text_embeddings")
        text_embedding_dict = dict(zip(unique_texts, text_embeddings))

        for item in self.data:
            text_value = item["texts"]
            item["text_embeds"] = text_embedding_dict.get(text_value, None)
            del item["texts"]


import unittest
import json
