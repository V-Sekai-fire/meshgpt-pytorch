from pathlib import Path
from functools import partial
import torch
from torch import Tensor
from torch import is_tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from numpy.lib.format import open_memmap

from einops import rearrange, reduce

from beartype import beartype
from beartype.typing import Tuple, List, Union, Optional, Callable, Dict, Callable

from torchtyping import TensorType

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

from meshgpt_pytorch import MeshAutoencoder, MeshTransformer

# helper fn

def exists(v):
    return v is not None

def identity(t):
    return t

# constants

Vertices = TensorType['nv', 3, float]   # 3 coordinates
Faces = TensorType['nf', 3, int]        # 3 vertices

# decorator for auto-caching texts -> text embeds

# you would decorate your Dataset class with this
# and then change your `data_kwargs = ["text_embeds", "vertices", "faces"]`

@beartype
def cache_text_embeds_for_dataset(
    embed_texts_fn: Callable[[List[str]], Tensor],
    max_text_len: int,
    cache_path: str = './text_embed_cache'
):
    # create path to cache folder

    path = Path(cache_path)
    path.mkdir(exist_ok = True, parents = True)
    assert path.is_dir()

    # global memmap handles

    text_embed_cache = None
    is_cached = None

    # cache function

    def get_maybe_cached_text_embed(
        idx: int,
        dataset_len: int,
        text: str,
        memmap_file_mode = 'w+'
    ):
        nonlocal text_embed_cache
        nonlocal is_cached

        # init cache on first call

        if not exists(text_embed_cache):
            test_embed = embed_texts_fn(['test'])
            feat_dim = test_embed.shape[-1]
            shape = (dataset_len, max_text_len, feat_dim)

            text_embed_cache = open_memmap(str(path / 'cache.text_embed.memmap.npy'), mode = memmap_file_mode, dtype = 'float32', shape = shape)
            is_cached = open_memmap(str(path / 'cache.is_cached.memmap.npy'), mode = memmap_file_mode, dtype = 'bool', shape = (dataset_len,))

        # determine whether to fetch from cache
        # or call text model

        if is_cached[idx]:
            text_embed = torch.from_numpy(text_embed_cache[idx])
        else:
            # cache

            text_embed = get_text_embed(text)
            text_embed_len = text_embed.shape[0]

            if text_embed_len > max_text_len:
                text_embed = text_embed[:max_text_len]
            elif text_embed_len < max_text_len:
                text_embed = F.pad(text_embed, (0, 0, 0, max_text_len - text_embed_len))

            is_cached[idx] = True
            text_embed_cache[idx] = text_embed.cpu().numpy()

        mask = ~reduce(text_embed == 0, 'n d -> n', 'all')
        return text_embed[mask]

    # get text embedding

    def get_text_embed(text: str):
        text_embeds = embed_texts_fn([text])
        return text_embeds[0]

    # inner function

    def inner(dataset_klass):
        assert issubclass(dataset_klass, Dataset)

        orig_init = dataset_klass.__init__
        orig_get_item = dataset_klass.__getitem__

        def __init__(
            self,
            *args,
            cache_memmap_file_mode = 'w+',
            **kwargs
        ):
            orig_init(self, *args, **kwargs)

            self._cache_memmap_file_mode = cache_memmap_file_mode

            if hasattr(self, 'data_kwargs'):
                self.data_kwargs = [('text_embeds' if data_kwarg == 'texts' else data_kwarg) for data_kwarg in self.data_kwargs]

        def __getitem__(self, idx):
            items = orig_get_item(self, idx)

            get_text_embed_ = partial(get_maybe_cached_text_embed, idx, len(self), memmap_file_mode = self._cache_memmap_file_mode)

            if isinstance(items, dict):
                if 'texts' in items:
                    text_embed = get_text_embed_(items['texts'])
                    items['text_embeds'] = text_embed
                    del items['texts']

            elif isinstance(items, tuple):
                new_items = []

                for maybe_text in items:
                    if not isinstance(maybe_text, str):
                        new_items.append(maybe_text)
                        continue

                    new_items.append(get_text_embed_(maybe_text))

                items = tuple(new_items)

            return items

        dataset_klass.__init__ = __init__
        dataset_klass.__getitem__ = __getitem__

        return dataset_klass

    return inner


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
        print(f"[MeshDataset] Created from {len(self.data)} entrys")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    def save(self, path):
        np.savez_compressed(path, self.data, allow_pickle=True)
        print(f"[MeshDataset] Saved {len(self.data)} entrys at {path}")

    @classmethod
    def load(cls, path):
        loaded_data = np.load(path, allow_pickle=True)
        data = []
        for item in loaded_data["arr_0"]:
            data.append(item)
        print(f"[MeshDataset] Loaded {len(data)} entrys")
        return cls(data)

    def generate_face_edges(self):
        for i in range(0, len(self.data)):
            item = self.data[i]
            item["face_edges"] = derive_face_edges_from_faces(item["faces"])

        desired_order = ["vertices", "faces", "face_edges", "text"]
        self.data = [{key: d[key] for key in desired_order} for d in self.data]
        print(f"[MeshDataset] Generated face_edges for {len(self.data)} entrys")

    def generate_codes(self, autoencoder: MeshAutoencoder):
        for i in range(0, len(self.data)):
            item = self.data[i]

            codes = autoencoder.tokenize(
                vertices=item["vertices"],
                faces=item["faces"],
                face_edges=item["face_edges"],
            )
            item["codes"] = codes

        print(f"[MeshDataset] Generated codes for {len(self.data)} entrys")

    def embed_texts(self, transformer: MeshTransformer):
        unique_texts = set(item["text"] for item in self.data)

        text_embeddings = transformer.embed_texts(list(unique_texts))
        print(f"[MeshDataset] Generated {len(text_embeddings)} text_embeddings")
        text_embedding_dict = dict(zip(unique_texts, text_embeddings))

        for item in self.data:
            text_value = item["text"]
            item["text_embeds"] = text_embedding_dict.get(text_value, None)
            del item["text"]


# tensor helper functions

def derive_face_edges_from_faces(
    faces: TensorType['b', 'nf', 3, int],
    pad_id = -1,
    neighbor_if_share_one_vertex = False,
    include_self = True
) -> TensorType['b', 'e', 2, int]:

    is_one_face, device = faces.ndim == 2, faces.device

    if is_one_face:
        faces = rearrange(faces, 'nf c -> 1 nf c')

    max_num_faces = faces.shape[1]
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_faces, device = device),
        torch.arange(max_num_faces, device = device),
    indexing = 'ij'), dim = -1)

    face_masks = reduce(faces != pad_id, 'b nf c -> b nf', 'all')
    face_edges_masks = rearrange(face_masks, 'b i -> b i 1') & rearrange(face_masks, 'b j -> b 1 j')

    face_edges = []

    for face, face_edge_mask in zip(faces, face_edges_masks):

        shared_vertices = rearrange(face, 'i c -> i 1 c 1') == rearrange(face, 'j c -> 1 j 1 c')
        num_shared_vertices = shared_vertices.any(dim = -1).sum(dim = -1)

        is_neighbor_face = (num_shared_vertices >= face_edges_vertices_threshold) & face_edge_mask

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3

        face_edge = all_edges[is_neighbor_face]
        face_edges.append(face_edge)

    face_edges = pad_sequence(face_edges, padding_value = pad_id, batch_first = True)

    if is_one_face:
        face_edges = rearrange(face_edges, '1 e ij -> e ij')

    return face_edges

# custom collater

def first(it):
    return it[0]

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output
