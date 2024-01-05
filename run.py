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
from meshgpt_pytorch.data import MeshDataset
from mesh_utils import (
    generate_mesh_data,
    process_mesh_data,
    load_and_process_files,
    train_autoencoder,
    train_transformer,
)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="MeshGPT PyTorch Training Script")
    parser.add_argument(
        "--dataset_directory",
        default="dataset/blockmesh_test",
        help="Path to the directory containing the dataset. Default is dataset/blockmesh_test.",
    )
    parser.add_argument(
        "--data_augment",
        type=int,
        default=100,
        help="Number of data augmentations to apply. Default is 100.",
    )
    parser.add_argument(
        "--autoencoder_learning_rate",
        type=float,
        default=0.4,
        help="Learning rate for the autoencoder. Default is 0.4.",
    )
    parser.add_argument(
        "--transformer_learning_rate",
        type=float,
        default=0.2,
        help="Learning rate for the transformer. Default is 0.2.",
    )
    parser.add_argument(
        "--autoencoder_train",
        type=int,
        default=6000,
        help="Number of training steps for the autoencoder. Default is 6000.",
    )
    parser.add_argument(
        "--transformer_train",
        type=int,
        default=6000,
        help="Number of training steps for the transformer. Default is 6000.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training. Default is 1.",
    )
    parser.add_argument(
        "--grad_accum_every",
        type=int,
        default=2,
        help="Gradient accumulation steps. Default is 2.",
    )
    parser.add_argument(
        "--num_discrete_coors",
        type=int,
        default=1024,
        help="Number of discrete coordinates. Default is 1024.",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="If set, only inference will be performed.",
    )
    parser.add_argument(
        "--autoencoder_path", help="Path to the pre-trained autoencoder model."
    )
    parser.add_argument(
        "--transformer_path", help="Path to the pre-trained transformer model."
    )
    parser.add_argument(
        "--num_quantizers",
        type=int,
        default=2,
        help="Number of quantizers for the autoencoder. Default is 2.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="If set, the script will run in test mode with reduced training steps and a fixed dataset directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="object",
        help="Text to generate meshes for.",
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="If set, continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--load_dataset",
        action="store_true",
        help="If set, load the dataset from 'mesh_dataset.npz' instead of generating it.",
    )
    parser.add_argument(
        "--max_faces_allowed",
        type=int,
        default=2345,
        help="Maximum number of faces for the transformer. Default is 100.",
    )
    args = parser.parse_args()

    if args.test_mode:
        args.autoencoder_train = 1
        args.transformer_train = 1

    dataset_directory = args.dataset_directory
    data_augment = args.data_augment
    autoencoder = None

    run = wandb.init(
        project="meshgpt-pytorch",
        config={
            "max_faces_allowed": args.max_faces_allowed,
            "transformer_path": args.transformer_path,
            "autoencoder_path": args.autoencoder_path,
            "dim": 512,
            "inference_only": args.inference_only,
            "autoencoder_learning_rate": args.autoencoder_learning_rate,
            "transformer_learning_rate": args.transformer_learning_rate,
            "architecture": "MeshGPT",
            "dataset_directory": dataset_directory,
            "autoencoder_train": args.autoencoder_train,
            "transformer_train": args.transformer_train,
            "batch_size": args.batch_size,
            "grad_accum_every": args.grad_accum_every,
            "device": str(device),
            "num_quantizers": args.num_quantizers,
            "autoencoder": {
                "num_discrete_coors": args.num_discrete_coors,
            },
        },
    )

    folder_path = args.dataset_directory
    supported_formats = (".glb", ".gltf")
    files = sorted(
        [
            file
            for file in os.listdir(folder_path)
            if os.path.splitext(file)[1] in supported_formats
        ]
    )

    idx_to_file_idx = load_and_process_files(
        folder_path, supported_formats, args.max_faces_allowed
    )

    dataset = (
        MeshDataset.load("mesh_dataset.npz")
        if args.load_dataset
        else MeshDataset(
            [
                generate_mesh_data(
                    idx, idx_to_file_idx, files, folder_path, args.max_faces_allowed
                )
                for idx in range(len(idx_to_file_idx))
            ]
        )
    )
    if not args.load_dataset:
        dataset.save("mesh_dataset.npz")

    seq_len = args.max_faces_allowed * 3 * run.config.num_quantizers

    if not args.inference_only:
        autoencoder = MeshAutoencoder(
            num_quantizers=run.config.num_quantizers,
            num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
        ).to(device)
        if args.autoencoder_path:
            autoencoder.init_and_load(run.config.autoencoder_path, strict=False)
        if args.continue_train or not args.autoencoder_path:
            train_autoencoder(run, dataset, autoencoder)

        transformer = None
        if args.transformer_path:
            print(f"Sequence length: {seq_len}")
            transformer = MeshTransformer(
                autoencoder,
                dim=run.config.dim,
                max_seq_len=seq_len,
                condition_on_text=True,
            ).to(device)
            transformer.load(run.config.transformer_path)
        elif not args.transformer_path:
            transformer = train_transformer(autoencoder, run, dataset, device, seq_len)
    else:
        if args.autoencoder_path and args.transformer_path:
            autoencoder = MeshAutoencoder(
                num_quantizers=run.config.num_quantizers,
                num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
            ).to(device)
            autoencoder.init_and_load(run.config.autoencoder_path)
            transformer = MeshTransformer(
                autoencoder,
                dim=run.config.dim,
                max_seq_len=seq_len,
                condition_on_text=True,
            ).to(device)
            transformer.load(run.config.transformer_path)
        else:
            print(
                "Both autoencoder and transformer paths must be provided for inference."
            )
    process_mesh_data(run, device, transformer, args.text)
