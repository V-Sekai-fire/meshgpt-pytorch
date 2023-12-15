import torch, wandb, os
import numpy as np

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshAutoencoderTrainer,
)

from dataset.dataset import MeshDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_directory = "dataset/unit_test"

dataset = MeshDataset(dataset_directory, 4)

seq_len = len(dataset.__getitem__(0)[0]) * 12

print(f"Sequence length: {seq_len}")

run = wandb.init(
    project="meshgpt-pytorch",
    config={
        "autoencoder_learning_rate": 0.1,
        "transformer_learning_rate": 0.1,
        "architecture": "MeshGPT",
        "dataset": dataset_directory,
        "num_train_steps": 3000,
        "num_transformer_train_steps": 1000,
        "warmup_steps": 500,
        "batch_size": 4,
        "grad_accum_every": 1,
        "checkpoint_every": 40,
        "device": str(device),
        "autoencoder_train": 2000,
        "transformer_train": 500,
        "autoencoder": {
            "dim": 512,
            "encoder_depth": 6,
            "decoder_depth": 6,
            "num_discrete_coors": 128,
        },
        "dataset_size": dataset.__len__(),
    },
)

load_from_checkpoint = True
autoencoder = None
num_train_steps = None

autoencoder = MeshAutoencoder(
    num_quantizers=1,
    dim=run.config.autoencoder["dim"],
    encoder_depth=run.config.autoencoder["encoder_depth"],
    decoder_depth=run.config.autoencoder["decoder_depth"],
    num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
).to(device)  
num_train_steps = wandb.config.num_train_steps

trainer = MeshAutoencoderTrainer(
    autoencoder,
    dataset=dataset,
    batch_size=wandb.config.batch_size,
    grad_accum_every=wandb.config.grad_accum_every,
    num_train_steps=num_train_steps,
    checkpoint_every=wandb.config.checkpoint_every,
    warmup_steps=wandb.config.warmup_steps,
    learning_rate=wandb.config.autoencoder_learning_rate,
    use_wandb_tracking=True,
)
trainer.train(run.config.autoencoder_train)
autoencoder.save(f"checkpoints/autoencoder.pt", overwrite = True)

from meshgpt_pytorch import MeshTransformer, MeshTransformerTrainer

transformer = MeshTransformer(
    autoencoder,
    dim=512,
    max_seq_len=seq_len,
).to(device)

transformer_trainer = MeshTransformerTrainer(
    transformer,
    dataset=dataset,
    batch_size=wandb.config.batch_size,
    grad_accum_every=wandb.config.grad_accum_every,
    num_train_steps=wandb.config.num_transformer_train_steps,
    checkpoint_every=wandb.config.checkpoint_every,
    warmup_steps=wandb.config.warmup_steps,
    learning_rate=wandb.config.transformer_learning_rate,
    use_wandb_tracking=True,
)

transformer_trainer.train(run.config.transformer_train)

continuous_coors = transformer.generate()

continuous_coors_list = continuous_coors.cpu().tolist()

import json

with open("continuous_coors.json", "w") as f:
    json.dump(continuous_coors.tolist(), f)

if False:
    import json

    with open("continuous_coors.json", "r") as f:
        continuous_coors_list = json.load(f)

flat_list = [item for sublist in continuous_coors_list for item in sublist]

vertices = [vertex for sublist in flat_list for vertex in sublist]
# print("Vertices:", vertices)

faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]

# Assuming dataset is an instance of a class that has a method convert_to_glb
dataset.convert_to_glb((vertices, faces), "output.glb")
