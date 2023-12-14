import torch, wandb, os
import numpy as np

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshAutoencoderTrainer,
)

from dataset.dataset import MeshDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_directory = "dataset/blockmesh_test/blockmesh"

dataset = MeshDataset(dataset_directory)

seq_len = 32112

print(f"Sequence length: {seq_len}")

run = wandb.init(
    project="meshgpt-pytorch",
    config={
        "autoencoder_learning_rate": 0.001,
        "transformer_learning_rate": 0.001,
        "architecture": "MeshGPT",
        "dataset": dataset_directory,
        "num_train_steps": 3000,
        "num_transformer_train_steps": 1000,
        "warmup_steps": 1000,
        "batch_size": 1,
        "grad_accum_every": 1,
        "checkpoint_every": 40,
        "device": str(device),
        "autoencoder_train": 10,
        "transformer_train": 20,
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
checkpoint_path = "checkpoints/mesh-autoencoder.ckpt.27.pt"
autoencoder = None
num_train_steps = None
if load_from_checkpoint and os.path.isfile(checkpoint_path):
    autoencoder = MeshAutoencoder(
        dim=run.config.autoencoder["dim"],
        encoder_depth=run.config.autoencoder["encoder_depth"],
        decoder_depth=run.config.autoencoder["decoder_depth"],
        num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
    ).to(device)
    autoencoder.init_and_load_from(checkpoint_path)
    print(f"Loaded checkpoint '{checkpoint_path}'")
    num_train_steps = wandb.config.num_train_steps
else:
    autoencoder = MeshAutoencoder(
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

# Move the tensor to CPU before converting to a list
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
