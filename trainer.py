import torch, wandb

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshAutoencoderTrainer,
)

from gltf_dataset.gltf_dataset import MeshDataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_directory = "gltf_dataset/blockmesh_test/blockmesh"

dataset = MeshDataset(dataset_directory)

run = wandb.init(
    project="meshgpt-pytorch",
    
    config={
        "learning_rate": 0.0001,
        "architecture": "MeshGPT",
        "dataset": dataset_directory,
        "num_train_steps": 2000,
        "warmup_steps": 1000,
        "batch_size": 1,
        "grad_accum_every": 1,
        "checkpoint_every": 10,
        "device": str(device),
        "autoencoder": {
            "dim": 512,
            "encoder_depth": 6,
            "decoder_depth": 6,
            "num_discrete_coors": 128,
        },
        "dataset_size": dataset.__len__(),
    }
)

autoencoder = MeshAutoencoder(
    dim = run.config.autoencoder["dim"],
    encoder_depth = run.config.autoencoder["encoder_depth"],
    decoder_depth = run.config.autoencoder["decoder_depth"],
    num_discrete_coors = run.config.autoencoder["num_discrete_coors"]
).to(device)

trainer = MeshAutoencoderTrainer(
    autoencoder,
    dataset = dataset,
    batch_size = wandb.config.batch_size,
    grad_accum_every = wandb.config.grad_accum_every,
    num_train_steps = wandb.config.num_train_steps,
    checkpoint_every = wandb.config.checkpoint_every,
    warmup_steps = wandb.config.warmup_steps,
    learning_rate = wandb.config.learning_rate,
    use_wandb_tracking = True,
)

trainer()
