import torch, wandb, os
import numpy as np

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshAutoencoderTrainer,
)

from dataset.dataset import MeshDataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_directory = "dataset/unit_test"

dataset = MeshDataset(dataset_directory)

run = wandb.init(
    project="meshgpt-pytorch",
    
    config={
        "learning_rate": 1e-2,
        "architecture": "MeshGPT",
        "dataset": dataset_directory,
        "num_train_steps": 800,
        "num_transformer_train_steps": 100,
        "warmup_steps": 1,
        "batch_size": 4,
        "grad_accum_every": 1,
        "checkpoint_every": 20,
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

if True:
    load_from_checkpoint = True
    checkpoint_path = 'checkpoints/mesh-autoencoder.ckpt.20.pt'
    autoencoder = None
    if load_from_checkpoint and os.path.isfile(checkpoint_path):
        autoencoder = MeshAutoencoder(
            dim = run.config.autoencoder["dim"],
            encoder_depth = run.config.autoencoder["encoder_depth"],
            decoder_depth = run.config.autoencoder["decoder_depth"],
            num_discrete_coors = run.config.autoencoder["num_discrete_coors"]
        ).to(device)
        autoencoder.init_and_load_from(checkpoint_path)
        print(f"Loaded checkpoint '{checkpoint_path}'")
    else:
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

    from meshgpt_pytorch import MeshTransformer, MeshTransformerTrainer

    transformer = MeshTransformer(
        autoencoder,
        dim=512,
        max_seq_len=768,
    ).to(device)

    transformer_trainer = MeshTransformerTrainer(
        transformer,
        dataset=dataset,
        batch_size=wandb.config.batch_size,
        grad_accum_every=wandb.config.grad_accum_every,
        num_train_steps=wandb.config.num_transformer_train_steps,
        checkpoint_every=wandb.config.checkpoint_every,
        warmup_steps=wandb.config.warmup_steps,
        learning_rate=wandb.config.learning_rate,
        use_wandb_tracking=True,
    )

    transformer_trainer()

    continuous_coors  = transformer.generate()

    # Move the tensor to CPU before converting to a list
    continuous_coors_list = continuous_coors.cpu().tolist()

    import json

    with open('continuous_coors.json', 'w') as f:
        json.dump(continuous_coors.tolist(), f)

else:
    import json

    with open('continuous_coors.json', 'r') as f:
        continuous_coors_list = json.load(f)

flat_list = [item for sublist in continuous_coors_list for item in sublist]

vertices = [vertex for sublist in flat_list for vertex in sublist]
# print("Vertices:", vertices)

faces = [[i, i+1, i+2] for i in range(0, len(vertices), 3)]

# Assuming dataset is an instance of a class that has a method convert_to_glb
dataset.convert_to_glb((vertices, faces), "output.glb")