import torch, wandb, os, random
import numpy as np

from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshAutoencoderTrainer,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

from dataset.dataset import MeshDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_directory = "dataset/unit_test"

data_augment = 4

dataset = MeshDataset(dataset_directory, data_augment)

run = wandb.init(
    project="meshgpt-pytorch",
    config={
        "seed": 42,
        "get_max_face_count": dataset.get_max_face_count(),
        "autoencoder_learning_rate": 0.2,
        "transformer_learning_rate": 0.1,
        "architecture": "MeshGPT",
        "dataset": dataset_directory,
        "data_augment": data_augment,
        "autoencoder_train": 300,
        "transformer_train": 375,
        "batch_size": 1,
        "grad_accum_every": 1,
        "checkpoint_every": 40,
        "device": str(device),
        "autoencoder": {
            "dim": 512,
            "encoder_depth": 6,
            "decoder_depth": 6,
            "num_discrete_coors": 256,
        },
        "dataset_size": dataset.__len__(),
    },
)

set_seed(wandb.config.seed)

seq_len = dataset.get_max_face_count() * 3
seq_len = ((seq_len + 2) // 3) * 3

print(f"Sequence length: {seq_len}")

load_from_checkpoint = True
autoencoder = None

autoencoder = MeshAutoencoder(
    num_quantizers=1,
    dim=run.config.autoencoder["dim"],
    encoder_depth=run.config.autoencoder["encoder_depth"],
    decoder_depth=run.config.autoencoder["decoder_depth"],
    num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
).to(device)  

trainer = MeshAutoencoderTrainer(
    autoencoder,
    num_train_steps=2000, # Ignored?
    dataset=dataset,
    batch_size=wandb.config.batch_size,
    grad_accum_every=wandb.config.grad_accum_every,
    checkpoint_every=wandb.config.checkpoint_every,
    warmup_steps=500, # Ignored?
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
    num_train_steps=2000, # Ignored?
    dataset=dataset,
    batch_size=wandb.config.batch_size,
    grad_accum_every=wandb.config.grad_accum_every,
    checkpoint_every=wandb.config.checkpoint_every,
    warmup_steps=500, # Ignored?
    learning_rate=wandb.config.transformer_learning_rate,
    use_wandb_tracking=True,
)

transformer_trainer.train(run.config.transformer_train)

codes, continuous_coors = transformer.generate(return_codes = True)

codes_list = codes.cpu().tolist()

import json

with open("output_codes.json", "w") as f:
    json.dump(codes_list, f)
    
continuous_coors_list = continuous_coors.cpu().tolist()

with open("continuous_coors.json", "w") as f:
    json.dump(continuous_coors.tolist(), f)

flat_list = [item for sublist in continuous_coors_list for item in sublist]

vertices = [vertex for sublist in flat_list for vertex in sublist]

faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]

dataset.convert_to_glb((vertices, faces), "output.glb")
dataset.convert_to_obj((vertices, faces), "output.obj")

def encode_to_pua(codes):
    flat_codes = [item for sublist in codes for subsublist in sublist for item in subsublist]
    return ''.join(chr(code + 0xE000) for code in flat_codes)

encoded_codes = encode_to_pua(codes_list)

with open('output.obj', 'r') as file:
    obj_contents = file.read()

new_data = [[
    {"role": "system", "content": "This assistant can understand 3D models using the meshgpt-pytorch Unicode plane 15 codebook for 16384 triangles and the .ply 3d format."},
    {"role": "user", "content": f"{encoded_codes}"},
    {"role": "assistant", "content": f"{obj_contents}"}
]]
data = []
try:
    with open('chatml.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
except FileNotFoundError:
    print("The file 'chatml.jsonl' does not exist.")

data = new_data + data

with open('chatml.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')