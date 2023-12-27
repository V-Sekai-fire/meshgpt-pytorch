import argparse
import torch
import wandb
import os
from dataset.dataset import MeshDataset
from meshgpt_pytorch import MeshAutoencoder, MeshAutoencoderTrainer
from datetime import datetime
import re
import json
import wandb
from meshgpt_pytorch import MeshTransformer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_directory = args.dataset_directory
    data_augment = args.data_augment
    autoencoder = None

    run = wandb.init(
        project="fire-meshgpt-pytorch",
        config={
            "transformer_path": args.transformer_path,
            "autoencoder_path": args.autoencoder_path,
            "dim": 512,
            "inference_only": args.inference_only,
            "autoencoder_learning_rate": args.autoencoder_learning_rate,
            "transformer_learning_rate": args.transformer_learning_rate,
            "architecture": "MeshGPT",
            "dataset_directory": dataset_directory,
            "data_augment": data_augment,
            "autoencoder_train": args.autoencoder_train,
            "transformer_train": args.transformer_train,
            "batch_size": args.batch_size,
            "grad_accum_every": args.grad_accum_every,
            "checkpoint_every": args.checkpoint_every,
            "device": str(device),
            "num_quantizers": args.num_quantizers,
            "autoencoder": {
                "num_discrete_coors": args.num_discrete_coors,
            },
        },
    )
    dataset = MeshDataset(dataset_directory, data_augment)
    seq_len = dataset.get_max_face_count() * 3 * run.config.num_quantizers
    if seq_len < 8196:
        seq_len = 8196
    if not args.inference_only:

        if args.autoencoder_path:
            autoencoder = MeshAutoencoder(
                num_quantizers=run.config.num_quantizers,
                num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
            ).to(device)
            autoencoder.init_and_load(run.config.autoencoder_path, strict = False)
            if args.continue_train:
                train_autoencoder(run, dataset, autoencoder)
        else:
            autoencoder = MeshAutoencoder(
                num_quantizers=run.config.num_quantizers,
                num_discrete_coors=run.config.autoencoder["num_discrete_coors"],
            ).to(device)
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
        else:
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
            print("Both autoencoder and transformer paths must be provided for inference.")
            return

    texts = args.texts.split(',')
    process_mesh_data(run, device, transformer, texts)

def train_autoencoder(run, dataset, autoencoder):

    trainer = MeshAutoencoderTrainer(
        autoencoder,
        num_train_steps=2000,
        dataset=dataset,
        batch_size=run.config.batch_size,
        grad_accum_every=run.config.grad_accum_every,
        checkpoint_every_epoch=run.config.checkpoint_every,
        learning_rate=run.config.autoencoder_learning_rate,
        use_wandb_tracking=True,
    )
    trainer.train(run.config.autoencoder_train)

    current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    trainer.save(f"checkpoints/mesh_autoencoder_{os.path.basename(run.config.dataset_directory)}_{current_time}.pt")

    return autoencoder

from datetime import datetime
from meshgpt_pytorch import MeshTransformer, MeshTransformerTrainer

def train_transformer(autoencoder, run, dataset, device, seq_len):
    transformer = MeshTransformer(
        autoencoder,
        max_seq_len=seq_len,
        dim=run.config.dim,
        condition_on_text=True,
    ).to(device)

    transformer_trainer = MeshTransformerTrainer(
        transformer,
        num_train_steps=2000,
        dataset=dataset,
        batch_size=wandb.config.batch_size,
        grad_accum_every=wandb.config.grad_accum_every,
        checkpoint_every_epoch=wandb.config.checkpoint_every,
        warmup_steps=500,
        learning_rate=wandb.config.transformer_learning_rate,
        use_wandb_tracking=True,
    )
    transformer_trainer.train(run.config.transformer_train)

    current_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    transformer_trainer.save(f"checkpoints/mesh_transformer_{os.path.basename(run.config.dataset_directory)}_{current_time}.pt")
    return transformer

def process_mesh_data(run, device, transformer, texts):
    codes = transformer.generate(return_codes=True, texts=texts)

    transformer.autoencoder.eval()

    continuous_coors = transformer.autoencoder.decode_from_codes_to_faces(codes)[0]

    continuous_coors_list = [np_array.tolist() for np_array in continuous_coors]

    flat_list = [item for sublist in continuous_coors_list for item in sublist]

    vertices = [vertex for sublist in flat_list for vertex in sublist]

    faces = [[i, i + 1, i + 2] for i in range(0, len(vertices), 3)]

    MeshDataset.convert_to_glb((vertices, faces), "output.glb")
    MeshDataset.convert_to_obj((vertices, faces), "output.obj")

    def encode_to_pua(codes):
        flat_codes = [
            item for sublist in codes for subsublist in sublist for item in subsublist
        ]
        return "".join(chr(code + 0xF0000) for code in flat_codes)

    encoded_codes = encode_to_pua(codes.cpu().tolist())

    with open("output.obj", "r") as file:
        obj_contents = file.read()

    new_data = [
        [
            {
                "role": "system",
                "content": "This assistant can understand 3D models using the meshgpt-pytorch Unicode plane 15 codebook for 16384 triangles and the .obj 3d format.",
            },
            {"role": "user", "content": f"{encoded_codes}"},
            {"role": "assistant", "content": f"{obj_contents}"},
        ]
    ]

    data = []
    try:
        with open("chatml.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("The file 'chatml.jsonl' does not exist.")

    data = new_data + data

    with open("chatml.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeshGPT PyTorch Training Script")
    parser.add_argument("--dataset_directory", default="dataset/blockmesh_test",
                        help="Path to the directory containing the dataset. Default is dataset/blockmesh_test.")   
    parser.add_argument("--data_augment", type=int, default=100, 
                        help="Number of data augmentations to apply. Default is 100.")
    parser.add_argument("--autoencoder_learning_rate", type=float, default=0.4, 
                        help="Learning rate for the autoencoder. Default is 0.4.")
    parser.add_argument("--transformer_learning_rate", type=float, default=0.2, 
                        help="Learning rate for the transformer. Default is 0.2.")
    parser.add_argument("--autoencoder_train", type=int, default=1200, 
                        help="Number of training steps for the autoencoder. Default is 1200.")
    parser.add_argument("--transformer_train", type=int, default=600, 
                        help="Number of training steps for the transformer. Default is 600.")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for training. Default is 1.")
    parser.add_argument("--grad_accum_every", type=int, default=2, 
                        help="Gradient accumulation steps. Default is 2.")
    parser.add_argument("--checkpoint_every", type=int, default=20, 
                        help="Save a checkpoint every N steps. Default is 1.")
    parser.add_argument("--num_discrete_coors", type=int, default=1024, 
                        help="Number of discrete coordinates. Default is 1024.")
    parser.add_argument("--inference_only", action='store_true', 
                        help="If set, only inference will be performed.")
    parser.add_argument("--autoencoder_path", 
                        help="Path to the pre-trained autoencoder model.")
    parser.add_argument("--transformer_path", 
                        help="Path to the pre-trained transformer model.")
    parser.add_argument("--num_quantizers", type=int, default=2, 
                        help="Number of quantizers for the autoencoder. Default is 2.")
    parser.add_argument("--test_mode", action='store_true', 
                        help="If set, the script will run in test mode with reduced training steps and a fixed dataset directory.")
    parser.add_argument("--texts", type=str, 
                        help="Comma-separated list of texts to generate meshes for.")
    parser.add_argument("--continue_train", action='store_true', 
                        help="If set, continue training from the last checkpoint.")
    args = parser.parse_args()

    if args.test_mode:
        args.autoencoder_train = 1
        args.transformer_train = 1

    main(args)