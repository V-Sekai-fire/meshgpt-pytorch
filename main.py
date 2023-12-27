import argparse

from run import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeshGPT PyTorch Training Script")
    parser.add_argument(
        "--dataset_directory",
        default="dataset/unit_test",
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
        default=600,
        help="Number of training steps for the autoencoder. Default is 1200.",
    )
    parser.add_argument(
        "--transformer_train",
        type=int,
        default=600,
        help="Number of training steps for the transformer. Default is 600.",
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
        "--checkpoint_every",
        type=int,
        default=20,
        help="Save a checkpoint every N steps. Default is 1.",
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
        "--texts",
        type=str,
        help="Comma-separated list of texts to generate meshes for.",
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="If set, continue training from the last checkpoint.",
    )
    args = parser.parse_args()

    if args.test_mode:
        args.autoencoder_train = 1
        args.transformer_train = 1

    main(args)
