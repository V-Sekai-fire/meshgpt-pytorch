from gltf_dataset.gltf_dataset import MeshDataset
import wandb, torch
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshAutoencoderTrainer,
)
from meshgpt_pytorch.data import derive_face_edges_from_faces

from einops import rearrange, reduce

from meshgpt_pytorch.meshgpt_pytorch import undiscretize

wandb.init(
    project="meshgpt-pytorch"
)

dataset = MeshDataset('gltf_dataset/blockmesh_test/blockmesh')

checkpoint_path = 'checkpoints/mesh-autoencoder.ckpt.1.pt'

autoencoder = MeshAutoencoder.init_and_load_from(checkpoint_path)

autoencoder.eval()

sample_data = dataset.__getitem__(0)
vertices = sample_data[0].unsqueeze(0) 
faces = sample_data[1].unsqueeze(0) 

with torch.no_grad():
    pad_value = -1
    face_edges = derive_face_edges_from_faces(faces, pad_value)

    num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device
    face_mask = reduce(faces != pad_value, 'b nf c -> b nf', 'all')
    face_edges_mask = reduce(face_edges != pad_value, 'b e ij -> b e', 'all')

    encoded = autoencoder.encode(
        vertices = vertices,
        faces = faces,
        face_edges = face_edges,
        face_edges_mask = face_edges_mask,
        face_mask = face_mask,
    )
    rvq_sample_codebook_temp = 1
    quantized, codes, commit_loss = autoencoder.quantize(
        face_embed = encoded,
        faces = faces,
        face_mask = face_mask,
        rvq_sample_codebook_temp = rvq_sample_codebook_temp
    )

    decode = autoencoder.decode(
        quantized,
        face_mask = face_mask
    )

    pred_face_coords = autoencoder.to_coor_logits(decode)

    pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')

    continuous_coors = undiscretize(
                pred_face_coords,
                num_discrete = 128,
                continuous_range = (-1., 1.)
            )
    # Todo