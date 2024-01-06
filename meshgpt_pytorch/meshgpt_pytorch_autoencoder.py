from pathlib import Path
from functools import partial
from math import ceil, pi

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from torchtyping import TensorType

from pytorch_custom_utils import save_load

from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Dict, Any

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers import Decoder
from x_transformers.attend import Attend
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

from local_attention import LocalMHA

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.version import __version__

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from torch_geometric.nn.conv import SAGEConv

from gateloop_transformer import SimpleGateLoopLayer

from tqdm import tqdm

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(l):
    return len(l) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

# continuous embed

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )

# additional encoder features
# 1. angle (3), 2. area (1), 3. normals (3)

def derive_angle(x, y, eps = 1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()

@torch.no_grad()
def get_derived_face_features(
    face_coords: TensorType['b', 'nf', 3, 3, float]  # 3 vertices with 3 coordinates
):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2)

    angles  = derive_angle(face_coords, shifted_face_coords)

    edge1, edge2, _ = (face_coords - shifted_face_coords).unbind(dim = 2)

    normals = l2norm(torch.cross(edge1, edge2, dim = -1))
    area = normals.norm(dim = -1, keepdim = True) * 0.5

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )   

# tensor helper functions

@beartype
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@beartype
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@beartype
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, channels, _, device = *t.shape, t.device

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = torch.float, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)
    return F.conv1d(t, kernel, padding = half_width, groups = channels)

@beartype
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)

# resnet block

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        reduction_factor = 4,
        min_dim = 16
    ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min = 1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        groups = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        groups = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, groups = groups, dropout = dropout)
        self.block2 = Block(dim_out, dim_out, groups = groups, dropout = dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        res = self.residual_conv(x)
        h = self.block1(x, mask = mask)
        h = self.block2(h, mask = mask)
        h = self.excite(h, mask = mask)
        return h + res

# gateloop layers

class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    ):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim)
            self.gateloops.append(gateloop)

    def forward(
        self,
        x,
        cache = None
    ):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        return x, new_caches

# main classes

@save_load(version = __version__)
class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_area = 128,
        dim_area_embed = 16,
        num_discrete_normals = 128,
        dim_normal_embed = 64,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        encoder_dims_through_depth: Tuple[int, ...] = (
            64, 128, 256, 256, 576
        ),
        init_decoder_conv_kernel = 7,
        decoder_dims_through_depth: Tuple[int, ...] = (
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256, 256, 256,
            384, 384, 384
        ),
        dim_codebook = 192,
        num_quantizers = 2,           # or 'D' in the paper
        codebook_size = 16384,        # they use 16k, shared codebook between layers
        use_residual_lfq = True,      # whether to use the latest lookup-free quantization
        rq_kwargs: dict = dict(
            quantize_dropout = True,
            quantize_dropout_cutoff_index = 1,
            quantize_dropout_multiple_of = 1,
        ),
        rvq_kwargs: dict = dict(
            kmeans_init = True,
            threshold_ema_dead_code = 2,
        ),
        rlfq_kwargs: dict = dict(),
        rvq_stochastic_sample_codes = True,
        sageconv_kwargs: dict = dict(
            normalize = True,
            project = True
        ),
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions
        local_attn_encoder_depth = 0,
        local_attn_decoder_depth = 0,
        local_attn_kwargs: dict = dict(
            dim_head = 32,
            heads = 8
        ),
        local_attn_window_size = 64,
        pad_id = -1,
        flash_attn = True,
        sageconv_dropout = 0.,
        attn_dropout = 0.,
        ff_dropout = 0.,
        resnet_dropout = 0
    ):
        super().__init__()

        # main face coordinate embedding

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # attention related

        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        # initial dimension

        init_dim = dim_coor_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed

        # project into model dimension

        self.project_in = nn.Linear(init_dim, dim_codebook)

        # initial sage conv

        sageconv_kwargs = {**sageconv_kwargs, 'sageconv_dropout' : sageconv_dropout}

        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook, init_encoder_dim, **sageconv_kwargs)

        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        )

        self.encoders = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )

            self.encoders.append(sage_conv)
            curr_dim = dim_layer

        self.encoder_local_attn_blocks = ModuleList([])

        for _ in range(local_attn_encoder_depth):
            self.encoder_local_attn_blocks.append(nn.ModuleList([
                LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = ff_dropout))
            ]))

        # residual quantization

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * 3)

        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                commitment_loss_weight = 1.,
                **rlfq_kwargs,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                shared_codebook = True,
                commitment_weight = 1.,
                stochastic_sample_codes = rvq_stochastic_sample_codes,
                **rvq_kwargs,
                **rq_kwargs
            )

        self.pad_id = pad_id # for variable lengthed faces, padding quantized ids will be set to this value

        # decoder

        decoder_input_dim = dim_codebook * 3

        self.decoder_local_attn_blocks = ModuleList([])

        for _ in range(local_attn_decoder_depth):
            self.decoder_local_attn_blocks.append(nn.ModuleList([
                LocalMHA(dim = decoder_input_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(decoder_input_dim), FeedForward(decoder_input_dim, glu = True, dropout = ff_dropout))
            ]))

        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * 3, init_decoder_dim, kernel_size = init_decoder_conv_kernel, padding = init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.decoders = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout = resnet_dropout)

            self.decoders.append(resnet_block)
            curr_dim = dim_layer

        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * 9),
            Rearrange('... (v c) -> ... v c', v = 9)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 3, int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask:        TensorType['b', 'nf', bool],
        face_edges_mask:  TensorType['b', 'e', bool],
        return_face_coordinates = False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        c - coordinates (3)
        d - embed dim
        """

        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        _, num_faces, _ = faces.shape

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)

        # continuous face coords

        face_coords = vertices.gather(-2, faces_vertices)

        # compute derived features and embed

        derived_features = get_derived_face_features(face_coords)

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        # discretize vertices for face coordinate embedding

        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 coordinates per face

        face_coor_embed = self.coor_embed(discrete_face_coords)
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        # combine all features and project into model dimension

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        face_embed = self.project_in(face_embed)

        # handle variable lengths by using masked_select and masked_scatter

        # first handle edges
        # needs to be offset by number of faces for each batch

        face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
        face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')

        face_edges = face_edges + face_index_offsets
        face_edges = face_edges[face_edges_mask]
        face_edges = rearrange(face_edges, 'be ij -> ij be')

        # next prepare the face_mask for using masked_select and masked_scatter

        orig_face_embed_shape = face_embed.shape[:2]

        face_embed = face_embed[face_mask]

        # initial sage conv followed by activation and norm

        face_embed = self.init_sage_conv(face_embed, face_edges)

        face_embed = self.init_encoder_act_and_norm(face_embed)

        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges)

        shape = (*orig_face_embed_shape, face_embed.shape[-1])

        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)

        for attn, ff in self.encoder_local_attn_blocks:
            face_embed = attn(face_embed, mask = face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @beartype
    def quantize(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, num_faces, device = *faces.shape[:2], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nv d) -> b nf nv d', nv = 3)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nv -> b (nf nv) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict()

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # residual VQ

        quantized, codes, commit_loss = self.quantizer(averaged_vertices, mask = mask, **quantize_kwargs)

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = quantized.gather(-2, faces_with_dim)
        face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, faces_with_quantized_dim)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nv) 1', nv = 3)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized

        for attn, ff in self.decoder_local_attn_blocks:
            x = attn(x, mask = face_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n')

        x = x.masked_fill(~conv_face_mask, 0.)
        x = self.init_decoder_conv(x)

        for resnet_block in self.decoders:
            x = resnet_block(x, mask = conv_face_mask)

        return rearrange(x, 'b d n -> b n d')

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: Optional[TensorType['b', 'n', bool]] = None,
        return_discrete_codes = False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nv q) -> b nf', 'all', nv = 3, q = self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        decoded = self.decode(
            quantized,
            face_mask = face_mask
        )

        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = 3)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        # mask out with nan

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, face_mask

        return continuous_coors, pred_face_coords, face_mask

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges = None, **kwargs):
        assert 'return_codes' not in kwargs

        inputs = [vertices, faces, face_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'faces', 'face_edges'], inputs))

        self.eval()

        codes = self.forward(
            **input_kwargs,
            return_codes = True,
            **kwargs
        )

        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
        rvq_sample_codebook_temp = 1.
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )

        quantized, codes, commit_loss = self.quantize(
            face_embed = encoded,
            faces = faces,
            face_mask = face_mask,
            rvq_sample_codebook_temp = rvq_sample_codebook_temp
        )

        if return_codes:
            assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'

            codes = codes.masked_fill(~repeat(face_mask, 'b nf -> b (nf 3) 1'), self.pad_id)
            return codes

        decode = self.decode(
            quantized,
            face_mask = face_mask
        )

        pred_face_coords = self.to_coor_logits(decode)

        # compute reconstructed faces if needed

        if return_recon_faces or only_return_recon_faces:

            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))

        if only_return_recon_faces:
            return recon_faces

        # prepare for recon loss

        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma)

            # cross entropy with localized smoothing

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = 9)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss

        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # calculate loss breakdown if needed

        loss_breakdown = (recon_loss, commit_loss)

        # some return logic

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown
