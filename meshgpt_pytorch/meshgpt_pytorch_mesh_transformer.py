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

from meshgpt_pytorch.meshgpt_pytorch_autoencoder import (
    MeshAutoencoder,
)

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
class MeshTransformer(Module):
    @beartype
    def __init__(
        self,
        autoencoder: Union[MeshAutoencoder],
        *,
        dim: Union[int, Tuple[int, int]] = 768,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            num_mem_kv = 4
        ),
        dropout = 0.,
        coarse_pre_gateloop_depth = 2,
        fine_pre_gateloop_depth = 2,
        gateloop_use_heinsen = True,
        fine_attn_depth = 2,
        fine_attn_dim_head = 32,
        fine_attn_heads = 8,
        pad_id = -1,
        condition_on_text = False,
        text_condition_model_types = ('t5',),
        text_condition_cond_drop_prob = 0.25
    ):
        super().__init__()

        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim_fine))
        self.eos_token_id = self.codebook_size

        # they use axial positional embeddings

        assert divisible_by(max_seq_len, 3 * self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by (3 x {self.num_quantizers}) = {3 * self.num_quantizers}' # 3 vertices per face, with D codes per vertex

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(3, dim))

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        # text condition

        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                cond_drop_prob = text_condition_cond_drop_prob
            )
            cross_attn_dim_context = self.conditioner.dim_latent

        # for summarizing the vertices of each face

        self.to_face_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * 3 * dim, dim),
            nn.LayerNorm(dim)
        )

        self.coarse_gateloop_block = GateLoopBlock(dim, depth = coarse_pre_gateloop_depth, use_heinsen = gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else nn.Identity()

        # main autoregressive attention network
        # attending to a face token

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            flash_attn = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = condition_on_text,
            cross_attn_dim_context = cross_attn_dim_context,
            **attn_kwargs
        )

        # projection from coarse to fine, if needed

        self.maybe_project_coarse_to_fine = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()

        # address a weakness in attention

        self.fine_gateloop_block = GateLoopBlock(dim, depth = fine_pre_gateloop_depth) if fine_pre_gateloop_depth > 0 else nn.Identity()

        # decoding the vertices, 2-stage hierarchy

        self.fine_decoder = Decoder(
            dim = dim_fine,
            depth = fine_attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            flash_attn = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **attn_kwargs
        )

        # to logits

        self.to_logits = nn.Linear(dim_fine, self.codebook_size + 1)

        # padding id
        # force the autoencoder to use the same pad_id given in transformer

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    @torch.no_grad()
    def embed_texts(self, texts: Union[str, List[str]]):
        single_text = not isinstance(texts, list)
        if single_text:
            texts = [texts]

        assert exists(self.conditioner)
        text_embeds = self.conditioner.embed_texts(texts).detach()

        if single_text:
            text_embeds = text_embeds[0]

        return text_embeds

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_scale = 1.,
        cache_kv = True,
        face_coords_to_file: Optional[Callable[[Tensor], Any]] = None
    ):
        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'
            if exists(texts):
                text_embeds = self.embed_texts(texts)

            batch_size = default(batch_size, text_embeds.shape[0])

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        cache = (None, None)

        for i in tqdm(range(curr_length, self.max_seq_len)):
            # v1([q1] [q2] [q1] [q2] [q1] [q2]) v2([eos| q1] [q2] [q1] [q2] [q1] [q2]) -> 0 1 2 3 4 5 6 7 8 9 10 11 12 -> v1(F F F F F F) v2(T F F F F F) v3(T F F F F F)

            can_eos = i != 0 and divisible_by(i, self.num_quantizers * 3)  # only allow for eos to be decoded at the end of each face, defined as 3 vertices with D residual VQ codes

            output = self.forward_on_codes(
                codes,
                text_embeds = text_embeds,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
                cond_scale = cond_scale,
                cfg_routed_kwargs = dict(
                    cache = cache
                )
            )

            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

            logits = logits[:, -1]

            if not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                sample = filtered_logits.argmax(dim = -1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate

            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim = -1).all():
                break

        # mask out to padding anything after the first eos

        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        # remove a potential extra token from eos, if breaked early

        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_quantizers * self.num_quantizers
        codes = codes[:, :round_down_code_len]

        # early return of raw residual quantizer codes

        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        self.autoencoder.eval()
        face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

        if not exists(face_coords_to_file):
            return face_coords, face_mask

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, int],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        codes:          Optional[Tensor] = None,
        cache:          Optional[LayerIntermediates] = None,
        **kwargs
    ):
        if not exists(codes):
            codes = self.autoencoder.tokenize(
                vertices = vertices,
                faces = faces,
                face_edges = face_edges
            )

        return self.forward_on_codes(codes, cache = cache, **kwargs)

    @classifier_free_guidance
    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = 0.
    ):
        # handle text conditions

        attn_context_kwargs = dict()

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'

            if exists(texts):
                text_embeds = self.conditioner.embed_texts(texts)

            if exists(codes):
                assert text_embeds.shape[0] == codes.shape[0], 'batch size of texts or text embeddings is not equal to the batch size of the mesh codes'

            _, maybe_dropped_text_embeds = self.conditioner(
                text_embeds = text_embeds,
                cond_drop_prob = cond_drop_prob
            )

            attn_context_kwargs = dict(
                context = maybe_dropped_text_embeds.embed,
                context_mask = maybe_dropped_text_embeds.mask
            )

        # take care of codes that may be flattened

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # get some variable

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # auto append eos token

        if append_eos:
            assert exists(codes)

            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)

            codes = F.pad(codes, (0, 1), value = 0)

            batch_arange = torch.arange(batch, device = device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')

            codes[batch_arange, code_lens] = self.eos_token_id

        # if returning loss, save the labels for cross entropy

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed (each residual VQ id)

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions

        seq_arange = torch.arange(codes.shape[-2], device = device)

        codes = codes + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        code_len = codes.shape[1]

        level_embed = repeat(self.quantize_level_embed, 'q d -> (r q) d', r = ceil(code_len / self.num_quantizers))
        codes = codes + level_embed[:code_len]

        # embedding for each vertex

        vertex_embed = repeat(self.vertex_embed, 'nv d -> (r nv q) d', r = ceil(code_len / (3 * self.num_quantizers)), q = self.num_quantizers)
        codes = codes + vertex_embed[:code_len]

        # create a token per face, by summarizing the 3 vertices
        # this is similar in design to the RQ transformer from Lee et al. https://arxiv.org/abs/2203.01941

        num_tokens_per_face = self.num_quantizers * 3

        curr_vertex_pos = code_len % num_tokens_per_face # the current intra-face vertex-code position id, needed for caching at the fine decoder stage

        code_len_is_multiple_of_face = divisible_by(code_len, num_tokens_per_face)

        next_multiple_code_len = ceil(code_len / num_tokens_per_face) * num_tokens_per_face

        codes = pad_to_length(codes, next_multiple_code_len, dim = -2)

        # grouped codes will be used for the second stage

        grouped_codes = rearrange(codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # create the coarse tokens for the first attention network

        face_codes = grouped_codes if code_len_is_multiple_of_face else grouped_codes[:, :-1]
        face_codes = rearrange(face_codes, 'b nf n d -> b nf (n d)')
        face_codes = self.to_face_tokens(face_codes)

        face_codes_len = face_codes.shape[-2]

        # cache logic

        (
            cached_attended_face_codes,
            coarse_cache,
            fine_cache,
            coarse_gateloop_cache,
            fine_gateloop_cache
        ) = cache if exists(cache) else ((None,) * 5)

        if exists(cache):
            cached_face_codes_len = cached_attended_face_codes.shape[-2]
            need_call_first_transformer = face_codes_len > cached_face_codes_len
        else:
            need_call_first_transformer = True

        should_cache_fine = not divisible_by(curr_vertex_pos + 1, num_tokens_per_face)

        # attention on face codes (coarse)

        if need_call_first_transformer:
            face_codes, coarse_gateloop_cache = self.coarse_gateloop_block(face_codes, cache = coarse_gateloop_cache)

            attended_face_codes, coarse_cache = self.decoder(
                face_codes,
                cache = coarse_cache,
                return_hiddens = True,
                **attn_context_kwargs
            )

            attended_face_codes = safe_cat((cached_attended_face_codes, attended_face_codes), dim = -2)
        else:
            attended_face_codes = cached_attended_face_codes

        # maybe project from coarse to fine dimension for hierarchical transformers

        attended_face_codes = self.maybe_project_coarse_to_fine(attended_face_codes)

        # auto prepend sos token

        sos = repeat(self.sos_token, 'd -> b d', b = batch)

        attended_face_codes_with_sos, _ = pack([sos, attended_face_codes], 'b * d')

        grouped_codes = pad_to_length(grouped_codes, attended_face_codes_with_sos.shape[-2], dim = 1)
        fine_vertex_codes, _ = pack([attended_face_codes_with_sos, grouped_codes], 'b n * d')

        fine_vertex_codes = fine_vertex_codes[..., :-1, :]

        # gateloop layers

        if not isinstance(self.fine_gateloop_block, nn.Identity):
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> b (nf n) d')
            orig_length = fine_vertex_codes.shape[-2]
            fine_vertex_codes = fine_vertex_codes[:, :(code_len + 1)]

            fine_vertex_codes, fine_gateloop_cache = self.fine_gateloop_block(fine_vertex_codes, cache = fine_gateloop_cache)

            fine_vertex_codes = pad_to_length(fine_vertex_codes, orig_length, dim = -2)
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # fine attention - 2nd stage

        if exists(cache):
            fine_vertex_codes = fine_vertex_codes[:, -1:]

        one_face = fine_vertex_codes.shape[1] == 1

        fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> (b nf) n d')

        if one_face:
            fine_vertex_codes = fine_vertex_codes[:, :(curr_vertex_pos + 1)]

        attended_vertex_codes, fine_cache = self.fine_decoder(
            fine_vertex_codes,
            cache = fine_cache,
            return_hiddens = True
        )

        if not should_cache_fine:
            fine_cache = None

        if not one_face:
            # reconstitute original sequence

            embed = rearrange(attended_vertex_codes, '(b nf) n d -> b (nf n) d', b = batch)
            embed = embed[:, :(code_len + 1)]
        else:
            embed = attended_vertex_codes

        # logits

        logits = self.to_logits(embed)

        if not return_loss:
            if not return_cache:
                return logits

            next_cache = (
                attended_face_codes,
                coarse_cache,
                fine_cache,
                coarse_gateloop_cache,
                fine_gateloop_cache
            )

            return logits, next_cache

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return ce_loss
