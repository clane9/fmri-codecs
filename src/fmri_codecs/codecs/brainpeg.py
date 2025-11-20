import zlib
from typing import Literal

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstd
from torch import Tensor
from scipy.sparse import coo_array
from neuromaps.datasets import fetch_fslr

from fmri_codecs import register_codec
from fmri_codecs.utils import fetch_schaefer, get_cifti_surf_data, encode_numpy, decode_numpy

NUM_VERTICES = 64984


class BrainPEGCodec(nn.Module):
    def __init__(
        self,
        num_rois: int = 400,
        dim: int = 32,
        n_bins: int = 4096,
        vmax: float = 5.0,
        compression: Literal["gzip", "zstd", "none"] = "zstd",
    ):
        super().__init__()
        self.num_rois = num_rois
        self.dim = dim
        self.n_bins = n_bins
        self.vmax = vmax
        self.compression = compression

        parc = load_schaefer(num_rois)
        self.parcellate = Parcellate(parc)

        self.proj = ParcelLinear(num_rois, self.parcellate.max_size, dim)
        weight = load_schaefer_spectral_basis(num_rois, dim)
        weight = self.parcellate.forward(weight)  # [d, P, S]
        weight = weight.transpose(0, 1).contiguous()  # [P, d, S]
        self.proj.weight.data.copy_(weight)

        self.scale = nn.Parameter(torch.ones((num_rois, 1)), requires_grad=False)

        self.bin_width = 2 * self.vmax / self.n_bins
        self.compress = {
            "gzip": zlib.compress,
            "zstd": zstd.compress,
            "none": _noop,
        }[compression]
        self.decompress = {
            "gzip": zlib.decompress,
            "zstd": zstd.decompress,
            "none": _noop,
        }[compression]

    def forward(self, x: Tensor):
        x = self.parcellate(x)
        x = self.proj(x, mask=self.parcellate.parc_ids >= 0)
        return x

    def inverse(self, x: Tensor):
        x = self.proj.inverse(x, mask=self.parcellate.parc_ids >= 0)
        x = self.parcellate.inverse(x)
        return x

    def encode(self, x: np.ndarray) -> bytes:
        x = torch.as_tensor(x, dtype=self.scale.dtype, device=self.scale.device)
        x = self.forward(x)
        x = x / self.scale
        x = x.cpu().numpy()

        x = np.round(x / self.bin_width)
        info = np.iinfo(np.int16)
        x = np.clip(x, info.min, info.max)
        x = x.astype(np.int16)
        x = encode_numpy(x)
        x = self.compress(x)
        return x

    def decode(self, x: bytes) -> np.ndarray:
        x = self.decompress(x)
        x = decode_numpy(x)
        x = x * self.bin_width

        x = torch.as_tensor(x, dtype=self.scale.dtype, device=self.scale.device)
        x = x * self.scale
        x = self.inverse(x)
        x = x.cpu().numpy()
        return x

    def fit(self, X: list[np.ndarray]):
        coefs = []
        for x in X:
            x = torch.as_tensor(x, dtype=self.scale.dtype, device=self.scale.device)
            x = self.forward(x)
            coefs.append(x[:, :, :1])
        coefs = torch.cat(coefs)
        scale = torch.std(coefs, axis=0)
        self.scale.data.copy_(scale)
        return self


def _noop(x):
    return x


class Parcellate(nn.Module):
    parc: Tensor
    parc_ids: Tensor

    def __init__(self, parc: Tensor):
        super().__init__()
        # parcel_indices is an array of shape (num_parcels, max_size), where max_size is
        # the size (in voxels) of the biggest parcel. The ith row contains the indices of
        # the voxels belonging to the ith parcel.
        parc = torch.as_tensor(parc, dtype=torch.int32)
        parc_ids = get_parcel_indices(parc[parc > 0])

        self.normalized_shape = parc.shape
        self.num_rois, self.max_size = parc_ids.shape
        self.register_buffer("parc", parc)
        self.register_buffer("parc_ids", parc_ids)

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, D]
        x = x[:, self.parc > 0]  # [N, M]
        x = x[:, self.parc_ids]  # [N, P, S]
        x = (self.parc_ids >= 0) * x
        return x

    def inverse(self, x: Tensor) -> Tensor:
        # get valid values (but not in correct order)
        parc_ids_mask = self.parc_ids >= 0
        x = x[:, parc_ids_mask]  # [N, M]
        # get correct sort indices
        parc_ids_flat = self.parc_ids[parc_ids_mask]  # [M]
        ids_restore = torch.argsort(parc_ids_flat)
        # create full data and scatter values.
        N, M = x.shape
        shape = self.parc.shape
        x_ = torch.zeros((N, *shape), dtype=x.dtype, device=x.device)
        x_[:, self.parc > 0] = x[:, ids_restore]
        return x_


def get_parcel_indices(parc: torch.Tensor) -> torch.Tensor:
    """
    Get the voxel indices for each parcel.
    """
    assert parc.ndim == 1
    # convert parcellation map into one hot
    parc_onehot = F.one_hot(parc.long()).t()
    # drop background one hot map
    parc_onehot = parc_onehot[1:]
    num_rois = len(parc_onehot)
    # size of biggest parcel
    max_count = parc_onehot.sum(dim=1).max().item()

    # get voxel indices of each parcel. fill the rest with -1.
    parc_ids = torch.full((num_rois, max_count), fill_value=-1, dtype=torch.int64)
    for ii, mask in enumerate(parc_onehot):
        mask_ids = mask.nonzero().flatten()
        parc_ids[ii, : len(mask_ids)] = mask_ids
    return parc_ids


class ParcelLinear(nn.Module):
    def __init__(self, num_rois: int, in_features: int, out_features: int):
        super().__init__()
        self.num_rois = num_rois
        self.in_features = in_features
        self.out_features = out_features

        # projection weights from native parcel dimension to target embedding dimension.
        P, S, d = num_rois, in_features, out_features
        self.weight = nn.Parameter(torch.empty(P, d, S), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        # random orth basis
        P, d, S = self.weight.shape
        weight, _ = torch.linalg.qr(torch.randn(P, S, d))
        weight = weight.transpose(1, 2).contiguous()
        self.weight.data.copy_(weight)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            x = x * mask
        x = (x.transpose(0, 1) @ self.weight.transpose(1, 2)).transpose(0, 1)  # [N, P, d]
        return x

    def inverse(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = (x.transpose(0, 1) @ self.weight).transpose(0, 1)  # [N, P, S]
        if mask is not None:
            x = x * mask
        return x


def load_schaefer_spectral_basis(num_rois: int, dim: int) -> np.ndarray:
    print("loading fslr adjacency matrix")
    A = load_fslr_adjacency()
    print(f"loading schaefer {num_rois} parcellation")
    parc = load_schaefer(num_rois)
    print("fitting per-roi spectral basis")
    weight = np.zeros((dim, len(parc)))
    for roi_id in range(1, num_rois + 1):
        mask = parc == roi_id
        Ai = A[mask, :][:, mask]
        Ai = Ai.todense()
        ui = find_spectral_basis(Ai)
        ui = ui[:dim]
        weight[: len(ui), mask] = ui
    return weight


def load_schaefer(num_rois: int) -> np.ndarray:
    path = fetch_schaefer(num_rois)
    img = nib.load(path)
    parc = get_cifti_surf_data(img)
    return parc.squeeze()


def load_fslr_adjacency(density: str = "32k") -> coo_array:
    paths = fetch_fslr(density=density)
    path_lh, path_rh = paths["midthickness"]

    surf_lh = nib.load(path_lh)
    surf_rh = nib.load(path_rh)

    polys_lh = surf_lh.darrays[1].data
    polys_rh = surf_rh.darrays[1].data
    polys = np.concatenate([polys_lh, polys_lh.max() + 1 + polys_rh])
    return polys_to_adjacency(polys)


def polys_to_adjacency(polys: np.ndarray) -> coo_array:
    n_points = polys.max() + 1
    edges = np.concatenate(
        [
            polys[:, [0, 1]],
            polys[:, [0, 2]],
            polys[:, [1, 2]],
            polys[:, [1, 0]],
            polys[:, [2, 0]],
            polys[:, [2, 1]],
        ]
    )
    mat = coo_array((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), (n_points, n_points))
    mat.sum_duplicates()
    mat = (mat > 0).astype(np.float32)
    return mat


def find_spectral_basis(A: np.ndarray):
    rd = 1 / np.sqrt(A.sum(axis=-1))
    L = np.eye(len(A)) - rd[:, None] * A * rd
    _, u = np.linalg.eigh(L)
    u = u * np.sign(u.sum(axis=0))
    return u.T


@register_codec
def bpeg_n400_d32_nb1024_none():
    return BrainPEGCodec(num_rois=400, dim=32, n_bins=1024, compression="none")


@register_codec
def bpeg_n400_d48_nb1024_none():
    return BrainPEGCodec(num_rois=400, dim=48, n_bins=1024, compression="none")


@register_codec
def bpeg_n400_d64_nb1024_none():
    return BrainPEGCodec(num_rois=400, dim=64, n_bins=1024, compression="none")
