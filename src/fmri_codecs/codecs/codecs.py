import io
import zstd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Parcellate(nn.Module):
    parc: Tensor
    parc_ids: Tensor

    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        n_parcels: int,
        max_size: int,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.n_parcels = n_parcels
        self.max_size = max_size
        self.register_buffer("parc", torch.zeros(normalized_shape, dtype=torch.int64))
        self.register_buffer("parc_ids", torch.zeros(n_parcels, max_size, dtype=torch.int64))

    def load_parcellation(self, parc: Tensor):
        # parcel_indices is an array of shape (num_parcels, max_size), where max_size is
        # the size (in voxels) of the biggest parcel. The ith row contains the indices of
        # the voxels belonging to the ith parcel.
        parc_ids = get_parcel_indices(parc[parc > 0])

        self.normalized_shape = parc.shape
        self.n_parcels, self.max_size = parc_ids.shape
        self.register_buffer("parc", parc.long())
        self.register_buffer("parc_ids", parc_ids.long())

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
    n_parcels = len(parc_onehot)
    # size of biggest parcel
    max_count = parc_onehot.sum(dim=1).max().item()

    # get voxel indices of each parcel. fill the rest with -1.
    parc_ids = torch.full((n_parcels, max_count), fill_value=-1, dtype=torch.int64)
    for ii, mask in enumerate(parc_onehot):
        mask_ids = mask.nonzero().flatten()
        parc_ids[ii, : len(mask_ids)] = mask_ids
    return parc_ids


class PatchPCA(nn.Module):
    def __init__(self, seq_len: int, in_features: int, n_components: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.in_features = in_features
        self.n_components = n_components

        # projection weights from native parcel dimension to target embedding dimension.
        P, S, d = seq_len, in_features, n_components
        self.weight = nn.Parameter(torch.empty(P, d, S), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(P, S), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        # random orth basis
        P, d, S = self.weight.shape
        weight, _ = torch.linalg.qr(torch.randn(P, S, d))
        weight = weight.transpose(1, 2).contiguous()
        self.weight.data.copy_(weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x - self.bias
        if mask is not None:
            x = x * mask
        x = (x.transpose(0, 1) @ self.weight.transpose(1, 2)).transpose(0, 1)  # [N, P, d]
        return x

    def inverse(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = (x.transpose(0, 1) @ self.weight).transpose(0, 1)  # [N, P, S]
        x = x + self.bias
        if mask is not None:
            x = x * mask
        return x

    def fit(self, x: torch.Tensor, mask: Tensor | None = None):
        N, P, S = x.shape
        if mask is not None:
            x = x * mask

        bias = x.mean(dim=0)
        x = x - bias

        x = x.transpose(0, 1)  # [P, N, S]
        _, s, vt = torch.linalg.svd(x, full_matrices=False)

        weight = vt[:, : self.n_components, :]
        # flip sign so that most of the values are positive
        sign = torch.sign(weight.sum(dim=-1, keepdim=True))
        weight = weight * sign

        if mask is not None:
            weight = weight * mask[:, None, :]

        self.weight.data.copy_(weight)
        self.bias.data.copy_(bias)
        return self


class Quantize(nn.Module):
    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        max_bins: int = 4096,
        std_range: float = 4.0,
        dtype: torch.dtype = torch.int16,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.max_bins = max_bins
        self.std_range = std_range
        self.dtype = dtype

        self.scale = nn.Parameter(torch.ones(normalized_shape), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        sigma_max = self.scale.amax(dim=-1, keepdim=True)
        bin_width = 2 * self.std_range * sigma_max / self.max_bins
        x = torch.round(x / bin_width)
        info = torch.iinfo(self.dtype)
        x = x.clip(info.min, info.max).to(self.dtype)
        return x

    def inverse(self, x: Tensor) -> Tensor:
        sigma_max = self.scale.amax(dim=-1, keepdim=True)
        bin_width = 2 * self.std_range * sigma_max / self.max_bins
        x = x * bin_width
        return x

    def fit(self, x: Tensor):
        scale = x.std(dim=0)
        self.scale.data.copy_(scale)
        return self


class ParcelPCAQuantize(nn.Module):
    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        n_parcels: int,
        max_size: int,
        n_components: int = 64,
        max_bins: int = 4096,
        std_range: float = 4.0,
        dtype: torch.dtype = torch.int16,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.n_parcels = n_parcels
        self.max_size = max_size
        self.n_components = n_components
        self.max_bins = max_bins
        self.std_range = std_range
        self.dtype = dtype

        self.parcellate = Parcellate(normalized_shape, n_parcels, max_size)
        self.pca = PatchPCA(n_parcels, max_size, n_components=n_components)
        self.quantize = Quantize(
            (n_parcels, n_components),
            max_bins=max_bins,
            std_range=std_range,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.parcellate(x)
        x = self.pca(x, mask=self.parcellate.parc_ids >= 0)
        x = self.quantize(x)
        return x

    def inverse(self, x: Tensor) -> Tensor:
        x = self.quantize.inverse(x)
        x = self.pca.inverse(x, mask=self.parcellate.parc_ids >= 0)
        x = self.parcellate.inverse(x)
        return x

    def fit(self, x: Tensor):
        x = self.parcellate(x)
        mask = self.parcellate.parc_ids >= 0
        self.pca.fit(x, mask=mask)
        x = self.pca.forward(x, mask=mask)
        self.quantize.fit(x)
        return self

    @classmethod
    def from_parcellation(
        cls,
        parc: Tensor,
        n_components: int = 64,
        max_bins: int = 4096,
        std_range: float = 4.0,
        dtype: torch.dtype = torch.int16,
    ):
        parc = torch.as_tensor(parc)
        parc_ids = get_parcel_indices(parc[parc > 0])
        P, S = parc_ids.shape
        model = cls(
            parc.shape,
            n_parcels=P,
            max_size=S,
            n_components=n_components,
            max_bins=max_bins,
            std_range=std_range,
            dtype=dtype,
        )
        model.parcellate.load_parcellation(parc)
        return model


def encode_tensor(x: Tensor, compress: bool = True) -> bytes:
    x = x.cpu().numpy()
    with io.BytesIO() as f:
        np.save(f, x)
        x = f.getvalue()
    if compress:
        x = zstd.compress(x)
    return x


def decode_tensor(x: bytes, compress: bool = True) -> Tensor:
    if compress:
        x = zstd.decompress(x)
    with io.BytesIO(x) as f:
        x = np.load(f)
    x = torch.from_numpy(x)
    return x
