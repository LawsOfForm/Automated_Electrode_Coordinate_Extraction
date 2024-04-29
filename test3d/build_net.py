import torch
import torch.nn as nn
from torch.nn.functional import relu


def encoder(
    in_chan: int,
    out_chan: int,
    conv: dict = {"kernel_size": 3, "padding": 1},
    pool: None | dict = {"kernel_size": 2, "stride": 2},
) -> tuple[nn.Conv3d, nn.Conv3d, nn.Maxpool3d] | tuple[nn.Conv3d, nn.Conv3d]:
    en1 = nn.Conv3d(
        in_chan,
        out_chan,
        kernel_size=conv["kernel_size"],
        padding=conv["padding"],
    )
    en2 = nn.Conv3d(
        out_chan,
        out_chan,
        kernel_size=conv["kernel_size"],
        padding=conv["padding"],
    )
    if pool is None:
        return (en1, en2)
    pooln = nn.MaxPool3d(kernel_size=pool["kernel_size"], stride=pool["stride"])
    return (en1, en2, pooln)


def decoder(
    in_chan: int,
    out_chan: int,
    conv: dict = {"kernel_size": 3, "padding": 1},
    transpose: dict = {"kernel_size": 2, "stride": 2},
) -> tuple[nn.ConvTranspose3d, nn.Conv3d, nn.Conv3d]:
    upconvn = nn.ConvTranspose3d(
        in_chan,
        out_chan,
        kernel_size=transpose["kernel_size"],
        stride=transpose["stride"],
    )
    dn1 = nn.Conv3d(
        in_chan,
        out_chan,
        kernel_size=conv["kernel_size"],
        padding=conv["padding"],
    )
    dn2 = nn.Conv3d(
        in_chan,
        out_chan,
        kernel_size=conv["kernel_size"],
        padding=conv["padding"],
    )
    return (upconvn, dn1, dn2)


def forward_encoder(
    x: torch.Tensor, en1: nn.Conv3d, en2: nn.Conv3d, pooln: None | nn.MaxPool3d
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    xen1 = relu(en1(x))
    xen2 = relu(en2(xen1))
    if pooln is None:
        return xen2
    xpn = pooln(xen2)
    return xen2, xpn


def forward_decoder(
    x: torch.Tensor,
    enc_feature_map: torch.Tensor,
    upconvn: nn.ConvTranspose3d,
    dn1: nn.Conv3d,
    dn2: nn.Conv3d,
) -> torch.Tensor:
    xun = upconvn(x)
    xun1 = torch.cat([xun, enc_feature_map], dim=1)
    xdn1 = relu(dn1(xun1))
    xdn2 = relu(dn2(xdn1))

    return xdn2
