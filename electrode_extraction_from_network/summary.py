from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt

def summary_grid(
    img_batch: torch.Tensor, mask_batch: torch.Tensor, nrow: int = 3
) -> plt.figure:
    img_grid = make_grid(img_batch, nrow=nrow).permute(1, 2, 0)
    g = b = torch.zeros(mask_batch.shape)
    rgb_mask = torch.cat((mask_batch, g, b), dim = 1) 
    mask_grid = make_grid(rgb_mask, nrow=nrow).permute(1, 2, 0)
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(img_grid, figure=fig)
    plt.imshow(mask_grid, alpha=0.3, cmap="Reds", figure=fig)

    return fig

def show_batch(img_batch, mask_batch, n_cols=3):
    """
    Plot all images and masks of a batch as subplots with masks overlayed in red and opacity
    """

    n_axs = n_img = len(img_batch)

    if n_img % n_cols != 0:
        n_axs += n_cols - (n_img % n_cols)

    _, axs = plt.subplots(
        int(n_axs / n_cols), n_cols, figsize=(15, 5 * int(n_axs / n_cols))
    )

    for img, mask, ax in zip(img_batch, mask_batch, axs.flatten()):
        ax.imshow(img.permute(1, 2, 0))
        ax.imshow(mask.squeeze(), alpha=0.3, cmap="Reds")

    return axs