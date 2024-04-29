import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from util.io import load_nifti, save_nifti

base_path = op.join(
    "/media",
    "MeMoSLAP_Subjects",
    "derivatives",
    "automated_electrode_extraction",
    "sub-003",
    "electrode_extraction",
    "ses-1",
    "run-01",
)

petra_path = op.join(
    base_path,
    "petra_cut_pads.nii.gz",
)
mask_path = op.join(
    base_path,
    "cylinder_plus_plug_ROI.nii.gz",
)

skull_mask_path = op.join(
    base_path,
    "finalmask.nii.gz",
)

petra_nifti, petra_img = load_nifti(petra_path)
mask_nifti, mask_img = load_nifti(mask_path)
skull_nifti, skull_img = load_nifti(skull_mask_path)

inv_skull_img = np.where(skull_img == 0, 1, 0)

slice_is_null = np.sum(mask_img, axis=(1, 2))
slice_idx = np.where(slice_is_null != 0)[0]

# petra_prep = np.zeros_like(petra_img)
# mask_segmented = np.zeros_like(mask_img)

petra_prep = sitk.GetImageFromArray(petra_img)
petra_prep = sitk.Cast(petra_prep, sitk.sitkFloat32)
petra_prep = sitk.AdaptiveHistogramEqualization(
    petra_prep,
    radius=(5, 5, 5),
)
petra_prep = sitk.BlackTopHat(
    petra_prep,
    kernelType=sitk.sitkBall,
    kernelRadius=(3, 3, 3),
)
# petra_prep = sitk.WhiteTopHat(
#     petra_prep,
#     kernelType=sitk.sitkBall,
#     kernelRadius=(5, 5, 5),
# )
petra_prep = sitk.CannyEdgeDetection(
    petra_prep,
    lowerThreshold=30.0,
    upperThreshold=95.0,
    variance=(2, 2, 2),
)
# petra_prep = sitk.BinaryFillhole(sitk.Cast(petra_prep, sitk.sitkUInt8))
petra_prep = sitk.GetArrayFromImage(sitk.Cast(petra_prep, sitk.sitkFloat32))


def mask_dilation(in_mask, kernel_radius=(1, 1, 1)):
    in_mask = sitk.Cast(sitk.GetImageFromArray(in_mask), sitk.sitkUInt8)
    in_mask = sitk.BinaryDilate(in_mask, kernelRadius=kernel_radius)
    return sitk.GetArrayFromImage(in_mask)


def mask_erosion(in_mask, kernel_radius=(1, 1, 1)):
    in_mask = sitk.Cast(sitk.GetImageFromArray(in_mask), sitk.sitkUInt8)
    in_mask = sitk.BinaryErode(in_mask, kernelRadius=kernel_radius)
    return sitk.GetArrayFromImage(in_mask)


class IndexTracker:
    def __init__(
        self,
        ax_obj,
        slices,
        petra_img,
        edges_img,
    ) -> None:
        self.slice = slices[0]
        self.slices = slices
        self.ax = ax_obj
        self.petra = petra_img
        self.edges = edges_img
        self.im = self.ax.imshow(self.petra[self.slice], cmap="gray")
        self.im_overlay = self.ax.imshow(
            self.edges[self.slice], cmap="Reds", alpha=0.7
        )
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == "up" else -1
        max_index = self.slices[-1]
        min_index = self.slices[0]
        self.slice = np.clip(self.slice + increment, min_index, max_index)
        self.update()

    def update(self):
        im_data = self.im.to_rgba(
            self.petra[self.slice], alpha=self.im.get_alpha()
        )
        im_overlay_data = self.im_overlay.to_rgba(
            self.edges[self.slice], alpha=self.im_overlay.get_alpha()
        )
        self.im.set_data(im_data)
        self.im_overlay.set_data(im_overlay_data)
        self.ax.set_title(f"Use scroll wheel to navigate\nslice {self.slice}")
        self.im.axes.figure.canvas.draw()
        self.im_overlay.axes.figure.canvas.draw()


def to_img(img, totype=np.uint8):
    info = np.iinfo(totype)
    a = (info.max - info.min) / (img.max() - img.min())
    b = info.min - a * img.min()
    return (a * img + b).astype(totype)

    # for i in slice_idx:
    #     petra_slice = petra_img[i].astype(np.float32)
    #     mask = mask_img[i].astype(np.float32)
    #     inv_skull_slice = inv_skull_img[i]
    #
    #     mask = mask_dilation(mask, kernel_radius=(5, 5, 5))
    #     inv_skull_slice = mask_erosion(inv_skull_slice, kernel_radius=(5, 5, 5))
    #
    #     petra_sl = sitk.Cast(
    #         sitk.GetImageFromArray(
    #             petra_slice
    #             # * mask
    #             # * inv_skull_slice
    #         ),
    #         sitk.sitkFloat32,
    #     )
    #
    # petra_sl = sitk.BlackTopHat(
    #     petra_sl,
    #     kernelType=sitk.sitkBall,
    #     kernelRadius=(5, 5, 5),
    # )
    #
    # petra_sl = sitk.WhiteTopHat(
    #     petra_sl,
    #     kernelType=sitk.sitkBall,
    #     kernelRadius=(3, 3, 3),
    # )

    # edges = sitk.CannyEdgeDetection(
    #     # sitk.DiscreteGaussian(
    #     #     petra_sl,
    #     #     variance=[2, 2, 2],
    #     # ),
    #     petra_sl,
    #     lowerThreshold=30.0,
    #     upperThreshold=95.0,
    #     variance=(1, 1, 1),
    # )

    edges = sitk.BinaryFillhole(sitk.Cast(edges, sitk.sitkUInt8))
    edges = sitk.GetArrayFromImage(sitk.Cast(edges, sitk.sitkFloat32))
    petra_sl = sitk.GetArrayFromImage(petra_sl)
    print(petra_sl.sum(), edges.sum())

    petra_prep[i] = petra_sl
    mask_segmented[i] = edges


# petra_img = to_img(petra_img)
# mask_segmented = to_img(mask_segmented)
# petra_img = np.clip(
#     petra_img,
#     np.percentile(petra_img, 1),
#     np.percentile(petra_img, 99),
# )

# rgb_mask = np.stack(
#     [
#         mask_segmented,
#         np.zeros_like(mask_segmented),
#         np.zeros_like(mask_segmented),
#     ],
#     axis=-1,
# )

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(
    ax_obj=ax,
    slices=np.where(petra_prep.sum(axis=(1, 2)) != 0)[0],
    petra_img=petra_img,
    edges_img=petra_prep,
)
fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
plt.show()

# save_nifti(
#     img=mask_segmented,
#     path=op.join(base_path, "mask_segmented.nii.gz"),
#     ref=mask_nifti,
# )
#
# save_nifti(
#     img=omask_segmented,
#     path=op.join(base_path, "omask_segmented.nii.gz"),
#     ref=mask_nifti,
# )
