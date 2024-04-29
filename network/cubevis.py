# %% imports
import os.path as op
import re
from glob import glob

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# %% paths
base_path = op.join(
    "/media",
    "MeMoSLAP_Subjects",
    "derivatives",
    "automated_electrode_extraction",
    "sub-*",
    "electrode_extraction",
    "ses-*",
    "run-*",
    "cut_cubes",
)
images = glob(op.join(base_path, "volume_*.nii.gz"))
images.sort()
mask = glob(op.join(base_path, "mask_*.nii.gz"))
mask.sort()


# %% Sidebar
def format_sidebar_label(x: str) -> str:
    return "_".join(re.findall("(sub-[0-9]+|ses-[0-9]+|run-[0-9]+|volume_[0-9])", x))


def submit_form(fits):
    checked[checked.img == select_img]["mask_fits"] = fits
    checked.to_csv(CHECKED_PATH, sep="\t", index=False)


with st.sidebar:
    select_img = st.selectbox(
        label="Show Cube:", options=images, format_func=format_sidebar_label
    )

    if select_img not in st.session_state:
        st.session_state[select_img] = False

    select_mask = mask[images.index(select_img)]
    "### Volume:"
    st.write(select_img)
    "### Mask:"
    st.write(select_mask)

    # fits = st.toggle(
    #     "Electrode mask is good",
    #     st.session_state[select_img],
    # )
    #
    # st.session_state[select_img] = fits
    #
    # st.session_state

with st.container():

    nii = nib.load(select_img)

    img = np.array(nii.dataobj).T

    mask_nii = nib.load(select_mask)
    mask_img = np.array(mask_nii.dataobj).T

    nb_frames, r, c = img.shape

    def z(k):
        return (6.7 - k * 0.1) * np.ones((r, c))

    fig = go.Figure(
        [
            go.Surface(
                z=z(0),
                surfacecolor=np.flipud(img[nb_frames - 1]),
                colorscale="Gray",
                visible=True,
                legend="legend1",
                opacity=0.9,
            ),
            go.Surface(
                z=z(0),
                surfacecolor=np.flipud(mask_img[nb_frames - 1]),
                cmin=0,
                cmax=1,
                colorscale="hot",
                opacity=0.4,
                visible=True,
                legend="legend2",
            ),
        ],
    )

    frames = [
        go.Frame(
            data=[
                go.Surface(z=z(k), surfacecolor=np.flipud(img[nb_frames - k - 1])),
                go.Surface(z=z(k), surfacecolor=np.flipud(mask_img[nb_frames - k - 1])),
            ],
            traces=[0, 1],
            name=str(k),
        )
        for k in range(nb_frames)
    ]

    fig.update(frames=frames)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(
        width=900,
        height=900,
        title="Slice in cut cubes",
        scene={
            "zaxis": {"range": [-0.1, 6.8], "autorange": False},
            "aspectratio": {"x": 1, "y": 1, "z": 1},
        },
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    st.plotly_chart(fig, use_container_width=True)
