"""
Terrain Diffusion — live demo (Streamlit)
Generates elevation heightmaps with a DDPM diffusion model trained from
scratch on real Alps DEM data, optionally scaled to a large seamless map
with MultiDiffusion. See: github.com/Zee-daher/terrain-diffusion-ue5
"""

import numpy as np
import torch
import streamlit as st
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image

MODEL_ID = "Zee-daher/terrain-diffusion-ue5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    """Loaded once per server session, then reused across every request."""
    model = UNet2DModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    return model, scheduler


def to_display_image(arr: np.ndarray) -> Image.Image:
    """Normalize a raw model output array to an 8-bit grayscale PIL image."""
    g = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return Image.fromarray((g * 255).astype(np.uint8))


@torch.no_grad()
def generate_single(model, scheduler, steps: int, progress_bar) -> np.ndarray:
    """Reverse diffusion on a single 256x256 patch, starting from pure noise."""
    sample = torch.randn(1, 1, 256, 256, device=DEVICE)
    scheduler.set_timesteps(steps)
    total = len(scheduler.timesteps)
    for i, t in enumerate(scheduler.timesteps):
        noise_pred = model(sample, t).sample
        sample = scheduler.step(noise_pred, t, sample).prev_sample
        progress_bar.progress((i + 1) / total)
    return sample[0, 0].cpu().float().numpy()


@torch.no_grad()
def generate_multidiffusion(model, scheduler, out_size: int, patch: int,
                             stride: int, steps: int, progress_bar) -> np.ndarray:
    """
    Large, seamless heightmap via MultiDiffusion (Bar-Tal et al., 2023).
    Overlapping patch windows are denoised together and blended at every
    step, not just at the end, so neighboring windows stay consistent.
    """
    canvas = torch.randn(1, 1, out_size, out_size, device=DEVICE)

    coords = []
    for y in range(0, out_size - patch + 1, stride):
        for x in range(0, out_size - patch + 1, stride):
            coords.append((y, x))

    scheduler.set_timesteps(steps)
    total = len(scheduler.timesteps)
    for i, t in enumerate(scheduler.timesteps):
        acc = torch.zeros_like(canvas)
        cnt = torch.zeros_like(canvas)
        for (y, x) in coords:
            tile = canvas[:, :, y:y + patch, x:x + patch]
            noise_pred = model(tile, t).sample
            denoised = scheduler.step(noise_pred, t, tile).prev_sample
            acc[:, :, y:y + patch, x:x + patch] += denoised
            cnt[:, :, y:y + patch, x:x + patch] += 1
        canvas = acc / cnt
        progress_bar.progress((i + 1) / total)

    return canvas[0, 0].cpu().float().numpy()


st.set_page_config(page_title="Terrain Diffusion", page_icon="🏔️", layout="centered")

st.title("🏔️ Terrain Diffusion")
st.markdown(
    """
    A DDPM diffusion model trained from scratch on real Alps elevation data
    (DEM), scaled to large seamless maps with **MultiDiffusion**, and
    originally built for import into Unreal Engine 5.

    Pick a mode and generate a new heightmap from pure noise — every run
    produces a different terrain.

    [Code & full write-up on GitHub](https://github.com/Zee-daher/terrain-diffusion-ue5)
    """
)

model, scheduler = load_model()

mode = st.radio(
    "Generation mode",
    ["Single patch (256x256, DDPM)", "Large map (512x512, MultiDiffusion)"],
)
steps = st.slider(
    "Denoising steps", min_value=50, max_value=1000, value=250, step=50,
    help="Fewer steps = faster preview. 1000 = full quality (slow on CPU).",
)

if st.button("Generate terrain", type="primary"):
    progress_bar = st.progress(0)
    with st.spinner("Denoising..."):
        if mode.startswith("Single"):
            arr = generate_single(model, scheduler, steps, progress_bar)
        else:
            arr = generate_multidiffusion(model, scheduler, out_size=512,
                                           patch=256, stride=128, steps=steps,
                                           progress_bar=progress_bar)
    st.image(to_display_image(arr), caption="Generated heightmap", use_container_width=True)
