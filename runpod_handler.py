import runpod
import torch
import torchvision
import os

# 👇 import ديال model ديالك
from wan.text2video import WanT2V
from wan.config import config  # إلا عندك config

# =========================
# 🔥 Load model ONCE (important)
# =========================
model = WanT2V(
    config=config,
    checkpoint_dir="checkpoints",
    device_id=0
)

# =========================
# 🚀 RunPod handler
# =========================
def handler(event):
    input_data = event["input"]

    prompt = input_data.get("prompt", "")
    width = input_data.get("width", 1280)
    height = input_data.get("height", 720)
    steps = input_data.get("steps", 30)
    seed = input_data.get("seed", -1)

    # =========================
    # 🎬 Generate video
    # =========================
    video_tensor = model.generate(
        input_prompt=prompt,
        size=(width, height),
        frame_num=81,
        sampling_steps=steps,
        guide_scale=5.0,
        seed=seed,
        offload_model=True
    )

    # =========================
    # 💾 Save MP4
    # =========================
    output_path = "/tmp/output.mp4"

    # tensor format: (C, T, H, W)
    video_tensor = video_tensor.permute(1, 2, 3, 0).cpu()

    torchvision.io.write_video(
        output_path,
        video_tensor,
        fps=16
    )

    return {
        "video_path": output_path,
        "status": "success"
    }

# =========================
# 🔌 Start RunPod serverless
# =========================
runpod.serverless.start({
    "handler": handler
})
