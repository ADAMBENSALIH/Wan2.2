import runpod

model = None

def load_model():
    global model
    if model is None:
        from wan.text2video import WanT2V
        from wan.config import config

        model = WanT2V(
            config=config,
            checkpoint_dir="checkpoints",
            device_id=0
        )
    return model


def handler(event):
    model = load_model()

    input_data = event["input"]

    prompt = input_data["prompt"]

    video_tensor = model.generate(
        input_prompt=prompt,
        size=(1280, 720),
        frame_num=81,
        sampling_steps=30,
        guide_scale=5.0,
        seed=-1,
        offload_model=True
    )

    return {
        "status": "success",
        "video": "generated"
    }


runpod.serverless.start({
    "handler": handler
})
