import runpod

from generate import generate  # من repo ديالك

# load model مرة وحدة (مهم)
model = None

def load_model():
    global model
    if model is None:
        from wan.text2video import WanModelClass  # عدل حسب الكود ديالك
        model = WanModelClass()
    return model


def handler(event):
    prompt = event["input"]["prompt"]

    model = load_model()

    video = model.generate(
        input_prompt=prompt,
        sampling_steps=30,
        guide_scale=5.0,
        frame_num=81
    )

    return {
        "video": video
    }


runpod.serverless.start({
    "handler": handler
})
