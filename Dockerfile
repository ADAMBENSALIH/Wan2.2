FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install runpod

CMD ["python", "runpod_handler.py"]
