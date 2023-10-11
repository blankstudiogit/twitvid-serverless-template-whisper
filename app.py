from potassium import Potassium, Request, Response
import torch
import os
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torchaudio
import requests

# create a new Potassium app
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    config = WhisperConfig.from_pretrained("openai/whisper-base")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")
    
    with init_empty_weights():
        model = WhisperForConditionalGeneration(config)
    model.tie_weights()

    model = load_checkpoint_and_dispatch(
        model, "model.safetensors", device_map="auto"
    )
   
    context = {
        "model": model,
        "processor": processor,
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
   device = get_device()
    audio_url = request.json.get("audio_url")
    processor = app.state.context.get("processor")

    download_file_from_url(audio_url, "sample.wav")
    input_features = processor(load_audio("sample.wav"), sampling_rate=16000, return_tensors="pt").input_features.to(device)
    model = app.state.context.get("model")
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"outputs": transcription}

    # return output JSON to the client
    # return Response(
    #     json={"outputs": transcription},
    #     status=200
    # )

# Implement a function to download file from the given URL
def download_file_from_url(url, file_path):
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

# Note that since this function doesn't have a decorator, it's not a handler
def load_audio(audio_path):
    """Loads audio file into tensor and resamples to 16kHz"""
    speech, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    
    return speech.squeeze()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on MPS")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    return device

if __name__ == "__main__":
    app.serve()
