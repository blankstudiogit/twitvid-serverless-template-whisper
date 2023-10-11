from potassium import Potassium, Request, Response
import torch
import os
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torchaudio
import requests
from typing import Tuple, List
import time
import numpy as np

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

    # get file URL from request.json dict
    audio_url = request.json.get("audio_url")
    if audio_url is None:
        raise ValueError("audio_url is required")

    processor = context.get("processor")

    # download file from the given URL
    audio_path = download_audio_from_url(audio_url)

    # run inference on the sample
    model = context.get("model")

    temperature = 0
    temperature_increment_on_fallback = 0.2
    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]

    # Run the model
    args = {
        "language": None,
        "patience": None,
        "suppress_tokens": "-1",
        "initial_prompt": None,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": True,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、"
    }
    
    outputs = model.transcribe(str(audio_path), temperature=temperature, **args)
    start = time.time()
    outputs = model.transcribe(str(audio_path), temperature=temperature, **args)
    end = time.time()

    output = {"outputs": outputs}
    os.remove(audio_path)

    # Return the results as a dictionary
    return output

def download_audio_from_url(url):
    # Extract the filename from the URL
    filename = os.path.basename(url.split("?")[0])

    # Download the audio file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Save the downloaded file
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filename

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
