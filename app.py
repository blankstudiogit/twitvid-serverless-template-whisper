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
import whisper
import torch

# create a new Potassium app
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = whisper.load_model("base")
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    # get file URL from request.json dict
    audio_url = request.json.get("audio_url")
    if audio_url is None:
        raise ValueError("audio_url is required")

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

if __name__ == "__main__":
    app.serve()


