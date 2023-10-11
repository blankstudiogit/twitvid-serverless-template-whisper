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
from potassium import Potassium, Request, Response
import boto3

class WhisperWord:
    def __init__(self, word: str, start: int, end: int, probability: float):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability

    def to_dict(self):
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "probability": self.probability
        }

class WhisperSegment:
    def __init__(self, id: int, start: int, end: int, text: str, words: List[WhisperWord] = None):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []

    def add_word(self, word: WhisperWord):
        self.words.append(word)

    def to_dict(self):
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [word.to_dict() for word in self.words]  # Convert WhisperWord instances to dictionaries
        }


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
    processor = context.get("processor")

    # download file from the given URL
    download_file_from_url(audio_url, "sample.wav")

    # open the stored file and convert to tensors
    input_features = processor(load_audio("sample.wav"), sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # run inference on the sample
    model = context.get("model")
    generated_ids = model.generate(inputs=input_features)
    
    # convert the generated ids back to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Transcription:", transcription)

    # create WhisperWord objects for each word (dummy data for demonstration)
    words_data = [WhisperWord(word=word, start=start, end=end, probability=0.8) for word, (start, end) in enumerate(word_start_end_pairs(transcription))]

    print("words_data:", words_data)

    # create WhisperSegment object
    segment = WhisperSegment(id=1, start=0, end=len(transcription), text=transcription, words=words_data)
    segment_dict = segment.to_dict()

    print("segment:", segment)
    print("segment_dict:", segment_dict)


    # return output JSON to the client
    return Response(
        json={"outputs": {"text": transcription, "segments": [segment_dict], "language": "english"}},
        status=200
    )

import re

def word_start_end_pairs(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    # Use regular expression to find all words in the text
    words = re.findall(r'\b\w+\b', text)
    word_start_end_pairs = [(word, (text.find(word), text.find(word) + len(word))) for word in words]
    return [(word, (start, end), i) for i, (word, (start, end)) in enumerate(word_start_end_pairs)]

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
