from datasets import load_dataset, Audio, ClassLabel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import torch
from jiwer import wer
from pathlib import Path
import random
import pandas as pd

# define pipeline
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# loading and preprocessing of data
atcosim = load_dataset('csv', data_files='newdata.csv', split='train')
atcosim = atcosim.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# shows 10 random elements and outputs them in ShowRandomElements.csv
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    df.to_csv(Path("ShowRandomElements.csv"))

# show_random_elements(atcosim, num_examples=10)

# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# print([d["array"] for d in atcosim[:5]["audio"]])

# TRANSCRIBE ATCOSIM DATA HERE USING WAV2VEC2










