from datasets import load_dataset, Audio, ClassLabel
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline, HubertForCTC
import torch
from jiwer import wer
from pathlib import Path
import random
import pandas as pd

# define pipeline
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")

# loading and preprocessing of data
atcosim = load_dataset('csv', data_files='newdata.csv', split='train[:60]')
atcosim = atcosim.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids

    input_dict = processor(batch["input_values"], return_tensors="pt", padding=True)
    logits = model(input_dict.input_values).logits
    batch['pred_ids'] = torch.argmax(logits, dim=-1)[0]
    
    return batch

# atcosim = atcosim.map(prepare_dataset, num_proc=4)

# atcosim.to_csv("transcribed.csv", index = False, header=True)


# print(transcription)
from playsound import playsound


print(atcosim[57]['transcription'])
playsound(atcosim[57]["audio"]["path"])

# # shows 10 random elements and outputs them in ShowRandomElements.csv
# def show_random_elements(dataset, num_examples=10):
#     assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
#     picks = []
#     for _ in range(num_examples):
#         pick = random.randint(0, len(dataset)-1)
#         while pick in picks:
#             pick = random.randint(0, len(dataset)-1)
#         picks.append(pick)
    
#     df = pd.DataFrame(dataset[picks])
#     df.to_csv(Path("ShowRandomElements.csv"))

# show_random_elements(atcosim, num_examples=10)




