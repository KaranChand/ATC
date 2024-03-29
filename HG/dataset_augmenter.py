import pandas as pd 
from pathlib import Path  
from datasets import load_dataset, Audio
from scipy.io.wavfile import read
import numpy as np
from datasets import Dataset
import json

# augmenting the ATCOSIM csv file to be compatible with HuggingFace
df = pd.read_csv("data/fulldata.csv") 
df = df.drop(columns = ["utterance_id", "recording_corrupt", "comment_transcriptionist", "speaker_id", "session_id", "length_sec","recording_startpos_sec"])
df = df.assign(path=lambda x: 'atcosim/WAVdata/' + x.directory + '/' + x.subdirectory + '/' + x.filename + '.wav')
df = df.drop(columns = ["directory", "subdirectory", "filename"])

df.to_csv(Path("data/newdata.csv"), index = False, header=True) 
atcosim = load_dataset('csv', data_files='data/newdata.csv', split='train')

atcosim = atcosim.rename_column("path", "audio")
atcosim.to_csv("data/newdata.csv", index = False, header=True)

# remove rows that contain unusable information
import re
chars_to_remove_regex = '[\=\~\@\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

atcosim = pd.read_csv("data/newdata.csv") 
atcosim = atcosim[atcosim["transcription"].str.contains("<OT>|<FL>|[EMPTY]|[FRAGMENT]|[HNOISE]|[NONSENSE]|[UNKNOWN]") == False]
atcosim.set_index('recording_id')
atcosim_clean = Dataset.from_pandas(atcosim)
atcosim_clean = atcosim_clean.map(remove_special_characters)
atcosim_clean = atcosim_clean.remove_columns('__index_level_0__')
print(atcosim_clean)
atcosim_clean.to_csv("data/pruneddata.csv", index = False, header=True) 



########################## dataset split
# atcosim = load_dataset('csv', data_files='data/pruneddata.csv', split='train')
# atcosim_clean = atcosim.train_test_split(train_size=0.9, seed=42)
# atcosim_main = atcosim_clean['train'].train_test_split(train_size=0.89, seed=42)
# atcosim_main["validation"] = atcosim_clean["test"]

# atcosim = DatasetDict({
#     'train': atcosim_main['train'],
#     'test': atcosim_main['test'],
#     'valid': atcosim_main['validation']})
# print(atcosim)
# atcosim.save_to_disk("atcosim_pruned")

########################## Vocabulary builder
# # make a vocabulary
# def extract_all_chars(batch):
#   all_text = " ".join(batch["transcription"])
#   vocab = list(set(all_text))
#   return {"vocab": [vocab], "all_text": [all_text]}


# vocab = atcosim.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=atcosim.column_names)
# vocab_list = list(set(vocab["vocab"][0]))
# vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)