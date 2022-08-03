from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoProcessor
from datasets import Audio, load_dataset, load_from_disk
import torch
import pickle

torch.cuda.empty_cache()
# define pipeline
# checkpoint = "facebook/wav2vec2-base-960h"
# checkpoint = "facebook/wav2vec2-large-robust-ft-swbd-300h"
# checkpoint = "facebook/hubert-large-ls960-ft"
checkpoint = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
model = AutoModelForCTC.from_pretrained(checkpoint)
processor = Wav2Vec2Processor.from_pretrained(checkpoint)
filename = "transcribed_xlsr"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# loading and preprocessing of data
# atcosim = load_from_disk("atcosim_split")['validation']
atcosim = load_dataset('csv', data_files='data/pruneddata.csv', split='train')
# atcosim = load_dataset("KaranChand/atcosim_pruned", split = "test")
atcosim = atcosim.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(x):
  input_values = processor(x['audio']["array"], return_tensors="pt", padding=True, sampling_rate=x['audio']["sampling_rate"]).to(device).input_values
  x['input_values'] = input_values[0]
  with processor.as_target_processor():
        x["labels"] = processor(x["transcription"]).input_ids
  logits = model(input_values).logits
  pred_id = torch.argmax(logits, dim=-1)[0]
  x['model_transcription'] = processor.decode(pred_id)
  return x

atcosim = atcosim.map(prepare_dataset, remove_columns='audio')
atcosim.save_to_disk(filename)
# pickle.dump(atcosim, open("output/"+filename+".p", "wb"))
atcosim = atcosim.remove_columns(['input_values'])
atcosim.to_csv("output/"+filename+".csv", index = False, header=True)











