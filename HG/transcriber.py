from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets import Audio, load_dataset
import torch

torch.cuda.empty_cache()
# define pipeline
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
# model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
filename = "transcribed_robust"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# loading and preprocessing of data
atcosim = load_dataset('csv', data_files='data/newdata.csv', split='train')
atcosim = atcosim.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(x):
  x['input_values'] = processor(x['audio']["array"], sampling_rate=x['audio']["sampling_rate"]).input_values[0]
  input_dict = processor(x['input_values'], return_tensors="pt", padding=True).to(device)
  with processor.as_target_processor():
        x["labels"] = processor(x["transcription"]).input_ids
  logits = model(input_dict.input_values).logits
  pred_id = torch.argmax(logits, dim=-1)[0]
  x['model_transcription'] = processor.decode(pred_id)
  return x

atcosim = atcosim.map(prepare_dataset, remove_columns='audio')
import pickle
pickle.dump(atcosim, open("output/"+filename+".p", "wb"))
atcosim.to_csv("output/"+filename+".csv", index = False, header=True)












