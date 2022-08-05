from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoProcessor, Wav2Vec2ProcessorWithLM
from datasets import Audio, load_dataset, load_from_disk
import torch
processor = Wav2Vec2ProcessorWithLM.from_pretrained("KaranChand/ATC_arpa")

model = AutoModelForCTC.from_pretrained("models/wav2vec2-XLSR-ft-1000", local_files_only=True)
filename = "transcribed_xlsr_ft_1000_ARPA"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# loading and preprocessing of data
# atcosim = load_from_disk("atcosim_split")['validation']                             # when arrow dataset is loaded on disk, not online or csv
# atcosim = load_dataset('csv', data_files='data/pruneddata.csv', split='train')      # for making a full dataset with input values
atcosim = load_dataset("KaranChand/atcosim_pruned", split = "test")                   # for transcribing and comparing on test cases
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
# atcosim.save_to_disk(filename)
# pickle.dump(atcosim, open("output/"+filename+".p", "wb"))
atcosim = atcosim.remove_columns(['input_values'])
atcosim.to_csv("output/"+filename+".csv", index = False, header=True)