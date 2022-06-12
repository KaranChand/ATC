from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Audio, Dataset
import soundfile as sf
import torch

# load model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
# load dummy dataset and read soundfiles
ds = Dataset.cast_column("data", Audio())

#tokenize
input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)