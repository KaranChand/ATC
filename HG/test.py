from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Audio, load_dataset, load_metric
import soundfile as sf
import torch
from jiwer import wer
import random
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np

atcosim = load_dataset('csv', data_files="newdata.csv", split="train")

# make a vocabulary
def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

# vocab = atcosim.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=atcosim.column_names)
# vocab_list = list(set(vocab["vocab"][0]))
# vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

atcosim = atcosim.cast_column("audio", Audio(sampling_rate=16_000))
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Listen to random audio data
# from playsound import playsound

# rand_int = random.randint(0, len(atcosim)-1)
# playsound(atcosim[rand_int]["audio"]["path"])
# print(atcosim[rand_int]["transcription"])

# test if data is processed correctly
# rand_int = random.randint(0, len(atcosim)-1)

# print("Target text:", atcosim[rand_int]["transcription"])
# print("Input array shape:", atcosim[rand_int]["audio"]["array"].shape)
# print("Sampling rate:", atcosim[rand_int]["audio"]["sampling_rate"])

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch

# split data in train and test
train_perc = 0.95
train_indices = random.sample(range(0, len(atcosim)-1), round(len(atcosim) * train_perc))
test_indices = set(range(0,len(atcosim)-1)) - set(train_indices) 
train = atcosim[train_indices]
test = atcosim[test_indices]

atcosim = atcosim.map(prepare_dataset, remove_columns=atcosim.column_names)

# @dataclass
# class DataCollatorCTCWithPadding:
#     """
#     Data collator that will dynamically pad the inputs received.
#     Args:
#         processor (:class:`~transformers.Wav2Vec2Processor`)
#             The processor used for proccessing the data.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence if provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#     """

#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lenghts and need
#         # different padding methods
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch

# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
# wer_metric = load_metric("wer")

# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids)
#     # we do not want to group tokens when computing the metrics
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     wer = wer_metric.compute(predictions=pred_str, references=label_str)

#     return {"wer": wer}

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-xls-r-300m", 
#     attention_dropout=0.0,
#     hidden_dropout=0.0,
#     feat_proj_dropout=0.0,
#     mask_time_prob=0.05,
#     layerdrop=0.0,
#     ctc_loss_reduction="mean", 
#     pad_token_id=processor.tokenizer.pad_token_id,
#     vocab_size=len(processor.tokenizer),
# )
# model.freeze_feature_extractor()

# from transformers import TrainingArguments

# training_args = TrainingArguments(
#   output_dir='KaranChand/HG',
#   group_by_length=True,
#   per_device_train_batch_size=16,
#   gradient_accumulation_steps=2,
#   evaluation_strategy="steps",
#   num_train_epochs=30,
#   gradient_checkpointing=True,
#   fp16=True,
#   save_steps=400,
#   eval_steps=400,
#   logging_steps=400,
#   learning_rate=3e-4,
#   warmup_steps=500,
#   save_total_limit=2,
#   push_to_hub=False,
# )
# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     data_collator=data_collator,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train,
#     eval_dataset=test,
#     tokenizer=processor.feature_extractor,
# )

# trainer.train()

input_dict = processor(atcosim[0]["input_values"], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(atcosim[0]["transcription"].lower())