import evaluate
from datasets import load_dataset, load_metric
import pandas as pd 
from pathlib import Path  

# remove rows that have NaN as model_transcription
df = pd.read_csv("transcribed_base.csv") 
clean_df = df.dropna(subset=['model_transcription'])
clean_df.to_csv(Path("to_eval.csv"), index = False, header=True)
print("removed %x model_transcriptions that contain NaN" % (len(df) - len(clean_df)))

# load as dataset
ds = load_dataset('csv', data_files='to_eval.csv', split='train')

wer_metric = evaluate.load("wer")
wer = wer_metric.compute(predictions=ds['model_transcription'], references=ds['transcription'])
# perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = ds[:5]['transcription']
# results = perplexity.compute(model_id='gpt2',
#                              input_texts=input_texts)

# print({"perplexity": round(results["mean_perplexity"], 2)})
print({"wer": wer})