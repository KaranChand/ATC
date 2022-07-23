from evaluate import load
from datasets import load_dataset
import pandas as pd 
from pathlib import Path
import numpy as np

filename = "transcribed_base"
# remove rows that have NaN as model_transcription for model evaluation
files = ['transcribed_base', 'transcribed_robust']
dirty_indices = set()
for f in files:
        df = pd.read_csv("output/" + f+".csv")
        dirty_indices.update(df.loc[pd.isna(df["model_transcription"]), :].index.values)

df = pd.read_csv("output/" + filename+".csv") 
clean_df = df.drop(dirty_indices)
clean_df.to_csv(Path("data/clean_newdata.csv"), index = False, header=True)
print(f"removed {len(dirty_indices)} model_transcriptions that contain NaN")

# load as dataset
ds = load_dataset('csv', data_files='data/clean_newdata.csv', split='train')
predictions = ds['model_transcription']
references = ds['transcription']

# word error rate
wer_metric = load("wer")
wer = wer_metric.compute(predictions=predictions, references=references)

# perplexity
perplexity = load("perplexity", module_type="metric")
input_texts = references
results = perplexity.compute(model_id='gpt2',
                             input_texts=input_texts)

# Character error rate
cer_metric = load("cer")
cer = cer_metric.compute(predictions=predictions, references=references)


metrics = {"filename" : filename+ '.csv',
           "perplexity" : results['mean_perplexity'],
           "wer" : wer,
           "cer" : cer
        }
print(metrics)

df = pd.read_csv("output/metrics.csv")
df.set_index('filename', inplace= True)
df.loc[filename + '.csv'] = metrics
df.reset_index(inplace=True)
df.to_csv(Path("output/metrics.csv"), index = False, header=True, float_format='%.5f')
