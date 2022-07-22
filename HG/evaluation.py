from evaluate import load
from datasets import load_dataset, load_metric
import pandas as pd 
from pathlib import Path  

# remove rows that have NaN as model_transcription for model evaluation
filename = "transcribed_base"
df = pd.read_csv(filename+".csv") 
clean_df = df.dropna(subset=['model_transcription'])
clean_df.to_csv(Path("to_eval.csv"), index = False, header=True)
print("removed %x model_transcriptions that contain NaN" % (len(df) - len(clean_df)))

# load as dataset
ds = load_dataset('csv', data_files='to_eval.csv', split='train')
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

df = pd.read_csv("metrics.csv")
df.set_index('filename', inplace= True)
df.loc[filename + '.csv'] = metrics
df.reset_index(inplace=True)
df.to_csv(Path("metrics.csv"), index = False, header=True)