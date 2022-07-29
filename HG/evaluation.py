from datasets import load_dataset, load_metric
import pandas as pd 
from pathlib import Path

# remove rows that have NaN as model_transcription for model evaluation
files = ['transcribed_base', 'transcribed_robust', 'transcribed_hubert']
dirty_indices = set()
for f in files:
        df = pd.read_csv("output/" + f+".csv")
        dirty_indices.update(df.loc[pd.isna(df["model_transcription"]), :].index.values)

for filename in files:
        df = pd.read_csv("output/" + filename+".csv") 
        clean_df = df.drop(dirty_indices)
        clean_df.to_csv(Path("data/clean_newdata.csv"), index = False, header=True)
        print(f"removed {len(dirty_indices)} model_transcriptions that contain NaN")

        # load as dataset
        ds = load_dataset('csv', data_files='data/clean_newdata.csv', split='train')
        predictions = ds['model_transcription']
        references = ds['transcription']

        # word error rate
        wer_metric = load_metric("wer")
        wer = wer_metric.compute(predictions=predictions, references=references)

        # perplexity
        perplexity = load_metric("perplexity", module_type="metric")
        input_texts = references
        results = perplexity.compute(model_id='gpt2',
                                     input_texts=input_texts)

        # Character error rate
        cer_metric = load_metric("cer")
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
