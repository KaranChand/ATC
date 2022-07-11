import pandas as pd 
from pathlib import Path  
from datasets import load_dataset, Audio
from scipy.io.wavfile import read
import numpy as np


df = pd.read_csv("fulldata.csv") 
df = df.drop(columns = ["utterance_id", "recording_corrupt", "comment_transcriptionist", "speaker_id", "session_id", "length_sec","recording_startpos_sec"])
df = df.assign(path=lambda x: 'atcosim/WAVdata/' + x.directory + '/' + x.subdirectory + '/' + x.filename + '.wav')
df = df.drop(columns = ["directory", "subdirectory", "filename"])


df.to_csv(Path("newdata.csv"), index = False, header=True) 
atcosim = load_dataset('csv', data_files='newdata.csv', split='train')

atcosim = atcosim.rename_column("path", "audio")
atcosim.to_csv("newdata.csv", index = False, header=True)

# arrays = [ [] for i in range(len(atcosim))]
# for i,x in enumerate(atcosim['path']):
#     arrays[i].append(read(x)[1])

# atcosim = atcosim.add_column("array", column = arrays)

# atcosim = atcosim.add_column(name='array', column=np.empty(len(atcosim)))
# for i,x in enumerate(atcosim['path']):
#     atcosim[i]["array"] = np.array(read(x)[1], dtype=np.float32).flatten()
# df.insert(0, 'id', range(len(df)))
# df["array"] = np.nan
# for i,x in enumerate(df['recording_id']):
#     df.at[i,'array'] =  np.array(read(df.at(i,'path')))[1]
# df.to_csv(Path("newdata.csv"), index = False, header=True)
# atcosim = load_dataset('csv', data_files='newdata.csv', split='train')

# df = df.assign(array=lambda x: np.array(read(x['path']))[1], dtype=np.float32)


# for i,x in enumerate(atcosim['path']):
#     audio_dict = {  "path": x,
#                     "array":list(np.array(read(x)[1], dtype=np.float32)), 
#                     "sampling_rate": 32000}
#     atcosim[i]["audio"] = Audio(audio_dict)
# atcosim = atcosim.add_column(name="array", column=np.zeros((len(atcosim)), dtype=np.float32))
# for i,x in enumerate(atcosim['path']):
#     atcosim[i]["array"] = np.array(read(x)[1], dtype=np.float32)
    


# a = read("adios.wav")
# numpy.array(a[1],dtype=float)

# def add_audio(example):
#     example["audio"] = Audio(example["path"])
#     return example
# atcosim = atcosim.map(add_audio)

# for i,x in enumerate(atcosim['path']):
#     atcosim[i][] = atcosim[i][Audio()]
# print(atcosim.features)
