# ATC

## ONLINE ASR Call
python asr_multiple_v2.1.py --code 5 --wavlist ASRpost_input.txt

## ATCOSIM Data
The ATCOSIM dataset can be obtained here:

https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html

Place the atcosim folder in the HG folder. 
The atcosim fodler should contain the folders: DOC, HTMLdata, TXTdata and WAVdata.

Funcions in the HuggingFace (HF) can be used to transcribe, train and evaluate data.

For models pretrained on the ATCOSIM dataset, HuggingFace can be used:

https://huggingface.co/KaranChand

