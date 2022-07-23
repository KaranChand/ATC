from huggingface_hub import notebook_login
from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets import Audio, load_dataset
import torch
from jiwer import wer
import numpy as np
import pandas as pd
