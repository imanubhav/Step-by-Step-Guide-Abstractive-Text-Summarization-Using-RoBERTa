#Installation
!pip install datasets==1.0.2
!pip install transformers
!rm seq2seq_trainer.py
!wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/seq2seq/seq2seq_trainer.py
!pip install rouge_score

import datasets
import transformers
import pandas as pd
from datasets import Dataset

#Tokenizer
from transformers import DistilBertTokenizer

#MEncode-Decoder Model
from transformers import EncoderDecoderModel

#Training
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional