
import os

from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf
import time
from utils import *

tf.config.optimizer.set_experimental_options({'remapping': False})

VOCAB_SIZE = 50257
SEQ_LEN=1024
STEPS=20

args = get_args()
set_precision_and_threads(args)
if args.model_type =='large':
  model = TFGPT2Model.from_pretrained('openai-community/gpt2-large')
  print("Running GPT2-Large")
elif args.model_type =='medium':
  model = TFGPT2Model.from_pretrained('openai-community/gpt2-medium')
  print("Running GPT2-Medium")
else :
  model = TFGPT2Model.from_pretrained('gpt2')
  print("Running GPT2")
run_llm(model,args,SEQ_LEN,VOCAB_SIZE)

