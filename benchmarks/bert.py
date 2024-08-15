
import os

from transformers import TFBertModel
import tensorflow as tf
import time
from utils import *

tf.config.optimizer.set_experimental_options({'remapping': False})

VOCAB_SIZE = 30522
SEQ_LEN = 384

args = get_args()
set_precision_and_threads(args)

def get_bert_model(model_type):
  if model_type == 'large':
    model = TFBertModel.from_pretrained('google-bert/bert-large-uncased')
  elif model_type == 'base':
    model = TFBertModel.from_pretrained('google-bert/bert-base-uncased')
  elif model_type == 'distil':
    model = TFBertModel.from_pretrained('distilbert/distilbert-base-uncased')
  else:
    model = TFBertModel.from_pretrained('google-bert/bert-large-uncased')
  return model

#run_llm_bm_new('google-bert/bert-large-uncased', 
#           TFBertModel.from_pretrained, 
#           args,[1, 16, 32, 48, 64, 128], SEQ_LEN,VOCAB_SIZE)
#model_args = ('llm', SEQ_LEN, VOCAB_SIZE)
#run_bm('large', get_bert_model,args,[1, 16], model_args)
model = get_bert_model('large')
run_llm(model,args,SEQ_LEN,VOCAB_SIZE)

