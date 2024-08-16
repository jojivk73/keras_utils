
import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

#import tf_keras as keras

from transformers import TFBertModel
import numpy as np
import tensorflow as tf
import time
import tf_keras as keras

tf.config.optimizer.set_experimental_options({'remapping': False})

VOCAB_SIZE = 30522
BS = 16
SEQ_LEN = 384
STEPS=1

def get_input_data(batch_size=1, seq_length=384):
  shape = (batch_size, seq_length)
  input_ids = np.random.randint(1, VOCAB_SIZE, size=shape).astype(np.int32)
  token_type_ids = np.ones(shape).astype(np.int32)
  attention_mask  = np.ones(shape).astype(np.int32)
  return [input_ids, attention_mask, token_type_ids]

dataset =[]
for i in range(STEPS):
  inputs = get_input_data(BS, SEQ_LEN)
  dataset.append(inputs)

#inputs = get_input_data(BS, SEQ_LEN)
keras.mixed_precision.set_global_policy("mixed_bfloat16")
model = TFBertModel.from_pretrained('google-bert/bert-large-uncased')

for submodule in model.submodules:
  if hasattr(submodule, 'kernel'):
    #print(submodule)
    submodule.kernel = submodule.kernel.numpy().astype('bfloat16')

import argparse

parser = argparse.ArgumentParser(description='Run bert.')
parser.add_argument('-optimize', type=bool, default=False,
                    help='an integer for the accumulator')
args = parser.parse_args()

print("=====Compile==================")
model.compile(jit_compile=True, run_eagerly=False)
print("=====Predict Optimize:", args.optimize, "==================")
for j in range(1):
  for i in range(STEPS):
    inputs = dataset[i]
    start = time.time()
    outputs = model.predict(inputs, optimize_for_inference=args.optimize)
    end = time.time()
    print(i,": Time taken:", (end-start))
print("=====Done==================")

