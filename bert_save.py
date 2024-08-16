
import os
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

#import tf_keras as keras

from transformers import TFBertModel
import numpy as np
import tensorflow as tf
import time
import tf_keras as keras
from tf_keras.export import ExportArchive


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

keras.mixed_precision.set_global_policy("mixed_bfloat16")
model = TFBertModel.from_pretrained('google-bert/bert-large-uncased')

import argparse

parser = argparse.ArgumentParser(description='Run bert.')
parser.add_argument('-optimize', type=bool, default=False,
                    help='an integer for the accumulator')
args = parser.parse_args()
print("=====Compile==================")
model.compile(jit_compile=True, run_eagerly=False)

tf.saved_model.save(model, "./saved_model")
serving_model = tf.saved_model.load("./saved_model")

def setattr_as_constant(mod, attr_name):
    attr_obj = getattr(mod, attr_name)
    if isinstance(attr_obj, tf.Variable):
        print("Setting var :", attr_name)
        setattr(
           mod, attr_name, attr_obj.numpy().astype('bfloat16')
        )

for var in serving_model.variables:
    print(var.name)
    if 'kernel' in var.name:
        print(" Setting :", var.name)
        setattr(
            serving_model, var.name, var.numpy().astype('bfloat16')
        )

#serving_model.optimize_for_inference()
#import pdb
#pdb.set_trace()
keras.mixed_precision.set_global_policy("mixed_bfloat16")
print("====================Loaded Model===============================")

print("=====Predict Optimize:", args.optimize, "==================")
for j in range(1):
  for i in range(STEPS):
    inputs = dataset[i]
    start = time.time()
    #outputs = model.predict(inputs, optimize_for_inference=args.optimize)
    outputs = serving_model(inputs)
    end = time.time()
    print(i,": Time taken:", (end-start))
print("=====Done==================")

