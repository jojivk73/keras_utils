
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
BS = 32
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

# Elsewhere, we can reload the artifact and serve it.
# The endpoint we added is available as a method:
keras.mixed_precision.set_global_policy("mixed_bfloat16")
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(52)

serving_model = tf.saved_model.load("./saved_model")
print("Model Loaded...................!")

for j in range(1):
  for i in range(STEPS):
    inputs = dataset[i]
    start = time.time()
    #outputs = model.predict(inputs, optimize_for_inference=args.optimize)
    #outputs = serving_model.infer(inputs)
    #outputs = model.predict(inputs)
    outputs = serving_model.call(inputs)
    end = time.time()
    print(i,": Time taken:", (end-start))
print("=====Done==================")

