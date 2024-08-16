
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
STEPS=10

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
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(52)
model = TFBertModel.from_pretrained('google-bert/bert-large-uncased')

import argparse

parser = argparse.ArgumentParser(description='Run bert.')
parser.add_argument('-optimize', type=bool, default=False,
                    help='an integer for the accumulator')
args = parser.parse_args()
print("=====Compile==================")
model.compile(jit_compile=True, run_eagerly=False)

export_archive = ExportArchive()
export_archive.track(model, optimize_for_inference=args.optimize)
#export_archive.track(model)
export_archive.add_endpoint(
    name="call",
    fn=model.call,
    input_signature=[
        [
            tf.TensorSpec(shape=(None, SEQ_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(None, SEQ_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(None, SEQ_LEN), dtype=tf.int32),
        ],
    ],
)

#if args.optimize:
#  model.optimize_for_inference()
export_archive.write_out("./saved_model")
keras.mixed_precision.set_global_policy("mixed_bfloat16")
serving_model = tf.saved_model.load("./saved_model")
print("====================Loaded Model===============================")
#sleep(5)

print("=====Predict Optimize:", args.optimize, "==================")
for j in range(2):
  for i in range(STEPS):
    inputs = dataset[i]
    start = time.time()
    #outputs = model.predict(inputs)
    #outputs = model.predict(inputs, optimize_for_inference=False)
    outputs = serving_model.call(inputs)
    end = time.time()
    print(i,": Time taken:", (end-start))
print("=====Done==================")

