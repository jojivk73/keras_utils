
import os

from transformers import TFBertModel
from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import AutoImageProcessor, TFViTForImageClassification
import tensorflow as tf
import time
from utils import *

tf.config.optimizer.set_experimental_options({'remapping': False})
args = get_args()
set_precision_and_threads(args)

model_args ={
    'bert' : ('llm', 384, 30522),
    'gpt2' : ('llm', 512, 50257),
    'vit' : ('vision'),
}

bert = {
        'large': 'google-bert/bert-large-uncased',
        'base': 'google-bert/bert-base-uncased',
        #'distil': 'distilbert/distilbert-base-uncased',
        'input_sign' : [
            [
                tf.TensorSpec(shape=(None, 384), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 384), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 384), dtype=tf.int32),
            ],
        ],
       }
gpt2 = {
        'large': 'openai-community/gpt2-large',
        'medium': 'openai-community/gpt2-medium',
        'def': 'gpt2',
        'input_sign' : [
            [
                tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 512), dtype=tf.int32),
            ],
         ],
       }
vit = {
        'large': 'google/vit-large-patch32-384',
        'medium': 'google/vit-base-patch16-384',
        'def': 'google/vit-base-patch16-224',
        'input_sign' : [
            [
                tf.TensorSpec(shape=(None, 3, 224, 224), dtype=tf.float32),
            ],
         ],
        }
Models = {
   'bert' : bert,
   'gpt2' : gpt2,
   'vit' : vit,
}

def show_model(msg, model_name):
    print("=============================================================================")
    print("\t", msg, ":", model_name)
    print("=============================================================================")


def get_bert_model(model_type):
  model_name = Models['bert'][model_type]
  show_model("Loading Model", model_name)
  model = TFBertModel.from_pretrained(model_name)
  input_sign = Models['bert']['input_sign']
  return model, input_sign

def get_gpt_model(model_type):
  model_name = Models['gpt2'][model_type]
  show_model("Loading Model", model_name)
  model = TFGPT2Model.from_pretrained(model_name)
  input_sign = Models['bert']['input_sign']
  return model, input_sign

def get_vit_model(model_type):
  model_name = Models['vit'][model_type]
  show_model("Loading Model", model_name)
  model = TFViTForImageClassification.from_pretrained(model_name)
  input_sign = Models['bert']['input_sign']
  return model, input_sign

model_fns ={
    'bert' : get_bert_model,
    'gpt2' : get_gpt_model,
    'vit' : get_vit_model,
}

bs_list = [16, 32, 48, 64, 128]
#bs_list = [16, 32]

if args.batches:
    bs_list= args.batches

if args.models:
    model_list= args.models
else:
    model_list= Models.keys()

print("Batch List:", bs_list)

all_data = {}
for model in model_list: #Models.keys():
    if args.model_name !='' and model != args.model_name:
        continue
    model_fn = model_fns[model]
    model_arg= model_args[model]
    model_data = {}
    for model_size in Models[model]:
        if model=='vit':
            model_arg= model_args[model]
            image_dim = int(Models[model][model_size].split("-")[-1])
            model_arg = (model_arg, image_dim)
        show_model("Running Model", Models[model][model_size])
        time.sleep(5)
        res= run_bm(model_size, model_fn, args, bs_list, model_arg)
        model_data[model_size] = res
    all_data[model] = model_data


print("========================All data==================================")
print(all_data)
print("========================All data==================================")

for model in all_data.keys():
    show_model("Model", model)
    for model_size in all_data[model].keys():
        print("Improvements For:", Models[model][model_size])
        for bs, (avg_per, avg) in all_data[model][model_size].items():
            print(f"  batch-size:{bs} = {avg_per}% : {avg}")

