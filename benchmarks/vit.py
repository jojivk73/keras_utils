
import os

from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import AutoImageProcessor, TFViTForImageClassification
import tensorflow as tf
import time
from utils import *

tf.config.optimizer.set_experimental_options({'remapping': False})

image_dim =384

args=get_args()
set_precision_and_threads(args)
#if args.model_type =='large':
model = TFViTForImageClassification.from_pretrained("google/vit-large-patch32-384")
#else:
#model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-384")
run_vision(model,args,image_dim)

