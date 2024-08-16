
from transformers import GPT2Tokenizer, TFGPT2Model
import numpy as np
import tf_keras as keras
import time
from gen_random_text_data import gen_random_text


VOCAB_SIZE = 50257
STEPS=10
BS=32
SEQ_LEN=1024
def get_input_data(batch_size=1, seq_length=384):
  shape = (batch_size, seq_length)
  input_ids = np.random.randint(1, VOCAB_SIZE, size=shape).astype(np.int32)
  #input_ids = gen_random_text(seq_length, 4)
  #print("==============================================")
  token_type_ids = np.ones(shape).astype(np.int32)
  attention_mask  = np.ones(shape).astype(np.int32)
  return [input_ids, attention_mask, token_type_ids]

dataset =[]
for i in range(STEPS):
  inputs = get_input_data(BS, SEQ_LEN)
  dataset.append(inputs)

#inputs = get_input_data(BS, SEQ_LEN)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')
#text = "Replace me by any text you'd like."

for i in range(STEPS):
  encoded_input = dataset[i] #tokenizer(dataset[i], return_tensors='tf')
  start = time.time()
  output = model.predict(encoded_input, optimize_for_inference=True)
  end = time.time()
  print(" Time :", (end-start))
#print(output)
