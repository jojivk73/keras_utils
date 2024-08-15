

LENGTH = 4
NO_CODES = 1024

import numpy as np

def gen_random_text(text_length, granuality):
    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'+ ' '*12)
    np_alphabet = np.array(alphabet)
    np_codes = np.random.choice(np_alphabet, [text_length, granuality])
    codes = ["".join(np_codes[i]) for i in range(len(np_codes))]
    text = "".join(codes)
    return text

text = gen_random_text(LENGTH, NO_CODES)
print(text)
