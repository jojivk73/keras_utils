
import time
import numpy as np
import tf_keras as keras
import argparse
import tensorflow as tf
from tf_keras.export import ExportArchive


def gen_image_input_data(batch_size=1, dim=224):
  shape = (batch_size, 3, dim, dim)
  #print(" DS Shape:", shape)
  input_ids = np.random.random_sample(size=shape).astype(np.float32)
  return input_ids

def gen_input_data(batch_size=1, seq_length=384, VOCAB_SIZE=30522):
  shape = (batch_size, seq_length)
  #print(" DS Shape:", shape)
  input_ids = np.random.randint(1, VOCAB_SIZE, size=shape).astype(np.int32)
  token_type_ids = np.ones(shape).astype(np.int32)
  attention_mask  = np.ones(shape).astype(np.int32)
  return [input_ids, attention_mask, token_type_ids]

def gen_dataset(steps, bs, seq_len, vocab_size, one_set=False):
    dataset =[]
    if one_set:
        dataset = gen_input_data(bs*steps, seq_len, vocab_size)
    for i in range(steps):
        inputs = gen_input_data(bs, seq_len, vocab_size)
        dataset.append(inputs)
    return dataset

def gen_image_dataset(steps, bs, dim=224):
    dataset =[]
    for i in range(steps):
        inputs = gen_image_input_data(bs, dim)
        dataset.append(inputs)
    return dataset

def set_precision_and_threads(args):
    tf.config.threading.set_inter_op_parallelism_threads(args.inter_ops)
    tf.config.threading.set_intra_op_parallelism_threads(args.intra_ops)
    if args.precision == 'bfloat16':
        print(" Precision: Bfloat16")
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
    elif args.precision == 'float16':
        print(" Precision: float16")
        keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        print(" Precision: float32")

def get_args():
    parser = argparse.ArgumentParser(description='Run Model.')
    parser.add_argument('-optimize', type=bool, default=False,
                    help='bool to enable optmize_for_inference')
    parser.add_argument('-precision', type=str, default='bfloat16',
                    help='pecision of the model')
    parser.add_argument('-inter_ops', type=int, default=2,
                    help='an integer for the accumulator')
    parser.add_argument('-intra_ops', type=int, default=52,
                    help='an integer for the accumulator')
    parser.add_argument('-bs', type=int, default=32,
                    help='an integer for the batch size')
    parser.add_argument('-steps', type=int, default=10,
                    help='Number of steps to run')
    parser.add_argument('-compare_steps', type=int, default=5,
                    help='Last steps to run')
    parser.add_argument('-model_size', type=str, default='',
                    help='size of the model Eg. large/medium/none for gpt2')
    parser.add_argument('-model_name', type=str, default='',
                    help='Eg. gpt2')
    parser.add_argument('-device', type=str, default='cpu',
                    help='Eg. xpu')
    parser.add_argument('-batches', nargs='+', type=int, default=None)
    parser.add_argument('-models', nargs='+', type=str, default=None)
    args = parser.parse_args()
    return args
 
def run_model_nsteps(model, dataset, steps, bs, show_time=False):
  step_time ={}
  for i in range(steps):
    inputs = dataset[i]
    start = time.time()
    #outputs = model.predict(inputs, batch_size=bs, verbose=1)
    outputs = model.call(inputs)
    end = time.time()
    step_time[i] = (end-start)
    if show_time:
      print(i,": Time taken:", (end-start))
  return step_time


def export_reload_model(model, optimize, input_sign):
    export_archive = ExportArchive()
    export_archive.track(model, optimize_for_inference=optimize)
    #export_archive.track(model)
    export_archive.add_endpoint(
        name="call",
        fn=model.call,
        input_signature=input_sign,
    )
    export_archive.write_out("./saved_model")
    serving_model = tf.saved_model.load("./saved_model")
    return serving_model

def compile_save_reload_model(model, optimize, input_sign):
  model.compile(jit_compile=True, run_eagerly=False)
  return export_reload_model(model, optimize, input_sign)

def run_model(model, dataset, args):
  print("Batch Size:",args.bs)
  print("=====Compile==================")
  run_model_nsteps(model, dataset, args.steps, args.bs, True)
  print("=====Done==================")


def compare_def_opt_time(stime_def, stime_opt, args, bs):
  start = args.steps-args.compare_steps
  totl = 0
  totl_per = 0
  for i in range(start, args.steps):
      def_time = stime_def[i]
      opt_time = stime_opt[i]
      improv_per = (def_time - opt_time)/def_time
      totl_per += improv_per
      improv = def_time/opt_time
      totl += improv
  avg_per = 100*totl_per/args.compare_steps
  avg = 100*totl/args.compare_steps
  print(f"  Batch Size:{bs} = {avg_per}%, : {avg}x")
  return (avg_per, avg)

def run_llm(model, args, seq_len, vocab_size, input_sign):
  dataset = gen_dataset(args.steps, args.bs, seq_len, vocab_size)
  model = compile_save_reload_model(model, args.optimize, input_sign)
  run_model(model,dataset, args)

def run_vision(model, args, dim=224):
  dataset = gen_image_dataset(args.steps,args.bs,dim, input_sign)
  model = compile_save_reload_model(model, args.optimize, input_sign)
  run_model(model,dataset,args)


def get_dataset(args, bs, model_args):
    model_type = model_args[0]
    if model_type =='llm':
        seq_len, vocab_size = model_args[1:]
        dataset = gen_dataset(args.steps, bs, seq_len, vocab_size)
    elif model_type == 'vision':
        dim = model_args[1]
        dataset = gen_image_dataset(args.steps,bs,dim)
    else :
        print("ERROR: Specify Model Type...!")
        dataset=[]
        exit(0)
    return dataset


def run_bm_single(model_size, load_function, args, bs_list, model_args, opt=False):
  def_tab = {}
  set_precision_and_threads(args)
  model, input_sign = load_function(model_size)
  #model.compile(jit_compile=True, run_eagerly=False)
  model = compile_save_reload_model(model, opt, input_sign)
  print("------------Default Model-----------------------")
  for bs in bs_list:
    dataset = get_dataset(args, bs, model_args)
    print(" Batch Size:",bs)
    stime_def = run_model_nsteps(model, dataset, args.steps, bs, True)
    def_tab[bs] = stime_def
  return def_tab

def run_bm(model_size, load_function, args, bs_list, model_args):
  def_tab = {}
  opt_tab = {}
  print("------------Default Model-----------------------")
  def_tab = run_bm_single(model_size, load_function, args, bs_list,
                              model_args)
  print("------------Optimized Model-----------------------")
  opt_tab = run_bm_single(model_size, load_function, args, bs_list,
                              model_args, True)
  ret_data={}
  for bs in bs_list:
      stime_def = def_tab[bs]
      stime_opt = opt_tab[bs]
      impv = compare_def_opt_time(stime_def, stime_opt, args, bs)
      ret_data[bs] = impv
  return ret_data

