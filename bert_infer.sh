export TF_USE_LEGACY_KERAS=1
#export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
#export TF_CPP_MAX_VLOG_LEVEL=1
#export DNNL_VERBOSE=2
#export TF_DUMP_GRAPH_PREFIX=/localdisk/jojimon/graph_dump
#export TF_CPP_VMODULE=xla_ops=5
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
echo "numactl -C0-52 -m 0 python infer_fix.py $*"
numactl -C0-51 -m 0 python bert_infer.py $*
