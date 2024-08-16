export TF_USE_LEGACY_KERAS=1
#export TF_CPP_MAX_VLOG_LEVEL=1
export DNNL_VERBOSE=2
#export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
echo "numactl -C0-52 -m 0 python bert.py $*"
numactl -C0-52 -m 0 python bert.py $*
