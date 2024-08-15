export TF_USE_LEGACY_KERAS=1
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
echo "numactl -C0-52 -m 0 python bm.py $*"
numactl -C0-40 -m 0 python bm.py -batches 8 16 32 -steps 10 -intra_ops 40 -model_name bert $*
#numactl -C0-52 -m 0 python bm.py -batches 8 16 32 -steps 40 -intra_ops 52 $*
