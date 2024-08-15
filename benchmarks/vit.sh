export TF_USE_LEGACY_KERAS=1
#export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
echo "numactl -C0-52 -m 0 python vit.py $*"
numactl -C0-52 -m 0 python vit.py $*
