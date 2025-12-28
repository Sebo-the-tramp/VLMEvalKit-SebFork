# CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_DEV_EN_V11 --model

# CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_DEV_EN_V11 --model Phi-3-Vision

CUDA_VISIBLE_DEVICES=0 python run.py --data NewtPhys_MultiImage --model InternVL2_5-4B --verbose

# torchrun --nproc-per-node=2 run.py --data MMBench_DEV_EN_V11 --model deepseek_vl2_small --verbose

# torchrun --nproc-per-node=1 run.py --data Video-MME --model InternVL2_5-4B --verbose
