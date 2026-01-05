# CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_DEV_EN_V11 --model

# CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_DEV_EN_V11 --model Phi-3-Vision

# CUDA_VISIBLE_DEVICES=0,1 python run.py --data NewtPhys_MultiImage --model Ovis1.6-Gemma2-27B --verbose

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=1 run.py --data NewtPhys_MultiImage --model Llama-3.2-11B-Vision-Instruct --verbose

# torchrun --nproc-per-node=1 run.py --data Video-MME --model InternVL2_5-4B --verbose

# uv pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git
