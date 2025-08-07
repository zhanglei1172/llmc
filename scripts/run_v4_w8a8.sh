export python_path='/workspace/zhangl98@xiaopeng.com/miniconda3/envs/q/bin/python'
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
export HF_ENDPOINT="https://alpha.hf-mirror.com"
export HF_DATASETS_CACHE="/workspace/zhangl98@xiaopeng.com/hf_cache/"

set -xe

PYTHONPATH='.' llmc=./llmc ${python_path} -m torch.distributed.run --nnode 1 --nproc_per_node 2 --rdzv_id 1175 --rdzv_backend c10d --rdzv_endpoint localhost:12358 ./llmc/__main__.py --config configs/quantization/v4_w8a8/v4_test_step_1_spinquant_fsdp.yml --task_id 1175 > ./log/log_v4-0807-w8a8_spinquant.log 2>&1

# huggingface-cli login first
# PYTHONPATH='.' llmc=./llmc ${python_path} -m torch.distributed.run --nnode 1 --nproc_per_node 1 --rdzv_id 1175 --rdzv_backend c10d --rdzv_endpoint localhost:12358 ./llmc/__main__.py --config configs/quantization/v4_w8a8/v4_test_step_2_gptq.yml --task_id 1175  > ./log/log_v4-0807-w8a8_sp_gptq.log 2>&1


# PYTHONPATH='.' llmc=./llmc ${python_path} -m torch.distributed.run --nnode 1 --nproc_per_node 1 --rdzv_id 1175 --rdzv_backend c10d --rdzv_endpoint localhost:12358 ./llmc/__main__.py --config configs/quantization/v4_w8a8/v4_test_step_1.5_smoothquantcustom.yml --task_id 1175  > ./log/log_v4-0807-w8a8_sp_sm.log 2>&1


