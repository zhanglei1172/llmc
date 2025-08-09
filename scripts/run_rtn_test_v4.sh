export python_path='/workspace/zhangl98@xiaopeng.com/miniconda3/envs/q/bin/python'
# export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_ENDPOINT="https://alpha.hf-mirror.com"
export HF_DATASETS_CACHE="/workspace/zhangl98@xiaopeng.com/hf_cache/"

set -xe


model_path=$2 #'/workspace/gaoy25@xiaopeng.com/v4_datas/v4_ckpts/v12-20250731-232339/checkpoint-17540/'
flag="$(basename "$(dirname "$model_path")"/)"

# prec=w8a8 # w8a8 w4a8 w4a16

####
precs=('w8a8' 'w4afp16')
for prec in ${precs[@]}
do
    echo ./log/log_v4-rtn_0807-${prec}_${flag}_.log
    sed -E -i "s#path: .* #path: ${model_path} #"  configs/quantization/rtn_${prec}_test.yml
    sed -E -i "s#save_path: .* #save_path: /code/rtn_${prec}_${flag} #"  configs/quantization/rtn_${prec}_test.yml

    PYTHONPATH='.' llmc=./llmc ${python_path} -m torch.distributed.run --nnode 1 --nproc_per_node 1 --rdzv_id $1 --rdzv_backend c10d --rdzv_endpoint localhost:1$1 ./llmc/__main__.py --config configs/quantization/rtn_${prec}_test.yml --task_id $1  > ./log/log_v4-rtn_0807-${prec}_${flag}_.log 2>&1

    sed -E -i "s#path: ${model_path} #path: input_path #"  configs/quantization/rtn_${prec}_test.yml
    sed -E -i "s#save_path: /code/rtn_${prec}_${flag} #save_path: out_path #"  configs/quantization/rtn_${prec}_test.yml
    echo ./log/log_v4-rtn_0807-${prec}_${flag}_.log

done

