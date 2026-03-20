#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi

gpu=0
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=-1 # start from 0 if you need to start from data preparation
stop_stage=8

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="127.0.0.1:29500" # "localhost:0"
num_nodes=1
job_id=2023

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
ASCEND=downloads
dstore_name=ASCEND_dstore

# model setting
train_config=conf/train.yaml
checkpoint=exp/20220506_u2pp_conformer_exp_wenetspeech/final.pt
dict=exp/20220506_u2pp_conformer_exp_wenetspeech/units.txt
dir=exp/20220506_u2pp_conformer_exp_ascend
tensorboard_dir=tensorboard

# training resources
num_workers=8
prefetch=10

# decoding setting
decode_checkpoint=$dir/final.pt
average_checkpoint=true
average_num=5
average_mode=step
max_step=88888888

# knn setting
decode_modes="knn_ctc"
dstore_dir="datastore/d_all"
use_null_mask=True # build datastore with skip-blank strategy
decode_skip_blank=True # decoding with skip-blank strategy
dstore_size=1798000 # dstore_size=13000001
lmbda=0.45 # interpolate weight, adjust to the dataset
thr=0.0 # threshold of CTC pseudo label, default = 0
knn_temp=1.0 # temperature, defalut = 1  
k=1024 # k neighbours

# gated monolingual knn setting
dstore_dir_zh="datastore/d_cn"
dstore_dir_en="datastore/d_en"
dstore_size_zh=1000000
dstore_size_en=798000

# no use
scale_lmbda=False
scale_lmbda_temp=1

# training engine
train_engine=torch_ddp

deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Download data"
  mkdir -p ${ASCEND}
  python local/create_dataset.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "stage 0: preparing training data"
  for part in "train_zh" "train_en" "train" "validation" "test"; do
    local/data_prep.pl "${ASCEND}" ${part} data/"${part}"
  done

  # remove test&dev data from validated sentences
  for part in "train_zh" "train_en" "train"; do
    tools/filter_scp.pl --exclude data/validation/wav.scp data/${part}/wav.scp > data/${part}/temp_wav.scp
    tools/filter_scp.pl --exclude data/test/wav.scp data/${part}/temp_wav.scp > data/${part}/wav.scp
    tools/fix_data_dir.sh data/${part}
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "stage 1: extract cmvn features"
  python tools/compute_cmvn_stats.py --num_workers 8 --train_config $train_config \
      --in_scp data/train/wav.scp \
      --out_cmvn data/train/global_cmvn
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: prepare wenet data format"
  for part in "train_zh" "train_en" "train" "validation" "test"; do
    python tools/make_raw_list.py \
                data/${part}/wav.scp \
                data/${part}/text \
                data/${part}/data.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: neural network training"
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"

  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type $data_type \
      --train_data data/train/data.list \
      --cv_data data/validation/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: ctc decoding"
  decode_modes="ctc_greedy_search"

  # no use
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_ascend
    mkdir -p $test_dir
    python wenet/bin/recognize.py --gpu $gpu \
      --modes $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/ascend_text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

    sed -i "s|▁| |g" $test_dir/ascend_text
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/ascend_text > $test_dir/ascend_wer
    tail $test_dir/ascend_wer
  }
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: build bilingual datastore (D_ALL)"
  decode_modes="knn_ctc"
  mkdir -p log

  # no use
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_${dstore_size}_${dstore_name}_on_ascend
    mkdir -p $test_dir
    mkdir -p $dstore_dir
    python wenet_knn_ctc.py --gpu $gpu \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/train/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --build_index \
      --dstore_dir $dstore_dir\
      --dstore_size $dstore_size \
      --lmbda $lmbda \
      --thr $thr \
      --use_null_mask $use_null_mask
  }
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: knn_ctc decoding"
  decode_modes="knn_ctc"

  # no use
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0

  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_${dstore_size}_${dstore_name}_on_ascend
    mkdir -p $test_dir
    python wenet_knn_ctc.py  --gpu $gpu \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/ascend_text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --knn \
      --dstore_dir $dstore_dir\
      --dstore_size $dstore_size \
      --lmbda $lmbda \
      --knn_temp $knn_temp \
      --k $k \
      --decode_skip_blank $decode_skip_blank\
      --scale_lmbda $scale_lmbda\
      --scale_lmbda_temp $scale_lmbda_temp
    sed -i "s|▁| |g" $test_dir/ascend_text
    python tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/ascend_text > $test_dir/ascend_wer
    tail $test_dir/ascend_wer
  }
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: build monolingual datastores (D_CN, D_EN)"
  decode_modes="knn_ctc"

  # no use
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  
  # build chinese datastore (D_CN)
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/test_${mode}_${dstore_size_zh}_${dstore_name}_on_ascend
    mkdir -p $test_dir
    mkdir -p $dstore_dir
    python wenet_knn_ctc.py --gpu $gpu \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/train/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --build_index \
      --dstore_dir $dstore_dir\
      --dstore_size $dstore_size \
      --lmbda $lmbda \
      --thr $thr \
      --use_null_mask $use_null_mask
  }
  done
  python wenet_knn_ctc.py --gpu $gpu --mode "knn_ctc" \
    --config $dir/train.yaml --data_type $data_type \
    --test_data data/train_zh/data.list \
    --checkpoint $decode_checkpoint \
    --build_index --dstore_dir "datastore/ascend_zh" \
    --dstore_size $dstore_size_zh ...

  # build english datastore (D_EN)
  python wenet_knn_ctc.py --gpu $gpu --mode "knn_ctc" \
    --config $dir/train.yaml --data_type $data_type \
    --test_data data/train_en/data.list \
    --checkpoint $decode_checkpoint \
    --build_index --dstore_dir "datastore/ascend_en" \
    --dstore_size $dstore_size_en ...
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: gated monolingual knn_ctc decoding"
  decode_modes="knn_ctc"
  
  # Ensure your script points to BOTH directories
  python wenet_knn_ctc.py --gpu $gpu --mode "knn_ctc" \
    --config $dir/train.yaml --data_type $data_type \
    --test_data data/test/data.list \
    --checkpoint $decode_checkpoint \
    --knn --gated_mode True \
    --dstore_dir_zh "datastore/ascend_zh" \
    --dstore_dir_en "datastore/ascend_en" \
    --result_file $dir/test_ours_gated/text ...
fi

# if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
#   echo "stage 9: Export the trained model"
#   python wenet/bin/export_jit.py \
#     --config $dir/train.yaml \
#     --checkpoint $dir/avg_${average_num}.pt \
#     --output_file $dir/final.zip
# fi
