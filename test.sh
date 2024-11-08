#!/bin/bash
######################################################
data_path=pdbbind2020
cuda_device=1
######################################################


prompt_nfs=(8 16 32 48)
base_config="options/test_args.yml"
sed -i "s|data_path: .*|data_path: \"$data_path\"|" $base_config

for prompt_nf in "${prompt_nfs[@]}"
do
  sed -i "s/prompt_nf: [0-9]*/prompt_nf: $prompt_nf/" $base_config
  CUDA_VISIBLE_DEVICES=$cuda_device python promptbind/test_promptbind.py
done