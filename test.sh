data_path=pdbbind2020

################################################################################################

prompt_nf=8
best_epoch=12
ckpt_path=results/prompt_$prompt_nf/models/epoch_$best_epoch/model.safetensors

python promptbind/test_promptbind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_prompt_$prompt_nf \
    --ckpt $ckpt_path \
    --local-eval \
    --n-iter 12 \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf

################################################################################################

prompt_nf=16
best_epoch=12
ckpt_path=results/prompt_$prompt_nf/models/epoch_$best_epoch/model.safetensors

python promptbind/test_promptbind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_prompt_$prompt_nf \
    --ckpt $ckpt_path \
    --local-eval \
    --n-iter 12 \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf

################################################################################################

prompt_nf=32
best_epoch=22
ckpt_path=results/prompt_$prompt_nf/models/epoch_$best_epoch/model.safetensors

python promptbind/test_promptbind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_prompt_$prompt_nf \
    --ckpt $ckpt_path \
    --local-eval \
    --n-iter 12 \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf 

################################################################################################

prompt_nf=48
best_epoch=40
ckpt_path=results/prompt_$prompt_nf/models/epoch_$best_epoch/model.safetensors

python promptbind/test_promptbind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_prompt_$prompt_nf \
    --ckpt $ckpt_path \
    --local-eval \
    --n-iter 12 \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf