data_path=pdbbind2020

################################################################################################

prompt_nf=8
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='no')"
accelerate launch promptbind/train_promptbind.py \
    --batch_size 6 \
    -d 0 \
    -m 5 \
    --data-path $data_path \
    --label baseline \
    --addNoise 5 \
    --resultFolder ./results \
    --tqdm-interval 60 \
    --use-compound-com-cls \
    --distmap-pred mlp \
    --total-epochs 50 \
    --exp-name new_prompt_$prompt_nf \
    --coord-loss-weight 1.0 \
    --pair-distance-loss-weight 1.0 \
    --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 \
    --pocket-distance-loss-weight 0.05 \
    --lr 1e-06 --lr-scheduler poly_decay \
    --distmap-pred mlp \
    --hidden-size 512 --pocket-pred-hidden-size 128 \
    --n-iter 12 --mean-layers 4 \
    --random-n-iter \
    --refine refine_coord \
    --coordinate-scale 5 \
    --geometry-reg-step-size 0.001 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --noise-for-predicted-pocket 0 \
    --clip-grad \
    --pocket-idx-no-noise \
    --pocket-cls-loss-func bce \
    --use-esm2-feat \
    --pocket-pred-layers 1 \
    --pocket-pred-n-iter 1 \
    --center-dist-threshold 4 \
    --mixed-precision no \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf \
    --disable-validate \
    --disable-tqdm

################################################################################################

prompt_nf=16
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='no')"
accelerate launch promptbind/train_promptbind.py \
    --batch_size 6 \
    -d 0 \
    -m 5 \
    --data-path $data_path \
    --label baseline \
    --addNoise 5 \
    --resultFolder ./results \
    --tqdm-interval 60 \
    --use-compound-com-cls \
    --distmap-pred mlp \
    --total-epochs 50 \
    --exp-name new_prompt_$prompt_nf \
    --coord-loss-weight 1.0 \
    --pair-distance-loss-weight 1.0 \
    --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 \
    --pocket-distance-loss-weight 0.05 \
    --lr 1e-06 --lr-scheduler poly_decay \
    --distmap-pred mlp \
    --hidden-size 512 --pocket-pred-hidden-size 128 \
    --n-iter 12 --mean-layers 4 \
    --random-n-iter \
    --refine refine_coord \
    --coordinate-scale 5 \
    --geometry-reg-step-size 0.001 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --noise-for-predicted-pocket 0 \
    --clip-grad \
    --pocket-idx-no-noise \
    --pocket-cls-loss-func bce \
    --use-esm2-feat \
    --pocket-pred-layers 1 \
    --pocket-pred-n-iter 1 \
    --center-dist-threshold 4 \
    --mixed-precision no \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf \
    --disable-validate \
    --disable-tqdm

################################################################################################

prompt_nf=32
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='no')"
accelerate launch promptbind/train_promptbind.py \
    --batch_size 6 \
    -d 0 \
    -m 5 \
    --data-path $data_path \
    --label baseline \
    --addNoise 5 \
    --resultFolder ./results \
    --tqdm-interval 60 \
    --use-compound-com-cls \
    --distmap-pred mlp \
    --total-epochs 50 \
    --exp-name new_prompt_$prompt_nf \
    --coord-loss-weight 1.0 \
    --pair-distance-loss-weight 1.0 \
    --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 \
    --pocket-distance-loss-weight 0.05 \
    --lr 1e-06 --lr-scheduler poly_decay \
    --distmap-pred mlp \
    --hidden-size 512 --pocket-pred-hidden-size 128 \
    --n-iter 12 --mean-layers 4 \
    --random-n-iter \
    --refine refine_coord \
    --coordinate-scale 5 \
    --geometry-reg-step-size 0.001 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --noise-for-predicted-pocket 0 \
    --clip-grad \
    --pocket-idx-no-noise \
    --pocket-cls-loss-func bce \
    --use-esm2-feat \
    --pocket-pred-layers 1 \
    --pocket-pred-n-iter 1 \
    --center-dist-threshold 4 \
    --mixed-precision no \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf \
    --disable-validate \
    --disable-tqdm

################################################################################################

prompt_nf=48
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='no')"
accelerate launch promptbind/train_promptbind.py \
    --batch_size 6 \
    -d 0 \
    -m 5 \
    --data-path $data_path \
    --label baseline \
    --addNoise 5 \
    --resultFolder ./results \
    --tqdm-interval 60 \
    --use-compound-com-cls \
    --distmap-pred mlp \
    --total-epochs 50 \
    --exp-name new_prompt_$prompt_nf \
    --coord-loss-weight 1.0 \
    --pair-distance-loss-weight 1.0 \
    --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 \
    --pocket-distance-loss-weight 0.05 \
    --lr 1e-06 --lr-scheduler poly_decay \
    --distmap-pred mlp \
    --hidden-size 512 --pocket-pred-hidden-size 128 \
    --n-iter 12 --mean-layers 4 \
    --random-n-iter \
    --refine refine_coord \
    --coordinate-scale 5 \
    --geometry-reg-step-size 0.001 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --noise-for-predicted-pocket 0 \
    --clip-grad \
    --pocket-idx-no-noise \
    --pocket-cls-loss-func bce \
    --use-esm2-feat \
    --pocket-pred-layers 1 \
    --pocket-pred-n-iter 1 \
    --center-dist-threshold 4 \
    --mixed-precision no \
    --pocket-prompt-nf $prompt_nf \
    --complex-prompt-nf $prompt_nf \
    --disable-validate \
    --disable-tqdm