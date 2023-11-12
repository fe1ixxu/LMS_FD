ws=8
python train.py \
    --distributed-world-size $ws --distributed-port 19086 \
    /private/home/shru/wmt_datasets/raw_v2/en_to_many_bin \
    --save-dir tmp_train \
    --fp16 --fp16-no-flatten-grads \
    --max-update 10 \
    --task translation_multi_simple_epoch \
    --lang-pairs en-es,en-hi,en-lv,en-lt \
    --encoder-langtok src \
    --sampling-method temperature --sampling-temperature 1.5 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --arch transformer_wmt_en_de --decoder-normalize-before --encoder-normalize-before \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.004 --stop-min-lr 1e-09 --clip-norm 0.0 \
    --criterion moe_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --seed 2 --log-format json --log-interval 1 \
    --validate-interval-updates 20000 \
    --keep-interval-updates 40 \
    --max-source-positions 512 --max-target-positions 512 \
    --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum --moe-second-expert-policy all --moe-gating-use-fp32 --moe-freq 2 --moe-expert-count $ws \
    --moe-eval-capacity-token-fraction 1.0 --max-tokens-valid 1024 --ddp-backend no_c10d \
    --decoder-langtok \
    --langs en,es,hi,lt,lv \
    --save-interval-updates 20000
