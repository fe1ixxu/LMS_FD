#data=/fsx/shru/data/wmt_no_cjk_1n.wmt_mined.joined.64k/sharded_bin/shard000
data=/private/home/chau/wmt21/multilingual_bin/wmt_no_cjk_1n.wmt_mined.joined.64k/sharded_bin/shard000
ws=1
# optionally add MoE params to the model training
moe_params=" --moe-expert-count $ws --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum --moe-second-expert-policy all --moe-gating-use-fp32 --moe-freq 2 "
CUDA_VISIBLE_DEVICES=0 python train.py \
--distributed-world-size $ws --distributed-port 13678 \
$data \
--fp16 --fp16-no-flatten-grads \
--required-batch-size-multiple 1 \
--max-source-positions 1024 --max-target-positions 1024 \
--no-save --no-epoch-checkpoints \
--moe-expert-count $ws --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum --moe-second-expert-policy all --moe-gating-use-fp32 --moe-freq 2 \
--validate-interval-updates 10 \
--max-update 50 --update-freq 1 \
--task translation_multi_simple_epoch \
--lang-pairs en-de,en-pl \
--encoder-langtok src \
--decoder-langtok --langs en,de,pl \
--sampling-method temperature --sampling-temperature 5 \
--share-all-embeddings \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--lr 0.001 --stop-min-lr 1e-09 \
--clip-norm 0.0 \
--dropout 0.1 --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.0 \
--criterion moe_cross_entropy \
--seed 2 \
--log-interval 1 \
--ddp-backend no_c10d \
 --max-tokens 256 \
--arch transformer \
--encoder-layers 2 --decoder-layers 2 \
--encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--encoder-attention-heads 16 --decoder-attention-heads 16 \
--encoder-normalize-before --decoder-normalize-before
