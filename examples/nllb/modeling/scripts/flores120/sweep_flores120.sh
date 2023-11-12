# bash sweep_multi.sh moe/dense en_to_many/many_to_en
partition=hpc2
num_trials=1
num_nodes=16
num_gpus_per_node=8
mem="0"
max_update=100000
max_tokens=4096
update_freq=2
max_time_mins=14999
arch=transformer_24_24
validate_interval_updates=20000
seed=2
type=$2

script_module="examples.nllb.modeling.sweep.sweep_mmt"

fairseq="/shared/home/shru/projects/fairseq-py"
cd $fairseq
langs_file="$fairseq/examples/nllb/modeling/scripts/flores120/langs.txt"
if [ "$type" == "en_to_many" ] ; then
    lang_pairs_file="$fairseq/examples/nllb/modeling/scripts/flores120/lang_pairs_rev.txt"
    eval_cap=1.0
else
    lang_pairs_file="$fairseq/examples/nllb/modeling/scripts/flores120/lang_pairs_both.txt"
    eval_cap=1.0
fi
#lang_pairs_file="eng-guj,eng-fra,eng-spa,eng-afr"
data_prefix="/data/nllb/flores125.en_xx_en.v3.1/data_bin"
num_shards=32
data_dir=""
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}
echo $data_dir

chk_prefix="/data/users/shru/models/flores120_v3.1"

if [ "$1" == "moe" ] ; then
    expert_count=$(( num_nodes * num_gpus_per_node ))
    prefix="${expert_count}ex"
    checkpoint_dir="${chk_prefix}/${type}_moe_fix/"
    moe_param=" --moe --moe-freq 2 "
else
    prefix="dense"
    checkpoint_dir="${chk_prefix}/${type}_fix/"
    moe_param=""
fi

for encoder_langtok in "tgt" ; do
for lr in 0.001 ; do # 0.0005 0.001 0.003 ; do
for warmup in 8000 ; do # 8000 16000 ; do
for temp in 1 ; do
for ddp_backend in "fully_sharded" ; do
    python -m ${script_module} -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} --partition ${partition} \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --time "$max_time_mins" \
        --mem ${mem} \
        --sampling-method temperature --sampling-temperature $temp \
        --decoder-langtok --encoder-langtok ${encoder_langtok} \
        --langs $langs_file \
        --lang-pairs ${lang_pairs_file} \
        ${moe_param} \
        --moe-eval-cap ${eval_cap} \
        --ddp-backend $ddp_backend \
        --max-update $max_update \
        --max-tokens $max_tokens \
        --update-freq $update_freq \
        --warmup-updates ${warmup} \
        --lr $lr \
        --opt adam16bit \
        --share-all-embeddings \
        --save-interval-updates 5000 \
        --tensorboard-logdir ${chk_prefix}/tb/ \
        --arch "$arch" \
        --dropout 0.0 \
        --validate-interval-updates "${validate_interval_updates}" \
        --seed "$seed" \
        --snapshot-code \
        --use-local-shard-size \
	    --checkpoint-activations \
        --zero2
done
done
done
done
done
