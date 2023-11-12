# Usage: bash examples/nllb/modeling/scripts/dense_99_languages/sweep_99_langs.sh moe/dense en_to_many/many_to_en
partition=nllb
num_trials=1
num_nodes=16
num_gpus_per_node=8
max_update=100000
max_tokens=3072
update_freq=3
max_time_mins=10999
arch=transformer_12_12
validate_interval_updates=10000
seed=2
type=$2

script_module="examples.nllb.modeling.sweep.sweep_mmt"

langs_file="$fairseq/examples/nllb/modeling/scripts/dense_99_languages/langs.txt"
if [ "$type" == "en_to_many" ] ; then
    lang_pairs_file="$fairseq/examples/nllb/modeling/scripts/dense_99_languages/lang_pairs_rev.txt"
    eval_cap=1.0
else
    lang_pairs_file="$fairseq/examples/nllb/modeling/scripts/dense_99_languages/lang_pairs.txt"
    eval_cap=1.0
fi

data_prefix="/large_experiments/nllb/mmt/multilingual_bin/flores99.x_en.256k/data_bin"
num_shards=16
data_dir=""
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}
echo $data_dir

chk_prefix="/large_experiments/nllb/mmt/h2_21_models/flores99_minedonly"
if [ "$1" == "moe" ] ; then
    expert_count=$(( num_nodes * num_gpus_per_node ))
    prefix="${expert_count}experts_demo"
    checkpoint_dir="${chk_prefix}/${type}_moe_$prefix/"
    moe_param=" --moe --moe-freq 2 "
else
    prefix="dense"
    checkpoint_dir="${chk_prefix}/${type}/"
    moe_param=""
fi
for encoder_langtok in "tgt" ; do
for lr in 0.004 ; do
for temp in 1.5 ; do
    python -m ${script_module} -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --time "$max_time_mins" \
        --sampling-method temperature --sampling-temperature $temp \
        --decoder-langtok --encoder-langtok ${encoder_langtok} \
        --langs $langs_file \
        --lang-pairs ${lang_pairs_file} \
        ${moe_param} \
        --moe-eval-cap ${eval_cap} \
        --ddp-backend no_c10d \
        --max-update $max_update \
        --update-freq 1 \
        --max-tokens $max_tokens \
        --update-freq $update_freq \
        --lr $lr \
        --opt adam16bit \
        --share-all-embeddings \
        --save-interval-updates 20000 \
        --tensorboard-logdir ${chk_prefix}/tb/ \
        --arch "$arch" \
        --dropout 0.0 \
        --validate-interval-updates "${validate_interval_updates}" \
        --seed "$seed" \
        --snapshot-code
    done
done
done
