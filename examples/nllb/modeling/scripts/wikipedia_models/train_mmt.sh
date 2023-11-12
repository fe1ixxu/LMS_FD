# Usage: bash examples/nllb/modeling/scripts/wikipedia_models/train_mmt.sh dense/moe en_to_many/many_to_en
partition=learnaccel
num_trials=1
num_nodes=8
num_gpus_per_node=8
max_time_mins=3999

# check if 100K works better or as well as 200K updates
max_update=100000

# bsz is ~600K tokens for 64 gpus
# increase max_tokens and decrease update_freq if there is a lot of free GPU memory for your arch
max_tokens=3072
update_freq=4

# For arch sweep, you can register new arch grids in sweep/sweep_mmt.py
# e.g. @register_grid("transformer_12_12")

validate_interval_updates=10000
seed=2
type=$2

script_module="examples.nllb.modeling.sweep.sweep_mmt"

if [ "$type" == "en_to_many" ] ; then
    lang_pairs_file="examples/nllb/modeling/scripts/wikipedia_models/en_xx_pairs.txt"
    eval_cap=1.0
else
    lang_pairs_file="examples/nllb/modeling/scripts/wikipedia_models/xx_en_pairs.txt"
    eval_cap=1.0
fi

data_prefix="/large_experiments/nllb/mmt/multilingual_bin/wiki11.en_x.128k.v2/data_bin"
# TODO: change #shards if there are more shards later
num_shards=4
data_dir=""
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}
echo $data_dir

chk_prefix="/large_experiments/nllb/mmt/h2_21_models/wikipedia_mmt"
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
# these default values should work well
for arch in transformer_12_12 transformer_12_12_small transformer_12_3; do
for lr in 0.002 ; do
for dropout in 0.1; do
    python -m ${script_module} -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --time "$max_time_mins" \
        --sampling-method concat \
        --decoder-langtok --encoder-langtok "tgt" \
        --langs "eng,cat,fra,hau,ibo,isl,lug,oci,swh,xho,zho_Hans,zul" \
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
        --save-interval-updates 10000 \
        --tensorboard-logdir ${chk_prefix}/tb/ \
        --arch "$arch" \
        --dropout ${dropout} \
        --validate-interval-updates "${validate_interval_updates}" \
        --seed "$seed" \
        --max-pos 1024 \
        --snapshot-code
done
done
done
