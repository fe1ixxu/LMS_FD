# bash sweep_multi.sh moe/dense en_to_many/many_to_en
partition=learnaccel
num_trials=1
num_nodes=$2
num_gpus_per_node=8
type="en_to_many"
# backend can be "no_c10d" or "fully_sharded"
backend="fully_sharded"

script_name="sweep_benchmark_mmt.py"

langs="en,cs,de,es,et,fi,fr,gu,hi,kk,lt,lv,ro,ru,tr,zh"
if [ "$type" == "en_to_many" ] ; then
    lang_pairs="en-cs,en-de,en-es,en-et,en-fi,en-fr,en-gu,en-hi,en-kk,en-lt,en-lv,en-ro,en-ru,en-tr,en-zh"
    eval_cap=1.0
else
    lang_pairs="cs-en,de-en,es-en,et-en,fi-en,fr-en,gu-en,hi-en,kk-en,lt-en,lv-en,ro-en,ru-en,tr-en,zh-en"
    eval_cap=1.0
fi

data_dir="/private/home/shru/wmt_datasets/raw_v2/${type}_bin"


if [ "$1" == "moe" ] ; then
    expert_count=$(( num_nodes * num_gpus_per_node ))
    prefix="${expert_count}experts"
    checkpoint_dir="/checkpoint/$USER/wmt30/${type}_moe_$prefix/"
    moe_param=" --moe --encoder-moe-freq 2 --decoder-moe-freq 2 --moe-eval-cap ${eval_cap} "
else
    prefix="dense"
    checkpoint_dir="/checkpoint/$USER/wmt30/${type}/"
    moe_param=""
fi

mkdir -p /checkpoint/$USER/wmt30/tb/

python scripts/wmt30/${script_name} -d ${data_dir} -p "$prefix" \
    --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
    -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
    --time 3999 \
    --sampling-method temperature --sampling-temperature 1.5 \
    --decoder-langtok --encoder-langtok src \
    --langs $langs \
    --lang-pairs ${lang_pairs} \
    ${moe_param} \
    --ddp-backend $backend \
    --max-update 5000 \
    --update-freq 1 \
    --max-tokens 4096 \
    --lr 0.004 \
    --tensorboard-logdir /checkpoint/$USER/wmt30/tb/ \
    --log-interval 50
