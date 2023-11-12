# bash sweep_multi.sh moe/dense en_to_many/many_to_en
partition=${partition:-learnaccel}
num_trials=${num_trials:-12}
num_nodes=${num_nodes:-4}
num_gpus_per_node=8
max_update=${max_update:-100000}
max_tokens=${max_tokens:-8192}
max_time_mins=${max_time_mins:-3999}
arch=${arch:-transformer_wmt_en_de}
validate_interval_updates=${validate_interval_updates:-20000}
seed=${seed:-2}
type=$2

script_name="sweep_wmt30_multi.py"

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
    moe_param=" --moe --moe-freq 2 "
else
    prefix="dense"
    checkpoint_dir="/checkpoint/$USER/wmt30/${type}/"
    moe_param=""
fi

for lr in 0.004 ; do
for temp in 1.5 ; do
    python examples/nllb/modeling/wmt15_benchmark/${script_name} -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --time "$max_time_mins" \
        --sampling-method temperature --sampling-temperature $temp \
        --decoder-langtok --encoder-langtok src \
        --langs $langs \
        --lang-pairs ${lang_pairs} \
        ${moe_param} \
        --moe-eval-cap ${eval_cap} \
        --ddp-backend no_c10d \
        --max-update "$max_update" \
        --update-freq 1 \
        --max-tokens "$max_tokens" \
        --lr $lr \
        --save-interval-updates 20000 \
        --virtual-epoch-size 100000000 \
        --tensorboard-logdir /checkpoint/$USER/wmt30/tb/ \
        --arch "$arch" \
        --dropout 0.0 \
        --validate-interval-updates "${validate_interval_updates}" \
        --seed "$seed" \
        --opt adam
done
done
