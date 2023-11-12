declare -A test_set
test_set=( ["de-en"]="wmt14" ["ro-en"]="wmt16" ["cs-en"]="wmt18" \
["fr-en"]="wmt14" ["ru-en"]="wmt19" ["zh-en"]="wmt19" ["es-en"]="wmt13"
["fi-en"]="wmt19" ["et-en"]="wmt18" ["lv-en"]="wmt17" ["lt-en"]="wmt19"
["hi-en"]="wmt14" ["kk-en"]="wmt19" ["tr-en"]="wmt18" ["gu-en"]="wmt19" )

declare -A valid_set
valid_set=( ["cs-en"]="wmt17" ["fr-en"]="wmt13" ["ru-en"]="wmt18" \
["zh-en"]="wmt18" ["es-en"]="wmt12" ["fi-en"]="wmt18" ["de-en"]="wmt13" \
["et-en"]="wmt18/dev" ["lv-en"]="wmt17/dev" ["lt-en"]="wmt19/dev" ["ro-en"]="wmt16/dev" \
["hi-en"]="wmt14" ["kk-en"]="wmt19/dev" ["tr-en"]="wmt17" ["gu-en"]="wmt19/dev" )

MODEL_TYPE=${1:-"dense"}
DIRECTION=${2:-"en_to_many"}


langs="en,cs,de,es,et,fi,fr,gu,hi,kk,lt,lv,ro,ru,tr,zh"
SPM="/private/home/shru/wmt_datasets/raw_v2/spm_64000_500M.model"

if [ $DIRECTION == "en_to_many" ] ; then
    lang_pairs="en-cs,en-de,en-es,en-et,en-fi,en-fr,en-gu,en-hi,en-kk,en-lt,en-lv,en-ro,en-ru,en-tr,en-zh"
    SRC="en"
else
    lang_pairs="cs-en,de-en,es-en,et-en,fi-en,fr-en,gu-en,hi-en,kk-en,lt-en,lv-en,ro-en,ru-en,tr-en,zh-en"
    TGT="en"
fi

if [ $MODEL_TYPE == "moe" ] ; then
    constraint="-C volta32gb"
else
    constraint=""
fi

partition=No_Language_Left_Behind

# =========================
# # 32 experts moe many-to-en
# MODEL_FOLDER="/large_experiments/moe/shru/wmt30/many_to_en_moe_32experts/32experts.fp16.mu300000.uf1.entsrc.SPL_temperature.tmp1.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr${lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok8192.seed2.max_pos512.moe_w0.01.all.no_c10d.det.ves100000000.ngpu32"
# # 16 experts moe many-to-en
# MODEL_FOLDER="/large_experiments/moe/shru/wmt30/many_to_en_moe_16experts/16experts.fp16.mu300000.uf2.entsrc.SPL_temperature.tmp1.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr${lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok8192.seed2.max_pos512.moe_w0.01.all.no_c10d.det.ves100000000.ngpu16"
# dense many-to-en
MODEL_FOLDER="/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.mu300000.uf1.entsrc.SPL_temperature.tmp1.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr${lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok8192.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"

# # dense en-to-many
# MODEL_FOLDER="/large_experiments/moe/shru/wmt30/en_to_many/dense.fp16.mu300000.uf1.entsrc.SPL_temperature.tmp1.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr${lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok8192.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"

# # 32 experts en-to-many
# MODEL_FOLDER="/large_experiments/moe/shru/wmt30/en_to_many_moe_32experts/32experts.fp16.mu300000.uf1.entsrc.SPL_temperature.tmp1.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr${lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok8192.seed2.max_pos512.moe_w0.01.all.no_c10d.det.ves100000000.ngpu32"

# # 16 experts en-to-many
# MODEL_FOLDER="/large_experiments/moe/shru/wmt30/en_to_many_moe_16experts/16experts.fp16.mu300000.uf2.entsrc.SPL_temperature.tmp1.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr${lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok8192.seed2.max_pos512.moe_w0.01.all.no_c10d.det.ves100000000.ngpu16"

CHECKPOINTS=" checkpoint_10_120000 "

OUT_MODEL_FOLDER=$MODEL_FOLDER

for CHECKPOINT in  $CHECKPOINTS  ; do
    echo $CHECKPOINT
    MODEL=$MODEL_FOLDER/${CHECKPOINT}.pt
    # MODEL=`find $MODEL_FOLDER -type f  -iname "checkpoint*${CHECKPOINT}.pt"`
    echo "model=$MODEL"
    for gen_split in valid  ; do # test ; do
        echo $gen_split
        for lang in cs de es et 'fi' fr  gu  hi kk lt  lv  ro  ru  tr  zh ; do
            if [ $DIRECTION == "en_to_many" ] ; then
                TGT=$lang
            else
                SRC=$lang
            fi
            key_lang_pair="$lang-en"
            if [ "${gen_split}" == "valid" ] ; then
                testset="${valid_set[${key_lang_pair}]}"
            else
                testset="${test_set[${key_lang_pair}]}"
            fi
            echo "===========STARTING $lang==========="
            DATA="/private/home/shru/wmt_datasets/raw_v2/binarized/$SRC-$TGT"
            if [ ${MODEL_TYPE} == "moe" ] ; then
                if [ $DIRECTION == "en_to_many" ] ; then
                    BSZ=32
                    gpus=2
                    # TODO: change
                    nodes=1
                    cap=1.0
                else
                    BSZ=32
                    gpus=2
                    # TODO: change
                    nodes=1
                    cap=0.25
                fi
                cpus=$(( gpus * 10 ))
                mem="$(( gpus * 50 ))G"
                echo "mem=$mem"
                port=$(( ( RANDOM % 119 )  + 15000 ))
                WS=$(( nodes * gpus ))
                MOE_PARAMS="--is-moe \
                --distributed-world-size ${WS} --distributed-port ${port} \
                --model-overrides \"{'world_size': ${WS}, 'moe_eval_capacity_token_fraction': ${cap}, 'use_moe_pad_mask': False}\" "
            else
                BSZ=50
                gpus=1
                nodes=1
                cpus=8
                mem="48G"
                MOE_PARAMS=""
            fi
            OUTDIR=${OUT_MODEL_FOLDER}/gen_output${cap}/${SRC}-${TGT}_${CHECKPOINT}_${gen_split}
            mkdir -p $OUTDIR
                echo "python fairseq_cli/generate.py \
                    ${DATA} \
                    --path ${MODEL} \
                    --task translation_multi_simple_epoch \
                    --langs "${langs}" \
                    --lang-pairs \"${lang_pairs}\" \
                    --source-lang ${SRC} --target-lang ${TGT} \
                    --encoder-langtok \"src\" \
                    --decoder-langtok \
                    --gen-subset ${gen_split} \
                    --beam 5 \
                    --bpe 'sentencepiece' \
                    --sentencepiece-model ${SPM} \
                    --sacrebleu \
                    --fp16 \
                    ${MOE_PARAMS} \
                    --max-sentences $BSZ \
                    --results-path ${OUTDIR} | tee ${OUTDIR}/gen_best.out
                cat ${OUTDIR}/generate-${gen_split}.txt | grep -P \"^D-\" | sort -nr -k1.2 | cut -f3 > ${OUTDIR}/gen_best.output
                lang=\"$lang\"
                if [ \$lang == \"hi\" ] ; then
                    cat ${OUTDIR}/gen_best.output | sacrebleu -l ${SRC}-${TGT} /private/home/shru/wmt_datasets/raw_v2/${gen_split}.$SRC-$TGT.$TGT > ${OUTDIR}/bleu.results
                else
                    cat ${OUTDIR}/gen_best.output | sacrebleu -l ${SRC}-${TGT} -t $testset > ${OUTDIR}/bleu.results
                fi
                " > ${OUTDIR}/gen.sh
                echo "out in ${OUTDIR}/eval.out"
                # bash ${OUTDIR}/gen.sh &> ${OUTDIR}/eval.out
                # srun --output ${OUTDIR}/eval.out --error ${OUTDIR}/eval.err bash ${OUTDIR}/gen.sh
                sbatch \
                    --output ${OUTDIR}/eval.out \
                    --error ${OUTDIR}/eval.out \
                    --job-name ${SRC}-${TGT}.${CHECKPOINT}.${gen_split}.eval \
                    --gpus-per-node $gpus --nodes $nodes --cpus-per-task $cpus \
                    --time 1000 --mem $mem \
                    ${constraint} \
                    --partition $partition \
                    --ntasks-per-node 1 \
                    --open-mode append --no-requeue \
                    --wrap "srun bash ${OUTDIR}/gen.sh"
        done
    done
done
