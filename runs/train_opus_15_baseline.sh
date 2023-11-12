#!/bin/bash
LANGS="eu,pt,bg,sk,zh,sl,de,hr,nb,ga,rw,as,fy,mr,se,en"
LANG_PAIRS="eu-en,pt-en,bg-en,sk-en,zh-en,sl-en,de-en,hr-en,nb-en,ga-en,rw-en,as-en,fy-en,mr-en,se-en,en-eu,en-pt,en-bg,en-sk,en-zh,en-sl,en-de,en-hr,en-nb,en-ga,en-rw,en-as,en-fy,en-mr,en-se"

FREQ=1
TEMPERATURE=5
DATA_DIR=./data/opus-15
DATA_BIN=${DATA_DIR}/data-bin/
SAVE_PATH=./checkpoint/OPUS_15_baseline
ARCH=transformer
MAX_TOKENS=4096

LAYER=6
DIM=512
FFN_DIM=2048
HEADS=8

HOMOBATCH="--one-dataset-per-batch"
CRITERION='label_smoothed_cross_entropy'


 python train.py  ${DATA_BIN} --arch ${ARCH}  --task translation_multi_simple_epoch \
 --lang-pairs ${LANG_PAIRS} --langs ${LANGS} --sampling-method temperature --sampling-temperature ${TEMPERATURE} --encoder-langtok tgt --enable-lang-ids \
 --encoder-layers ${LAYER} --decoder-layers ${LAYER} --encoder-ffn-embed-dim ${FFN_DIM} --decoder-ffn-embed-dim ${FFN_DIM} \
 --encoder-embed-dim ${DIM} --decoder-embed-dim ${DIM} --encoder-attention-heads ${HEADS} --decoder-attention-heads ${HEADS} --attention-dropout 0.1 --relu-dropout 0.0 \
 --decoder-normalize-before --encoder-normalize-before --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
 --max-update ${MAX_UPDATES} --update-freq ${FREQ}  --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0005 --stop-min-lr 1e-09 --clip-norm 0.0 --dropout 0.1 --weight-decay 0.0 --criterion ${CRITERION} \
 --label-smoothing 0.1  --max-tokens ${MAX_TOKENS}  --validate-interval-updates 500 --save-interval-updates 500 --save-interval 2 --no-epoch-checkpoints \
 --keep-interval-updates 1  --validate-interval 500  --seed 1234 --log-format simple --log-interval 100 --fp16 --optimizer adam --min-params-to-wrap 100000000  \
 --save-dir ${SAVE_PATH}  --skip-invalid-size-inputs-valid-test --memory-efficient-fp16  \
 --best-checkpoint-metric loss  --ddp-backend fully_sharded

## Evaluate
mkdir -p ${SAVE_PATH}/results
TGTS="eu,pt,bg,sk,zh,sl,de,hr,nb,ga,rw,as,fy,mr,se"
src=en
for tgt in ${TGTS//,/ }; do
    echo predict $src to $tgt
    FSRC=${DATA_DIR}/tok/test.${src}-${tgt}.${src}
    FTGT=${DATA_DIR}/raw/test.${src}-${tgt}.${tgt}
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}

    if [ ${tgt} == 'ru' ] || [ ${tgt} == 'fr' ] || [ ${tgt} == 'km' ] || [ ${tgt} == 'ta' ] || [ ${tgt} == 'ug' ] || [ ${tgt} == 'or' ]; then
        cat $FSRC | python scripts/truncate.py | \
        python fairseq_cli/interactive.py ${DATA_BIN}  --path $SAVE_PATH/checkpoint_best.pt \
        --task translation_multi_simple_epoch \
        --encoder-langtok tgt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --source-lang ${src} --target-lang ${tgt} \
        ${REPLACE_BOS_EOS} ${HOMOBATCH} \
        --buffer-size 1024 --batch-size 100 \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar \
        --enable-lang-ids | \
        grep -P "^D" | cut -f 3- > $FOUT
    else
        fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
            --langs ${LANGS} \
            --lang-pairs ${LANG_PAIRS} \
            --task translation_multi_simple_epoch \
            --remove-bpe "sentencepiece" \
            --enable-lang-ids \
            --sacrebleu \
            --encoder-langtok tgt \
            ${REPLACE_BOS_EOS} ${HOMOBATCH} \
            --source-lang ${src} --target-lang ${tgt} \
            --batch-size 100 \
            --beam 5 --lenpen 1.0 \
            --no-progress-bar |\
            sort -t '-' -nk 2 | grep -P "^D-" | cut -f 3- > $FOUT
fi
SACREBLEU_FORMAT=text sacrebleu -tok flores200 -w 2 $FOUT < ${FTGT} > $FOUT.bleu
cat ${FOUT}.bleu
done


SRCS="eu,pt,bg,sk,zh,sl,de,hr,nb,ga,rw,as,fy,mr,se"
tgt=en
mkdir -p ${SAVE_PATH}/results
for src in ${SRCS//,/ }; do
    echo predict $src to $tgt
    FSRC=${DATA_DIR}/tok/test.${tgt}-${src}.${src}
    FTGT=${DATA_DIR}/raw/test.${tgt}-${src}.${tgt}
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
    if [ ${src} == 'ru' ] || [ ${src} == 'fr' ] || [ ${src} == 'km' ] || [ ${src} == 'ta' ] || [ ${src} == 'ug' ] || [ ${src} == 'or' ]; then
        cat $FSRC | python scripts/truncate.py | \
        python fairseq_cli/interactive.py ${DATA_BIN}  --path $SAVE_PATH/checkpoint_best.pt \
        --task translation_multi_simple_epoch \
        --encoder-langtok tgt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --source-lang ${src} --target-lang ${tgt} \
        ${REPLACE_BOS_EOS} ${HOMOBATCH} \
        --buffer-size 1024 --batch-size 100 \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar \
        --enable-lang-ids | \
        grep -P "^D" | cut -f 3- > $FOUT
    else
        fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
            --langs ${LANGS} \
            --lang-pairs ${LANG_PAIRS} \
            --task translation_multi_simple_epoch \
            --sacrebleu \
            --encoder-langtok tgt \
            --remove-bpe "sentencepiece" \
            ${HOMOBATCH} \
            --source-lang ${src} --target-lang ${tgt} \
            --enable-lang-ids \
            --batch-size 100  \
            --beam 5 --lenpen 1.0 \
            --no-progress-bar |\
            sort -t '-' -nk 2 | grep -P "^D-" | cut -f 3- > $FOUT
    fi
    SACREBLEU_FORMAT=text sacrebleu -tok flores200 -w 2 $FOUT < ${FTGT} > $FOUT.bleu
    cat ${FOUT}.bleu
done


SRCS="eu,pt,bg,sk,zh,sl,de,hr,nb,ga,rw,as,fy,mr,se"
tgt=en
for src in ${SRCS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
done

# Print
TGTS="eu,pt,bg,sk,zh,sl,de,hr,nb,ga,rw,as,fy,mr,se"
src=en
for tgt in ${TGTS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
done



echo 'eng->xx'
python ./get_opus15_mean.py \
    --input ${SAVE_PATH}/results/
    
echo 'xx->eng'
python ./get_opus15_mean.py \
    --input ${SAVE_PATH}/results/ \
    --engtgt 1