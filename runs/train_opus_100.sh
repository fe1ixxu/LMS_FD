#!/bin/bash
LMS_RANK=${1:-4}
LMS_FREQ=${2:-1}
LMS_TYPE=${3:-'pair'}


LANGS="af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu"
LANG_PAIRS="es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig"

FREQ=4
TEMPERATURE=5
DATA_DIR=./opus-100
DATA_BIN=${DATA_DIR}/data-bin/
SAVE_PATH=./checkpoint/OPUS_100_${LMS_TYPE}_rank_${LMS_RANK}_FFN
MAX_UPDATES=100000
ARCH=transformer
MAX_TOKENS=4096

LAYER=6
DIM=1024
FFN_DIM=4096
HEADS=16

echo "LMS_TYPE ${LMS_TYPE}; LMS_RANK ${LMS_RANK}"

HOMOBATCH="--one-dataset-per-batch"
CRITERION='label_smoothed_cross_entropy'


 python train.py  ${DATA_BIN} --arch ${ARCH}  --task translation_multi_simple_epoch \
 --lms-freq ${LMS_FREQ} --lms-type ${LMS_TYPE} --lms-rank ${LMS_RANK} ${HOMOBATCH} --lms-ffn --fd-weight 1 \
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
TGTS='af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu'
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


SRCS='af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu'
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

SAVE_PATH=/brtx/606-nvme1/haoranxu/opus-checkpoint/base_xl_beta0.1_lid_5000_loga0.95_budget15000_full
SRCS='af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu'
tgt=en
for src in ${SRCS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
done

# Print
TGTS='af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu'
src=en
for tgt in ${TGTS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 3
done



echo 'eng->xx'
python ./get_opus_mean.py \
    --input ${SAVE_PATH}/results/
    
echo 'xx->eng'
python ./get_opus_mean.py \
    --input ${SAVE_PATH}/results/ \
    --engtgt 1