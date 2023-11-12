inputdir="/private/home/shru/wmt_datasets/raw_v2/spm_applied"
outputdir="/private/home/shru/wmt_datasets/raw_v2/binarized"
dict="/private/home/shru/wmt_datasets/raw_v2/spm_64000_500M.dict"

mkdir -p ${outputdir}

#for lp in "$@" ; do
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    src=en
    tgt=$lang
    lp="$src-$tgt"
    # echo $lp
    # src=`echo ${lp} | cut -d'-' -f1`
    # tgt=`echo ${lp} | cut -d'-' -f2`
    if [ $src == "en" ] ; then
        python fairseq_cli/preprocess.py \
            -s ${src} -t ${tgt} \
            --validpref ${inputdir}/valid.${lp} \
            --testpref ${inputdir}/test.${lp} \
            --destdir ${outputdir}/${lp} \
            --srcdict ${dict} \
            --joined-dictionary \
            --workers 72
            # --trainpref ${inputdir}/train.${lp} \
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.$tgt.bin ${outputdir}/en-$tgt/train.en-$tgt.$tgt.bin
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.$tgt.idx ${outputdir}/en-$tgt/train.en-$tgt.$tgt.idx
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.en.bin ${outputdir}/en-$tgt/train.en-$tgt.en.bin
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.en.idx ${outputdir}/en-$tgt/train.en-$tgt.en.idx
    else
        python fairseq_cli/preprocess.py \
            -s ${src} -t ${tgt} \
            --trainpref ${inputdir}/train.${lp} \
            --validpref ${inputdir}/valid.${lp} \
            --testpref ${inputdir}/test.${lp} \
            --destdir ${outputdir}/${lp} \
            --srcdict ${dict} \
            --joined-dictionary \
            --workers 72
    fi
done
