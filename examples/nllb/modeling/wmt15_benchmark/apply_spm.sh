inputdir="/private/home/shru/wmt_datasets/raw_v2"
outputdir="/private/home/shru/wmt_datasets/raw_v2/spm_applied"
logdir="/private/home/shru/wmt_datasets/raw_v2/spm_applied_logs"
spm_model=/private/home/shru/wmt_datasets/raw_v2/spm_64000_500M.model
mkdir -p ${outputdir}
mkdir -p ${logdir}

# replace train, with valid, test
# for lang in "$@" ; do
for type in valid test ; do
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    # echo $filename
    # type=`echo ${filename} | cut -d'.' -f2`
    # lp=`echo ${filename} | cut -d'.' -f2`
    # echo ${lp}
    # src=`echo $lp | cut -d'-' -f1`
    # tgt=`echo $lp | cut -d'-' -f2`
    # echo $src $tgt
    src=en
    tgt=$lang
    filename="${type}.${src}-${tgt}"
    inputpath="${inputdir}/${filename}"
    outputpath="${outputdir}/${filename}"
    logpath="${logdir}/${filename}"
    echo ${inputpath} ${outputpath}
    python scripts/spm_encode.py \
        --model $spm_model \
        --inputs ${inputpath}.${src} ${inputpath}.${tgt} \
        --outputs ${outputpath}.${src} ${outputpath}.${tgt} &> ${logpath}
done
done
