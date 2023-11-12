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

#basepath="/large_experiments/moe/shru/wmt30/en_to_many/.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.initlr1e-07.warmup4000.lr0.001.clip0.1.drop0.0.wd0.0.ls0.1.seed2.max_pos512.bsz16.no_c10d.det.ves100000000.ngpu32"
#basepath="/large_experiments/moe/shru/wmt30/en_to_many_moe_32experts/32experts.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.initlr1e-07.warmup4000.lr0.001.clip0.1.drop0.0.wd0.0.ls0.1.seed2.max_pos512.bsz16.2ndexpall.no_c10d.det.ves100000000.ngpu32"
basepath="/large_experiments/moe/shru/wmt30/en_to_many_moe_32experts_v2/32experts_v2.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup4000.lr0.001.clip0.1.drop0.0.wd0.0.ls0.1.seed2.max_pos512.bsz16.2ndexpall.no_c10d.det.ves100000000.ngpu32"
#basepath="/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup40000.lr0.001.clip0.0.drop0.0.wd0.0.ls0.1.maxtok3584.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
basepath="/large_experiments/moe/shru/wmt30/en_to_many/dense.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr0.001.clip0.0.drop0.0.wd0.0.ls0.1.maxtok3584.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
basepath="/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.mu50000.uf1.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr0.001.clip0.0.drop0.0.wd0.0.ls0.1.maxtok3584.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
#for lang in cs 'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
mkdir -p ${basepath}/bleu
direction="many_to_en"
for type in test ; do
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    if [ $direction == "many_to_en" ] ; then
        lang_pair="$lang-en"
        src=$lang
        tgt=en
    else
        lang_pair="en-$lang"
        src=en
        tgt=$lang
    fi
    key_lang_pair="$lang-en"
    if [ "${type}" == "valid" ] ; then
        testset="${valid_set[${key_lang_pair}]}"
    else
        testset="${test_set[${key_lang_pair}]}"
    fi
    if [ "$lang_pair" == "hi-en" ] || [ "$lang_pair" == "en-hi" ] ; then
        cat ${basepath}/gen/${type}.$src-$tgt | grep -P "^H" | sort -V | cut -f 3- | sacrebleu ~/wmt_datasets/raw/${type}.hi-en.$tgt | tee ${basepath}/bleu/${type}.$src-$tgt
    else
        cat ${basepath}/gen/${type}.$src-$tgt | grep -P "^H" | sort -V | cut -f 3- | sacrebleu -t $testset -l ${lang_pair} | tee ${basepath}/bleu/${type}.$src-$tgt
    fi
done
done

for type in test ; do
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    if [ $direction == "many_to_en" ] ; then
        src=$lang
        tgt=en
    else
        src=en
        tgt=$lang
    fi
bleu=`cat ${basepath}/bleu/${type}.$src-$tgt | grep "BLEU" | cut -d' ' -f3`
echo -e "$lang\t$bleu"
done
done
