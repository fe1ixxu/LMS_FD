SPM_VOCAB="/private/home/shru/wmt_datasets/raw_v2/spm_64000_500M.vocab"
FAIRSEQ_DICT="/private/home/shru/wmt_datasets/raw_v2/spm_64000_500M.dict"
tail -n +4 $SPM_VOCAB | awk '{print $1" "1}' > $FAIRSEQ_DICT
