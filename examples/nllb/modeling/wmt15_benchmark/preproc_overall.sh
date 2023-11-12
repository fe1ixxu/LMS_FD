datadir=/private/home/shru/wmt_datasets/raw_v2
TEMP=5
# python scripts/wmt30/create_temp_sampled_data_for_spm.py \
#     --files-to-line $datadir/lines_to_file.tsv \
#     --lang-position-in-filename 1 \
#     --datadir $datadir \
#     --temp $TEMP \
#     --total-lines 500000000

# # change input to up_down_sample_files.sh and run it
# bash scripts/wmt30/up_down_sample_files.sh

# bash scripts/wmt30/learn_spm_v2.sh

# apply learned SPM
# bash scripts/wmt30/apply_spm.sh

# SPM vocab to fairseq dict
# bash spm_vocab_to_fairseq_dict.sh

# binarize
# bash binarize.sh
