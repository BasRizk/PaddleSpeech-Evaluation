#! /usr/bin/env bash

cd /home/ironbas3/PaddleSpeech > /dev/null

# evaluate model
python -u test.py \
--batch_size=1 \
--trainer_count=1 \
--beam_size=500 \
--num_proc_bsearch=1 \
--num_proc_data=1 \
--alpha=2.5 \
--beta=0.3 \
--cutoff_prob=1.0 \
--cutoff_top_n=40 \
--use_gru=False \
--use_gpu=False \
--test_manifest='data/librispeech/manifest.test-clean' \
--mean_std_path='models/librispeech/mean_std.npz' \
--vocab_path='models/librispeech/vocab.txt' \
--model_path='models/librispeech/params.tar.gz' \
--lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='wer' \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--use_gru=False \
--share_rnn_weights=True \
--specgram_type='linear'

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
