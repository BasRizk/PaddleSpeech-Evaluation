#! /usr/bin/env bash

mkdir -p tests/LibriSpeech

# download data, generate manifests
PYTHONPATH=.:$PYTHONPATH python librispeech.py \
--manifest_prefix='tests/LibriSpeech/manifest' \
--target_dir='tests/LibriSpeech' \
--full_download='False'

if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

echo "LibriSpeech Test and Dev Data preparation done."
exit 0
