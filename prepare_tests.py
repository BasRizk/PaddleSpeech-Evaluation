"""
Script to prepare tests for programatically evaluating paddlepaddle DeepSpeech2

adapted from official repo librispeech.py script

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import soundfile
import json
import codecs


import sys
sys.path.insert(1, './PaddleSpeech')
from data_utils.utility import download, unpack

TARGET_DIR = "tests"
MANIFEST_PREFIX = "manifest"
    
URL_ROOT = "http://www.openslr.org/resources/12"
URL_TEST_OTHER = URL_ROOT + "/test-other.tar.gz"
MD5_TEST_OTHER = "fb5a50374b501bb3bac4815ee91d3135"

def create_manifest(data_dir, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    for folder, _, filelist in sorted(os.walk(data_dir)):
        text_filelist = [
            filename for filename in filelist if filename.endswith('trans.txt')
        ]
        if len(text_filelist) > 0:
            text_filepath = os.path.join(folder, text_filelist[0])
            for line in open(text_filepath):
                segments = line.strip().split()
                text = ' '.join(segments[1:]).lower()
                audio_filepath = os.path.join(folder,
                                              segments[0] + '.wav')
                audio_data, samplerate = soundfile.read(audio_filepath)
                duration = float(len(audio_data)) / samplerate
                json_lines.append(
                    json.dumps({
                        'audio_filepath': audio_filepath,
                        'duration': duration,
                        'text': text
                    }))
    with codecs.open(manifest_path, 'w', 'utf-8') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create summmary manifest file.
    """
    data_dir = os.path.join(target_dir, "LibriSpeech")
    if not os.path.exists(data_dir):
        # download
        filepath = download(url, md5sum, target_dir)
        # unpack
        unpack(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              data_dir)
    # create manifest json file
    if not os.path.exists(manifest_path):
        create_manifest(data_dir, manifest_path)


def main():
    prepare_dataset(
        url=URL_TEST_OTHER,
        md5sum=MD5_TEST_OTHER,
        target_dir=TARGET_DIR,
        manifest_path=MANIFEST_PREFIX + ".test-other")
    
    if os.path.exists(os.path.join(TARGET_DIR, "iisys")):
        create_manifest(
            data_dir=TARGET_DIR,
            manifest_path=MANIFEST_PREFIX + ".iisys-en")
    else:
        print("iisys data-set dir needs to be places in %s" % TARGET_DIR)
    


if __name__ == '__main__':
    main()
