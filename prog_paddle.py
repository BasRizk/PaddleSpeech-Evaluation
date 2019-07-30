# -*- coding: utf-8 -*-

import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.error_rate import char_errors, word_errors

from os import path, makedirs
from jiwer import wer    
import soundfile as sf
import sys

from timeit import default_timer as timer
from utils import get_platform_id, document_machine
from utils import prepare_pathes, get_metafiles_pathes

MODEL_VERSION="librispeech"

TEST_PATH="tests/LibriSpeech/test-other"
#TEST_PATH="tests/iisys"


test_manifest = path.join(TEST_PATH, 'manifest.test-clean')
mean_std_path = path.join(TEST_PATH,'mean_std.npz')
vocab_path = path.join('models',  MODEL_VERSION, 'vocab.txt')
model_path= path.join('models',  MODEL_VERSION, 'params.tar.gz' )
lang_model_path= path.join('models',  MODEL_VERSION, 'common_crawl_00.prune01111.trie.klm')

IS_GLOBAL_DIRECTORIES = True
USING_GPU = False
USING_GRU = False
USE_LANGUAGE_MODEL = True
USE_TFLITE = False
USE_MEMORY_MAPPED_MODEL = True
VERBOSE = True
assert(path.exists(TEST_PATH))

try:
    TEST_CORPUS = TEST_PATH.split("/")[1]
    if TEST_CORPUS.lower() == "librispeech":
        TEST_CORPUS += "_" + TEST_PATH.split("/")[2]
except:
    print("WARNING: Path 2nd index does not exist.\n")

if  TEST_CORPUS == "iisys":
    IS_TSV = True
    IS_RECURSIVE_DIRECTORIES = False
else:
    IS_TSV = False
    IS_RECURSIVE_DIRECTORIES = True
    
if IS_TSV:
    TS_INPUT = "tsv"
    AUDIO_INPUT = "wav"
else:
    TS_INPUT = "txt"
    AUDIO_INPUT = "wav"

try:
    if TEST_PATH.split("/")[2] == "Sprecher":
        AUDIO_INPUT="flac"
except:
    print("WARNING: Path 3rd index does not exist.\n")

##############################################################################
# ------------------------Documenting Machine ID
##############################################################################

platform_id = get_platform_id()

if USE_LANGUAGE_MODEL:
    platform_id += "_use_lm"
    
if USE_MEMORY_MAPPED_MODEL and not USE_TFLITE:
    platform_id += "_use_pbmm"
elif USE_TFLITE:
    platform_id += "_use_tflite"
if USING_GPU:
    platform_id += "_use_gpu"

if TEST_CORPUS:
    platform_id = TEST_CORPUS + "_" + AUDIO_INPUT + "_" + platform_id
    
platform_meta_path = "logs/v" + MODEL_VERSION + "/" + platform_id


    
if not path.exists(platform_meta_path):
    makedirs(platform_meta_path)
    
document_machine(platform_meta_path, USING_GPU)

##############################################################################
# ------------------------------Preparing pathes
##############################################################################

log_filepath, benchmark_filepath, summ_filepath = get_metafiles_pathes(platform_meta_path)

test_directories = prepare_pathes(TEST_PATH, recursive = IS_RECURSIVE_DIRECTORIES)
audio_pathes = list()
text_pathes = list()
for d in test_directories:
    audio_pathes.append(prepare_pathes(d, AUDIO_INPUT, recursive = False))
    text_pathes.append(prepare_pathes(d, TS_INPUT, recursive = False))
audio_pathes.sort()
text_pathes.sort()    

##############################################################################
# ----------------------------- Model Loading 
##############################################################################
batch_size=1 
trainer_count=1 
beam_size=500 
num_proc_bsearch=1 
num_proc_data=1 
alpha=2.5 
beta=0.3 
cutoff_prob=1.0 
cutoff_top_n=40 
decoding_method='ctc_beam_search' 
error_rate_type='wer' 
num_conv_layers=2 
num_rnn_layers=3 
rnn_layer_size=2048 
share_rnn_weights=True 
specgram_type='linear'

paddle.init(use_gpu=USING_GPU,
                rnn_use_batch=True,
                trainer_count=trainer_count)

data_generator = DataGenerator(
    vocab_filepath=vocab_path,
    mean_std_filepath=mean_std_path,
    augmentation_config='{}',
    specgram_type=specgram_type,
    num_threads=num_proc_data,
    keep_transcription_text=True)
batch_reader = data_generator.batch_reader_creator(
    manifest_path=test_manifest,
    batch_size=batch_size,
    min_batch_size=1,
    sortagrad=False,
    shuffle_method=None)

ds2_model = DeepSpeech2Model(
    vocab_size=data_generator.vocab_size,
    num_conv_layers=num_conv_layers,
    num_rnn_layers=num_rnn_layers,
    rnn_layer_size=rnn_layer_size,
    use_gru=USING_GRU,
    pretrained_model_path=model_path,
    share_rnn_weights=share_rnn_weights)

# decoders only accept string encoded in utf-8
vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

if decoding_method == "ctc_beam_search":
    ds2_model.init_ext_scorer(alpha, beta, lang_model_path,
                              vocab_list)
errors_func = char_errors if error_rate_type == 'cer' else word_errors
errors_sum, len_refs, num_ins = 0.0, 0, 0

##############################################################################
# ---Running the PADDLE DS2 STT Engine by running through the audio files
##############################################################################
processed_data = "filename,length(sec),proc_time(sec),wer,actual_text,processed_text\n"
avg_wer = 0
avg_proc_time = 0
num_of_audiofiles = len([item for sublist in audio_pathes for item in sublist])
current_audio_number = 1
all_text_pathes = [item for sublist in text_pathes for item in sublist]

for infer_data in batch_reader():
    probs_split = ds2_model.infer_batch_probs(
        infer_data=infer_data,
        feeding_dict=data_generator.feeding)

    if decoding_method == "ctc_greedy":
        result_transcripts = ds2_model.decode_batch_greedy(
            probs_split=probs_split,
            vocab_list=vocab_list)
    else:
        result_transcripts = ds2_model.decode_batch_beam_search(
            probs_split=probs_split,
            beam_alpha=alpha,
            beam_beta=beta,
            beam_size=beam_size,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n,
            vocab_list=vocab_list,
            num_processes=num_proc_bsearch)
        
    target_transcripts = [data[1] for data in infer_data]
    
    for target, result in zip(target_transcripts, result_transcripts):
        errors, len_ref = errors_func(target, result)
        errors_sum += errors
        len_refs += len_ref
        num_ins += 1
    print("Error rate [%s] (%d/?) = %f" %
          (error_rate_type, num_ins, errors_sum / len_refs))
print("Final error rate [%s] (%d/%d) = %f" %
      (error_rate_type, num_ins, num_ins, errors_sum / len_refs))


##############################################################################
# ---------------Finalizing processed data and Saving Logs
##############################################################################
avg_proc_time /= current_audio_number
avg_wer /= current_audio_number
if(VERBOSE):
    print("Avg. Proc. time (sec/second of audio) = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.write("Avg. Proc. time/sec = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
summ_file.write("Avg. Proc. time/sec," + str(avg_proc_time) + "\n" +\
          "Avg. WER," + str(avg_wer))
log_file.close()
summ_file.close()
processed_data+= "AvgProcTime (sec/second of audio)," + str(avg_proc_time) + ",,,," + "\n"
processed_data+= "AvgWER," + str(avg_wer) + ",,,,,"+ "\n"


with open(benchmark_filepath, 'w') as f:
    for line in processed_data:
        f.write(line)
    

