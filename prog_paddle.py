# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle

import sys
sys.path.insert(1, './PaddleSpeech')
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.error_rate import char_errors, word_errors

from os import path, makedirs
import soundfile as sf


from timeit import default_timer as timer
from basic_utils import get_platform_id, document_machine
from basic_utils import prepare_pathes, get_metafiles_pathes


TEST_PATH="tests/LibriSpeech/test-other"
#TEST_PATH="tests/iisys"
MODEL_VERSION="librispeech"

model_dir = "PaddleSpeech/models"

test_manifest = 'manifest.' + TEST_PATH.split("/")[-1]

mean_std_path = path.join(model_dir, MODEL_VERSION,'mean_std.npz')
vocab_path = path.join(model_dir,  MODEL_VERSION, 'vocab.txt')
model_path= path.join(model_dir,  MODEL_VERSION, 'params.tar.gz' )
lang_model_path= path.join(model_dir,  "lm", 'common_crawl_00.prune01111.trie.klm')

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
log_file = open(log_filepath, "w")
summ_file = open(summ_filepath, "w")

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

print('Loading inference model from files {}'.format(model_path))
log_file.write('Loading inference model from files {}'.format(model_path))
inf_model_load_start = timer()
ds2_model = DeepSpeech2Model(
    vocab_size=data_generator.vocab_size,
    num_conv_layers=num_conv_layers,
    num_rnn_layers=num_rnn_layers,
    rnn_layer_size=rnn_layer_size,
    use_gru=USING_GRU,
    pretrained_model_path=model_path,
    share_rnn_weights=share_rnn_weights)
inf_model_load_end = timer() - inf_model_load_start
print('Loaded inference model in {:.3}s.'.format(inf_model_load_end))
log_file.write('Loaded inference model in {:.3}s.'.format(inf_model_load_end))
summ_file.write('Loaded inference model in,{:.3}s.'.format(inf_model_load_end))

# decoders only accept string encoded in utf-8
vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

if decoding_method == "ctc_beam_search":
    print('Loading language model (scorer) from files {}'.format(lang_model_path),
          file=sys.stderr)
    log_file.write('Loading language model (scorer) from files {}'.format(lang_model_path))
    lm_load_start = timer()
    ds2_model.init_ext_scorer(alpha, beta, lang_model_path,
                              vocab_list)
    lm_load_end = timer() - lm_load_start
    print('Loaded language model (scorer)) in {:.3}s.'.format(lm_load_end))
    log_file.write('Loaded language model (scorer) in {:.3}s.'.format(lm_load_end))
    summ_file.write('Loaded language model (scorer) in,{:.3}s.'.format(lm_load_end))
    
    
errors_func = char_errors if error_rate_type == 'cer' else word_errors
errors_sum, len_refs, num_ins = 0.0, 0, 0

##############################################################################
# ---Running the PADDLE DS2 STT Engine by running through the audio files
##############################################################################
import pandas as pd
audio_files = pd.read_json(test_manifest, lines=True)
processed_data = "filename,length(sec),proc_time(sec),wer,actual_text,processed_text\n"
avg_wer = 0
avg_proc_time = 0
num_of_audiofiles = len([item for sublist in audio_pathes for item in sublist])
current_audio_number = 1

for infer_data, sample_audio_path, imported_text in \
            zip(batch_reader(), audio_files["audio_filepath"], audio_files["text"]):
    
    print("\n=> Progress = " + "{0:.2f}".format((current_audio_number/num_of_audiofiles)*100) + "%\n" )
    current_audio_number+=1
    
    infer_data_text = infer_data[0][1]
    if infer_data_text != imported_text:
        print("WARNING \n")
        print("infer_data_text: \n")
        print(infer_data_text + "\n")
        print("imported_text: \n")
        print(imported_text + "\n")
        break
        
    audio, fs = sf.read(sample_audio_path, dtype='int16')
    audio_len = len(audio)/fs 
        
    print('Running inference.', file=sys.stderr)
    inference_start = timer()

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

    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_len))       
    proc_time = round(inference_end, 3)
    
    target_transcripts = [data[1] for data in infer_data]
    
    for target, result in zip(target_transcripts, result_transcripts):
        errors, len_ref = errors_func(target, result)
        errors_sum += errors
        len_refs += len_ref
        num_ins += 1
    print("Error rate [%s] (%d/?) = %f" %
          (error_rate_type, num_ins, errors_sum / len_refs))

    # Processing WORD ERROR RATE (WER)
    actual_text = imported_text.lower()
    current_wer = -1 #wer(actual_text, processed_text, standardize=True)
    current_wer = round(current_wer,3)
    
    # Accumlated data
    avg_proc_time += (proc_time/(audio_len))
    avg_wer += current_wer
    
    processed_text = result_transcripts[0]
    progress_row = sample_audio_path + "," + str(audio_len) + "," + str(proc_time)  + "," +\
                    str(current_wer) + "," + actual_text + "," + processed_text
                     
    if(VERBOSE):
        print("# Audio number " + str(current_audio_number) + "/" + str(num_of_audiofiles) +"\n" +\
              "# File (" + sample_audio_path + "):\n" +\
              "# - " + str(audio_len) + " seconds long.\n"+\
              "# - actual    text: '" + actual_text + "'\n" +\
              "# - processed text: '" + processed_text + "'\n" +\
              "# - processed in "  + str(proc_time) + " seconds.\n"
              "# - WER = "  + str(current_wer) + "\n")
              
    log_file.write("# Audio number " + str(current_audio_number) + "/" + str(num_of_audiofiles) +"\n" +\
          "# File (" + sample_audio_path + "):\n" +\
          "# - " + str(audio_len) + " seconds long.\n"+\
          "# - actual    text: '" + actual_text + "'\n" +\
          "# - processed text: '" + processed_text + "'\n" +\
          "# - processed in "  + str(proc_time) + " seconds.\n"
          "# - WER = "  + str(current_wer) + "\n")
    
              
    processed_data+= progress_row + "\n"
    
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
    

