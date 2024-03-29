# -*- coding: utf-8 -*-

# =============================================================================
# ------------THIS SCRIPT NEEDS TO BE RUN IN PYTHON3 (DIFFERING THAN THE REST)
# =============================================================================

from jiwer import wer
import pandas as pd  
import os


RESULTS_PATH = "logs"

# =============================================================================
# ------------------------------Preparing pathes
# =============================================================================

benchmark_filepathes = []
summ_filepathes = []
for directory in os.listdir(RESULTS_PATH):
    directory = os.path.join(RESULTS_PATH, directory)
    for subdirectory in os.listdir(directory):
        subdirectory = os.path.join(directory, subdirectory)
        for file in os.listdir(subdirectory):
            print(file + '\n')
            if file.startswith("paddlespeech_benchmark"):
                file = os.path.join(subdirectory, file)
                benchmark_filepathes.append(file)
            elif file.startswith("summ"):
                file = os.path.join(subdirectory, file)
                summ_filepathes.append(file)
            
    

# =============================================================================
# ---Running the Kaldi STT Engine by running through the audio files
# =============================================================================

for i in range(len(benchmark_filepathes)):
    avg_wer = 0
    
    summ_file = pd.read_csv(summ_filepathes[i], header=None)
    results_data = pd.read_csv(benchmark_filepathes[i])[:-2]
#    print(str(wer(results_data["actual_text"], results_data["processed_text"], standardize=True)))
#    results_data["wer"] = wer(results_data["actual_text"], results_data["processed_text"], standardize=True)
    results_data["wer"] = results_data.apply(lambda row: wer(row["actual_text"], row["processed_text"], standardize=True), axis = 1)
    for current_wer in results_data["wer"]:
        current_wer = round(current_wer,3)
        avg_wer += current_wer
    avg_wer /= len(results_data)
    summ_file[1][3]=avg_wer
    print("Average WER = " + str(avg_wer))
    
    benchmark_dir = "/".join(benchmark_filepathes[i].split("/")[:-1])
    results_data.to_csv(os.path.join(benchmark_dir, "benchmark_fixed.csv"))
    print("Benchmark fixed and exported in benchmark_fixed.csv .")
    summ_file.to_csv(os.path.join(benchmark_dir, "summary_fixed.csv"))
    print("Summary fixed and exported in summary_fixed.csv .")
    
print( str(len(benchmark_filepathes)) + " benchmarks has been fixed supposedly and exported.")