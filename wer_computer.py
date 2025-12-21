from utils import asr_client

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
import datasets
import json
import numpy as np
import os
import dotenv

INFERENCE_FUNCTION = asr_client.inferenceFunction
LANGUAGE = "Greek"
OUTPUT_PATH = "results/el/canary-1b-v2"

#
##
###
##
#

def computeWer(dataset, text_column_name, inferenceFunction, language):

    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")

    predictions_list = [inferenceFunction(sample["audio"], language) for sample in dataset]
    references_list = dataset[text_column_name]

    norm_references_list = [normalizer(reference) for reference in references_list]
    norm_predictions_list = [normalizer(prediction) for prediction in predictions_list]

    wers_list = []
    for ref, pred in zip(norm_references_list, norm_predictions_list):
        # Handle empty strings to avoid division by zero
        if not ref.strip() and not pred.strip():
            wers_list.append(0.0)  # Both empty = perfect match
        elif not ref.strip():
            wers_list.append(1.0)  # Empty reference, non-empty prediction = 100% error
        else:
            wers_list.append(wer_metric.compute(references=[ref], predictions=[pred]))

    cardinality = len(wers_list)

    return wers_list, cardinality

def computeDataAndStats(dataset, text_column_name, inferenceFunction, language):
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    wers_list, cardinality = computeWer(dataset, text_column_name, inferenceFunction, language)
    print(f"Mean WER = {np.mean(wers_list)} [{cardinality} samples]")
    stats = {
        "num_samples": cardinality,
        "mean": np.mean(wers_list),
        "std": np.std(wers_list),
        "min": np.min(wers_list),
        "max": np.max(wers_list),
    }
    return wers_list, stats

def saveOnDisk(data, stats):

    # Raw data
    try:
        with open(os.path.join(OUTPUT_PATH, "data.json"), "r") as f:
            output_data = json.load(f)
    except FileNotFoundError:
        output_data = {}
    output_data.update(data)
    with open(os.path.join(OUTPUT_PATH, "data.json"), "w") as f:
        json.dump(output_data, f)
    
    # Stats
    try:
        with open(os.path.join(OUTPUT_PATH, "stats.json"), "r") as f:
            output_stats = json.load(f)
    except FileNotFoundError:
        output_stats = {}
    output_stats.update(stats)    
    with open(os.path.join(OUTPUT_PATH, "stats.json"), "w") as f:
        json.dump(output_stats, f)

#
##
###
##
#

os.makedirs(OUTPUT_PATH, exist_ok=True)
try:
    with open(os.path.join(OUTPUT_PATH, "stats.json"), "r") as f:
        output_stats = json.load(f)
except FileNotFoundError:
    output_stats = {}
already_computed_datasets = output_stats.keys()
dotenv.load_dotenv(".env.secrets")

# ################################ Voxpopuli #################################
# if "voxpopuli" not in already_computed_datasets:
#     print("Testing Voxpopuli...")
#     voxpopuli = datasets.load_dataset(
#         "facebook/voxpopuli", "cs", split="test", trust_remote_code=True,
#     )  # italian: 1177 samples (too often with incorrect labels)
#     voxpopuli_wers_list, voxpopuli_stats = computeDataAndStats(
#         dataset=voxpopuli,
#         text_column_name="raw_text",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"voxpopuli": voxpopuli_wers_list}, stats={"voxpopuli": voxpopuli_stats})

# ################################ MLS #################################
# if "mls" not in already_computed_datasets:
#     print("Testing MLS...")
#     mls = datasets.load_dataset(
#         "facebook/multilingual_librispeech", "dutch", split="test"
#     )  # italian: 1260 samples
#     mls_wers_list, mls_stats = computeDataAndStats(
#         dataset=mls,
#         text_column_name="transcript",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"mls": mls_wers_list}, stats={"mls": mls_stats})

################################ Common Voice 22.0 #################################
if "cv_22_0" not in already_computed_datasets:
    print("Testing CV-22.0...")
    cv_22_0 = datasets.load_dataset(
        "fsicoli/common_voice_22_0",
        "el",
        split="test",
        trust_remote_code=True,
        token=True,
    )
    cv_22_0_wers_list, cv_22_0_stats = computeDataAndStats(
        dataset=cv_22_0,
        text_column_name="sentence",
        inferenceFunction=INFERENCE_FUNCTION,
        language=LANGUAGE,
    )
    saveOnDisk(data={"cv_22_0": cv_22_0_wers_list}, stats={"cv_22_0": cv_22_0_stats})

# ################################# Minds14 #################################
# if "mind_14" not in already_computed_datasets:
#     print("Testing Minds14...")
#     mind_14 = datasets.load_dataset(
#         "PolyAI/minds14", "cs-CZ", split="train", trust_remote_code=True
#     )  # italian: (too often with incorrect labels)
#     mind_14_wers_list, mind_14_stats = computeDataAndStats(
#         dataset=mind_14,
#         text_column_name="transcription",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"mind_14": mind_14_wers_list}, stats={"mind_14": mind_14_stats})

# ################################# Speech-MASSIVE-test #################################
# if "sm_test" not in already_computed_datasets:
#     print("Testing Speech-MASSIVE-test...")
#     sm_test = datasets.load_dataset(
#         "FBK-MT/Speech-MASSIVE-test",
#         "nl-NL",
#         split="test",
#         trust_remote_code=True,
#     )
#     sm_test_wers_list, sm_test_stats = computeDataAndStats(
#         dataset=sm_test,
#         text_column_name="utt",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"sm_test": sm_test_wers_list}, stats={"sm_test": sm_test_stats})

# ################################# Romanian speech synthesis 0.8.1 #################################
# if "rss_0_8_1" not in already_computed_datasets:
#     print("Testing Romanian speech synthesis 0.8.1...")
#     rss_0_8_1 = datasets.load_dataset(
#         "gigant/romanian_speech_synthesis_0_8_1",
#         split="test",
#         trust_remote_code=True
#     )
#     rss_0_8_1_wers_list, rss_0_8_1_stats = computeDataAndStats(
#         dataset=rss_0_8_1,
#         text_column_name="sentence",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"rss_0_8_1": rss_0_8_1_wers_list}, stats={"rss_0_8_1": rss_0_8_1_stats})

# ################################# Echo (romanian only) #################################
# if "echo" not in already_computed_datasets:
#     print("Testing Echo...")
#     echo = datasets.load_dataset(
#         "readerbench/echo", split="test", trust_remote_code=True
#     )
#     echo_wers_list, echo_stats = computeDataAndStats(
#         dataset=echo,
#         text_column_name="text",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"echo": echo_wers_list}, stats={"echo": echo_stats})

# ################################# EuroSpeech #################################
if "eurospeech" not in already_computed_datasets:
    print("Testing EuroSpeech...")
    eurospeech = datasets.load_dataset(
        "disco-eth/EuroSpeech",
        "greece",
        split="test",
        # split="validation", # for italian
        trust_remote_code=True,
    )
    eurospeech_wers_list, eurospeech_stats = computeDataAndStats(
        dataset=eurospeech,
        text_column_name="human_transcript",
        inferenceFunction=INFERENCE_FUNCTION,
        language=LANGUAGE,
    )
    saveOnDisk(data={"eurospeech": eurospeech_wers_list}, stats={"eurospeech": eurospeech_stats})

################################# Fleurs #################################
if "fleurs" not in already_computed_datasets:
    print("Testing Fleurs...")
    fleurs = datasets.load_dataset(
        "google/fleurs", "el_gr", split="test", trust_remote_code=True
    )
    fleurs_wers_list, fleurs_stats = computeDataAndStats(
        dataset=fleurs,
        text_column_name="transcription",
        inferenceFunction=INFERENCE_FUNCTION,
        language=LANGUAGE,
    )
    saveOnDisk(data={"fleurs": fleurs_wers_list}, stats={"fleurs": fleurs_stats})

# ################################# ftspeech (danish only) #################################
# if "ftspeech" not in already_computed_datasets:
#     print("Testing ftspeech...")
#     ftspeech = datasets.load_dataset(
#         "alexandrainst/ftspeech", "default", split="test_balanced", trust_remote_code=True
#     )
#     ftspeech_wers_list, ftspeech_stats = computeDataAndStats(
#         dataset=ftspeech,
#         text_column_name="sentence",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"ftspeech": ftspeech_wers_list}, stats={"ftspeech": ftspeech_stats})

# ################################# nst-da (danish only) #################################
# if "nst_da" not in already_computed_datasets:
#     print("Testing nst-da...")
#     nst_da = datasets.load_dataset(
#         "alexandrainst/nst-da", "default", split="test", trust_remote_code=True
#     )
#     nst_da_wers_list, nst_da_stats = computeDataAndStats(
#         dataset=nst_da,
#         text_column_name="text",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"nst_da": nst_da_wers_list}, stats={"nst_da": nst_da_stats})

# ################################# masri_dev (maltese only) #################################
# if "masri_dev" not in already_computed_datasets:
#     print("Testing masri_dev...")
#     masri_dev = datasets.load_dataset(
#         "MLRS/masri_dev", split="validation"
#     )
#     masri_dev_wers_list, masri_dev_stats = computeDataAndStats(
#         dataset=masri_dev,
#         text_column_name="normalized_text",
#         inferenceFunction=INFERENCE_FUNCTION,
#         language=LANGUAGE,
#     )
#     saveOnDisk(data={"masri_dev": masri_dev_wers_list}, stats={"masri_dev": masri_dev_stats})
