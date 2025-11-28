from utils import asr_client

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
import datasets
import json
import numpy as np
import os
import dotenv

INFERENCE_FUNCTION = asr_client.inferenceFunction
OUTPUT_PATH = "results/cs/parakeet-tdt-0.6b-v3"

#
##
###
##
#


def computeWer(dataset, text_column_name, inferenceFunction):

    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")

    predictions_list = [inferenceFunction(sample["audio"]) for sample in dataset]
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


#
##
###
##
#

os.makedirs(OUTPUT_PATH)
output_data = {}
output_stats = {}
dotenv.load_dotenv(".env.secrets")

################################ Voxpopuli #################################
print("Testing Voxpopuli...")
voxpopuli = datasets.load_dataset(
    "facebook/voxpopuli", "cs", split="test", trust_remote_code=True
)  # italian: 1177 samples (too often with incorrect labels)
voxpopuli = voxpopuli.cast_column("audio", datasets.Audio(sampling_rate=16_000))
voxpopuli_wers_list, num_samples = computeWer(
    dataset=voxpopuli,
    text_column_name="raw_text",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(voxpopuli_wers_list)} [{num_samples} samples]")
output_data["voxpopuli"] = voxpopuli_wers_list
output_stats["voxpopuli"] = {
    "num_samples": num_samples,
    "mean": np.mean(voxpopuli_wers_list),
    "std": np.std(voxpopuli_wers_list),
    "min": np.min(voxpopuli_wers_list),
    "max": np.max(voxpopuli_wers_list),
}

################################# MLS #################################
# print("Testing MLS...")
# mls = datasets.load_dataset(
#     "facebook/multilingual_librispeech", "polish", split="test"
# )  # italian: 1260 samples
# mls = mls.cast_column("audio", datasets.Audio(sampling_rate=16_000))
# mls_wers_list, num_samples = computeWer(
#     dataset=mls,
#     text_column_name="transcript",
#     inferenceFunction=INFERENCE_FUNCTION,
# )
# print(f"WER = {np.mean(mls_wers_list)} [{num_samples} samples]")
# output_data["mls"] = mls_wers_list
# output_stats["mls"] = {
#     "num_samples": num_samples,
#     "mean": np.mean(mls_wers_list),
#     "std": np.std(mls_wers_list),
#     "min": np.min(mls_wers_list),
#     "max": np.max(mls_wers_list),
# }

################################# Common Voice 22.0 #################################
print("Testing CV-22.0...")
cv_22_0 = datasets.load_dataset(
    "fsicoli/common_voice_22_0",
    "cs",
    split="test",
    trust_remote_code=True,
    token=True,
)
cv_22_0 = cv_22_0.cast_column("audio", datasets.Audio(sampling_rate=16_000))
cv_22_0_wers_list, num_samples = computeWer(
    dataset=cv_22_0,
    text_column_name="sentence",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(cv_22_0_wers_list)} [{num_samples} samples]")
output_data["cv_22_0"] = cv_22_0_wers_list
output_stats["cv_22_0"] = {
    "num_samples": num_samples,
    "mean": np.mean(cv_22_0_wers_list),
    "std": np.std(cv_22_0_wers_list),
    "min": np.min(cv_22_0_wers_list),
    "max": np.max(cv_22_0_wers_list),
}

################################# Minds14 #################################
print("Testing Minds14...")
mind_14 = datasets.load_dataset(
    "PolyAI/minds14", "cs-CZ", split="train", trust_remote_code=True
)  # italian: (too often with incorrect labels)
mind_14 = mind_14.cast_column("audio", datasets.Audio(sampling_rate=16_000))
mind_14_wers_list, num_samples = computeWer(
    dataset=mind_14,
    text_column_name="transcription",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(mind_14_wers_list)} [{num_samples} samples]")
output_data["mind_14"] = mind_14_wers_list
output_stats["mind_14"] = {
    "num_samples": num_samples,
    "mean": np.mean(mind_14_wers_list),
    "std": np.std(mind_14_wers_list),
    "min": np.min(mind_14_wers_list),
    "max": np.max(mind_14_wers_list),
}

# ################################# Speech-MASSIVE-test #################################
# print("Testing Speech-MASSIVE-test...")
# speech_massive_test = datasets.load_dataset(
#     "FBK-MT/Speech-MASSIVE-test",
#     "ru-RU",
#     split="test",
#     trust_remote_code=True,
# )
# speech_massive_test = speech_massive_test.cast_column("audio", datasets.Audio(sampling_rate=16_000))
# speech_massive_test_wers_list, num_samples = computeWer(
#     dataset=speech_massive_test,
#     text_column_name="utt",
#     inferenceFunction=INFERENCE_FUNCTION,
# )
# print(f"WER = {np.mean(speech_massive_test_wers_list)} [{num_samples} samples]")
# output_data["speech_massive_test"] = speech_massive_test_wers_list
# output_stats["speech_massive_test"] = {
#     "num_samples": num_samples,
#     "mean": np.mean(speech_massive_test_wers_list),
#     "std": np.std(speech_massive_test_wers_list),
#     "min": np.min(speech_massive_test_wers_list),
#     "max": np.max(speech_massive_test_wers_list),
# }

# ################################# Romanian speech synthesis 0.8.1 #################################
# print("Testing Romanian speech synthesis 0.8.1...")
# romanian_speech_synthesis_0_8_1 = datasets.load_dataset(
#     "gigant/romanian_speech_synthesis_0_8_1", split="test", trust_remote_code=True
# )
# romanian_speech_synthesis_0_8_1 = romanian_speech_synthesis_0_8_1.cast_column(
#     "audio", datasets.Audio(sampling_rate=16_000)
# )
# romanian_speech_synthesis_0_8_1_wers_list, num_samples = computeWer(
#     dataset=romanian_speech_synthesis_0_8_1,
#     text_column_name="sentence",
#     inferenceFunction=INFERENCE_FUNCTION,
# )
# print(f"WER = {np.mean(romanian_speech_synthesis_0_8_1_wers_list)} [{num_samples} samples]")
# output_data["romanian_speech_synthesis_0_8_1"] = romanian_speech_synthesis_0_8_1_wers_list
# output_stats["romanian_speech_synthesis_0_8_1"] = {
#     "num_samples": num_samples,
#     "mean": np.mean(romanian_speech_synthesis_0_8_1_wers_list),
#     "std": np.std(romanian_speech_synthesis_0_8_1_wers_list),
#     "min": np.min(romanian_speech_synthesis_0_8_1_wers_list),
#     "max": np.max(romanian_speech_synthesis_0_8_1_wers_list),
# }

# ################################# Echo #################################
# print("Testing Echo...")
# echo = datasets.load_dataset(
#     "readerbench/echo", split="test", trust_remote_code=True
# )
# echo = echo.cast_column("audio", datasets.Audio(sampling_rate=16_000))
# echo_wers_list, num_samples = computeWer(
#     dataset=echo,
#     text_column_name="text",
#     inferenceFunction=INFERENCE_FUNCTION,
# )
# print(f"WER = {np.mean(echo_wers_list)} [{num_samples} samples]")
# output_data["echo"] = echo_wers_list
# output_stats["echo"] = {
#     "num_samples": num_samples,
#     "mean": np.mean(echo_wers_list),
#     "std": np.std(echo_wers_list),
#     "min": np.min(echo_wers_list),
#     "max": np.max(echo_wers_list),
# }

#
##
### Save
##
#

with open(f"{OUTPUT_PATH}/data.json", "w") as f:
    json.dump(output_data, f)
with open(f"{OUTPUT_PATH}/stats.json", "w") as f:
    json.dump(output_stats, f)
