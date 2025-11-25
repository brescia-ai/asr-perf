from utils import asr_client

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
import datasets
import json
import numpy as np
import os
import dotenv

INFERENCE_FUNCTION = asr_client.inferenceFunction
NUM_SAMPLES = 450  # samples per dataset
OUTPUT_PATH = "results/es/parakeet-tdt-0.6b-v3"

#
##
###
##
#


def computeWer(dataset, num_samples, text_column_name, inferenceFunction):

    dataset = dataset.select(range(num_samples))  # select the first num_samples

    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")

    predictions_list = [inferenceFunction(sample["audio"]) for sample in dataset]
    references_list = dataset[text_column_name]

    norm_references_list = [normalizer(reference) for reference in references_list]
    norm_predictions_list = [normalizer(prediction) for prediction in predictions_list]

    wers_list = []
    for i in range(num_samples):
        ref = norm_references_list[i]
        pred = norm_predictions_list[i]

        # Handle empty strings to avoid division by zero
        if not ref.strip() and not pred.strip():
            wers_list.append(0.0)  # Both empty = perfect match
        elif not ref.strip():
            wers_list.append(1.0)  # Empty reference, non-empty prediction = 100% error
        else:
            wers_list.append(wer_metric.compute(references=[ref], predictions=[pred]))

    return wers_list


#
##
###
##
#

os.makedirs(OUTPUT_PATH)
output_data = {}
output_stats = {"samples_per_dataset": NUM_SAMPLES}
dotenv.load_dotenv(".env.secrets")

################################# Voxpopuli #################################
print("Testing Voxpopuli...")
voxpopuli = datasets.load_dataset(
    "facebook/voxpopuli", "es", split="test", trust_remote_code=True
)  # 1177 samples (too often with incorrect labels)
voxpopuli = voxpopuli.cast_column("audio", datasets.Audio(sampling_rate=16_000))
voxpopuli_wers_list = computeWer(
    dataset=voxpopuli,
    num_samples=NUM_SAMPLES,
    text_column_name="raw_text",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(voxpopuli_wers_list)}")
output_data["voxpopuli"] = voxpopuli_wers_list
output_stats["voxpopuli"] = {
    "mean": np.mean(voxpopuli_wers_list),
    "std": np.std(voxpopuli_wers_list),
    "min": np.min(voxpopuli_wers_list),
    "max": np.max(voxpopuli_wers_list),
}

################################# MLS #################################
print("Testing MLS...")
mls = datasets.load_dataset(
    "facebook/multilingual_librispeech", "spanish", split="test"
)  # 1260 samples
mls = mls.cast_column("audio", datasets.Audio(sampling_rate=16_000))
mls_wers_list = computeWer(
    dataset=mls,
    num_samples=NUM_SAMPLES,
    text_column_name="transcript",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(mls_wers_list)}")
output_data["mls"] = mls_wers_list
output_stats["mls"] = {
    "mean": np.mean(mls_wers_list),
    "std": np.std(mls_wers_list),
    "min": np.min(mls_wers_list),
    "max": np.max(mls_wers_list),
}

################################# CV-17 #################################
print("Testing CV-17...")
cv_17 = datasets.load_dataset(
    "fsicoli/common_voice_17_0",
    "es",
    split="test",
    trust_remote_code=True,
    token=True,
)
cv_17 = cv_17.cast_column("audio", datasets.Audio(sampling_rate=16_000))
cv_17_wers_list = computeWer(
    dataset=cv_17,
    num_samples=NUM_SAMPLES,
    text_column_name="sentence",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(cv_17_wers_list)}")
output_data["cv_17"] = cv_17_wers_list
output_stats["cv_17"] = {
    "mean": np.mean(cv_17_wers_list),
    "std": np.std(cv_17_wers_list),
    "min": np.min(cv_17_wers_list),
    "max": np.max(cv_17_wers_list),
}

################################# Minds14 #################################
print("Testing Minds14...")
mind_14 = datasets.load_dataset(
    "PolyAI/minds14", "es-ES", split="train", trust_remote_code=True
)  # (too often with incorrect labels)
mind_14 = mind_14.cast_column("audio", datasets.Audio(sampling_rate=16_000))
mind_14_wers_list = computeWer(
    dataset=mind_14,
    num_samples=NUM_SAMPLES,
    text_column_name="transcription",
    inferenceFunction=INFERENCE_FUNCTION,
)
print(f"WER = {np.mean(mind_14_wers_list)}")
output_data["mind_14"] = mind_14_wers_list
output_stats["mind_14"] = {
    "mean": np.mean(mind_14_wers_list),
    "std": np.std(mind_14_wers_list),
    "min": np.min(mind_14_wers_list),
    "max": np.max(mind_14_wers_list),
}

#
##
### Save
##
#

with open(f"{OUTPUT_PATH}/data.json", "w") as f:
    json.dump(output_data, f)
with open(f"{OUTPUT_PATH}/stats.json", "w") as f:
    json.dump(output_stats, f)
