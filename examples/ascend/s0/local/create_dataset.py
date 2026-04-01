import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, concatenate_datasets, DatasetDict

raw_dataset = load_dataset("CAiRE/ASCEND", cache_dir="downloads/cache/")
ascend_asr = DatasetDict()
ascend_asr["train_zh"] = raw_dataset["train"].filter(lambda x: x["language"] == "zh")
ascend_asr["train_en"] = raw_dataset["train"].filter(lambda x: x["language"] == "en")
ascend_asr["train_mixed"] = raw_dataset["train"].filter(lambda x: x["language"] == "mixed")
ascend_asr["train"] = concatenate_datasets(
    [
        ascend_asr["train_zh"],
        ascend_asr["train_en"],
    ]
)
ascend_asr["validation"] = raw_dataset["validation"]
ascend_asr["test"] = raw_dataset["test"]

os.makedirs("storage", exist_ok=True)
for split in raw_dataset.keys():
    for example in raw_dataset[split]:
        audio_array = example["audio"]["array"]
        sample_rate = example["audio"]["sampling_rate"]
        filename = os.path.basename(example["path"])

        path = os.path.join("storage", filename)
        if not os.path.exists(path):
            sf.write(path, audio_array, sample_rate)

"""
kaldi data validation fails on certain white space characters, those are replaced here.
See https://apps.timwhitlock.info/unicode/inspect/hex/2000-206F
"""


def replace_bad_spaces(sample):
    sentence = sample["transcription"]

    sentence = sentence.strip()
    for i in range(8192, 8208):
        sentence = sentence.replace(chr(i), " ")
    for i in range(8232, 8240):
        sentence = sentence.replace(chr(i), " ")
    sentence = sentence.replace(chr(160), " ")

    sample["transcription"] = sentence

    return sample


def create_csv(split):
    ascend_asr[split] = ascend_asr[split].map(replace_bad_spaces)

    paths = [
        os.path.join("storage", os.path.basename(p)) for p in ascend_asr[split]["path"]
    ]
    transcriptions = ascend_asr[split]["transcription"]
    langs = ascend_asr[split]["language"]
    speaker_ids = ascend_asr[split]["original_speaker_id"]

    df = pd.DataFrame(
        data={
            "speaker_id": speaker_ids,
            "path": paths,
            "sentence": transcriptions,
            "accent": langs,
        }
    )

    df.to_csv(f"downloads/{split}.tsv", index=False, sep="\t")


create_csv("train_zh")
create_csv("train_en")
create_csv("train_mixed")
create_csv("train")
create_csv("validation")
create_csv("test")
