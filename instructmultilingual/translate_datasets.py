"""Translate datasets from huggingface hub using a variety of methods."""

import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import requests
from google.cloud import translate_v2
from huggingface_hub import hf_hub_download
from sentence_splitter import split_text_into_sentences

from datasets import DatasetDict, load_dataset
from instructmultilingual.cloud_translate_mapping import (cloud_translate_lang_code_to_name,
                                                          cloud_translate_lang_name_to_code)
from instructmultilingual.flores_200 import (lang_code_to_name, lang_name_to_code)

T5_LANG_CODES = [
    'afr_Latn', 'als_Latn', 'amh_Ethi', 'ace_Arab', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'ajp_Arab', 'apc_Arab',
    'arb_Arab', 'arb_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'bjn_Arab', 'kas_Arab', 'knc_Arab', 'min_Arab',
    'hye_Armn', 'azb_Arab', 'azj_Latn', 'eus_Latn', 'bel_Cyrl', 'ben_Beng', 'mni_Beng', 'bul_Cyrl', 'mya_Mymr',
    'cat_Latn', 'ceb_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'ces_Latn', 'dan_Latn', 'nld_Latn', 'eng_Latn',
    'epo_Latn', 'est_Latn', 'fin_Latn', 'fra_Latn', 'glg_Latn', 'kat_Geor', 'deu_Latn', 'ell_Grek', 'guj_Gujr',
    'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hun_Latn', 'isl_Latn', 'ibo_Latn', 'ind_Latn', 'gle_Latn',
    'ita_Latn', 'jpn_Jpan', 'jav_Latn', 'kan_Knda', 'kaz_Cyrl', 'khm_Khmr', 'kor_Hang', 'ckb_Arab', 'kmr_Latn',
    'kir_Cyrl', 'lao_Laoo', 'ace_Latn', 'bjn_Latn', 'knc_Latn', 'min_Latn', 'taq_Latn', 'lvs_Latn', 'lit_Latn',
    'ltz_Latn', 'mkd_Cyrl', 'plt_Latn', 'mal_Mlym', 'zsm_Latn', 'mal_Mlym', 'mlt_Latn', 'mri_Latn', 'mar_Deva',
    'khk_Cyrl', 'npi_Deva', 'nno_Latn', 'nob_Latn', 'pbt_Arab', 'pes_Arab', 'pol_Latn', 'por_Latn', 'ron_Latn',
    'rus_Cyrl', 'smo_Latn', 'gla_Latn', 'srp_Cyrl', 'sna_Latn', 'snd_Arab', 'sin_Sinh', 'slk_Latn', 'slv_Latn',
    'som_Latn', 'nso_Latn', 'sot_Latn', 'spa_Latn', 'sun_Latn', 'swh_Latn', 'swe_Latn', 'tgk_Cyrl', 'tam_Taml',
    'tel_Telu', 'tha_Thai', 'tur_Latn', 'ukr_Cyrl', 'urd_Arab', 'uzn_Latn', 'vie_Latn', 'cym_Latn', 'xho_Latn',
    'ydd_Hebr', 'yor_Latn', 'zul_Latn'
]

T5_CLOUD_TRANSLATE_LANG_CODES = [
    'eu', 'st', 'lo', 'fi', 'co', 'ro', 'sd', 'sv', 'ta', 'kn', 'gd', 'et', 'gl', 'fy', 'ru', 'mr', 'zu', 'ky', 'da',
    'sr', 'haw', 'hi', 'gu', 'su', 'tr', 'bn', 'hu', 'hy', 'jv', 'pa', 'de', 'la', 'uz', 'lt', 'no', 'xh', 'mk', 'ms',
    'ur', 'ar', 'am', 'vi', 'it', 'cy', 'en', 'eo', 'be', 'id', 'my', 'is', 'nl', 'sn', 'sm', 'so', 'ha', 'mi', 'th',
    'kk', 'ml', 'hmn', 'uk', 'ga', 'lb', 'zh-CN', 'zh-TW', 'mt', 'fr', 'ku', 'ht', 'sw', 'sk', 'km', 'si', 'ceb', 'tg',
    'cs', 'pl', 'ig', 'sl', 'ka', 'ca', 'mn', 'yo', 'fa', 'es', 'iw', 'bg', 'af', 'el', 'ps', 'mg', 'yi', 'pt', 'ja',
    'ny', 'ko', 'lv', 'te', 'sq', 'ne', 'az'
]


def inference_request(url: str, source_language: str, target_language: str, texts: List[str]) -> List[str]:
    """A HTTP POST request to the inference server for translation.

    Args:
        url (str): The URL of the inference API server
        source_language (str): Language code for the source language
        target_language (str): Language code for the target language
        texts (List[str]): List of texts to be translated

    Returns:
        List[str]: List of translated text
    """

    headers = {"Content-Type": "application/json"}
    data = {
        "source_language": source_language,
        "target_language": target_language,
        "texts": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["translated_texts"]

def mock_inference_request(url: str, source_language: str, target_language: str, texts: List[str]) -> List[str]:
    """A mock function to simulate an inference request for translation.

    Args:
        url (str): The URL of the inference API server (not used in mock)
        source_language (str): Language code for the source language
        target_language (str): Language code for the target language
        texts (List[str]): List of texts to be translated

    Returns:
        List[str]: List of translated text (lorem ipsum)
    """
    lorem_ipsum = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    )
    return [lorem_ipsum for _ in texts]

def call_inference_api(
    example: Dict[str, List[str]],
    url: str,
    source_lang_code: str,
    target_lang_code: str,
    keys_to_be_translated: List[str] = ["dialogue", "summary"],
) -> Dict[str, List[str]]:
    """Calls the inference server key-by-key and returns the translated
    examples back.

    Args:
        example (Dict[str,List[str]]): A batch of inputs from the dataset for translation. Keys are the column names, values are the batch of text inputs
        url (str): The URL of the inference API server.
        source_lang_code (str): Language code for the source language
        target_lang_code (str): Language code for the target language
        keys_to_be_translated (List[str], optional): The keys/columns for the texts you want translated. Defaults to ["dialogue", "summary"].

    Returns:
        Dict[str,List[str]]: Translated outputs based on the example Dict
    """
    for key in keys_to_be_translated:
        # NLLB model seems to ignore some sentences right before newline characters
        batch_str = [sen.replace('\n', '') for sen in example[key]]
        example[key] = inference_request(url, source_lang_code, target_lang_code, batch_str)
    return example


def translate_sent_by_sent(
    example: Dict[str, List[str]],
    url: str,
    source_lang_code: str,
    target_lang_code: str,
    keys_to_be_translated: List[str] = ["dialogue", "summary"],
) -> Dict[str, List[str]]:
    """A wrapper for call_inference_api that preprocess the input text by
    breaking them into sentences.

    Args:
        example (Dict[str,List[str]]): A batch of inputs from the dataset for translation. Keys are the column names, values are the batch of text inputs
        url (str): The URL of the inference API server.
        source_lang_code (str): Language code for the source language
        target_lang_code (str): Language code for the target language
        keys_to_be_translated (List[str], optional): The keys/columns for the texts you want translated. Defaults to ["dialogue", "summary"].

    Returns:
        Dict[str,List[str]]: Translated outputs based on the example Dict
    """
    from collections import defaultdict

    for k in example.keys():
        num_inputs = len(example[k])
        break

    sentences = []
    sentenized_example = defaultdict(list)

    for k in keys_to_be_translated:
        sentenized_example[f"{k}_pos"].append(0)

    for i in range(num_inputs):

        for k in keys_to_be_translated:
            sentences = split_text_into_sentences(text=example[k][i], language='en')
            sentenized_example[k].extend(sentences)
            sentenized_example[f"{k}_pos"].append(sentenized_example[f"{k}_pos"][-1] + len(sentences))

    result = call_inference_api(example=sentenized_example,
                                url=url,
                                keys_to_be_translated=keys_to_be_translated,
                                source_lang_code=source_lang_code,
                                target_lang_code=target_lang_code)

    for k in keys_to_be_translated:
        merged_texts = []
        l = 0
        r = 1
        while r < len(result[f"{k}_pos"]):
            start = result[f"{k}_pos"][l]
            end = result[f"{k}_pos"][r]
            merged_texts.append(' '.join(result[k][start:end]))
            l += 1
            r += 1
        example[k] = merged_texts
    return example

def translate_dataset_split_via_inference_api(
    dataset: DatasetDict,
    split: str,
    translate_keys: List[str],
    target_language_code: str,
    source_language_code: str = "eng_Latn",
    url: str = "http://localhost:8000/translate",
    num_proc: int = 1,
) -> None:
    """This function takes an DatasetDict object and translates it via the
    translation inference server API. The function then returns the result

    Args:
        dataset (DatasetDict): A DatasetDict object of the original text dataset. Needs to have at least one split.
        split (str): Split name in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        target_language_code (str): the language code you want translation to.
        source_language_code (str, optional): Languague of the original text. Defaults to "eng_Latn".
        url (str, optional): The URL of the inference API server. Defaults to "http://localhost:8000/translate".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 1.
    """
    start_time = time.time()
    ds = dataset[split]
    # print(f"[{split}] {len(ds)=}")
    ds = ds.map(
        lambda x: translate_sent_by_sent(
            x,
            url=url,
            source_lang_code=source_language_code,
            target_lang_code=target_language_code,
            keys_to_be_translated=translate_keys,
        ),
        batched=True,
        num_proc=num_proc,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return ds

def translate_dataset_via_inference_api(
    dataset: DatasetDict,
    dataset_name: str,
    template_name: str,
    splits: List[str],
    translate_keys: List[str],
    target_language: str,
    url: str = "http://localhost:8000/translate",
    output_dir: str = "./datasets",
    source_language: str = "English",
    checkpoint: str = "facebook/nllb-200-3.3B",
    num_proc: int = 1,
) -> None:
    """This function takes an DatasetDict object and translates it via the
    translation inference server API. The function then ouputs the translations
    in both json and csv formats into a output directory under the following
    naming convention:

       <output_dir>/<dataset_name>/<source_language_code>_to_<target_language_code>/<checkpoint>/<template_name>/<date>/<split>.<file_type>

    Args:
        dataset (DatasetDict): A DatasetDict object of the original text dataset. Needs to have at least one split.
        dataset_name (str): Name of the dataset for storing output.
        template_name (str): Name of the template for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        target_language (str): the language you want translation to.
        url (str, optional): The URL of the inference API server. Defaults to "http://localhost:8000/translate".
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 1.
    """

    date = datetime.today().strftime('%Y-%m-%d')

    source_language_code = lang_name_to_code[source_language]
    target_language_code = lang_name_to_code[target_language]

    checkpoint_str = checkpoint.replace("/", "-")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for split in splits:
        split_time = time.time()
        ds = dataset[split]
        print(f"[{split}] {len(ds)=}")
        ds = ds.map(
            lambda x: translate_sent_by_sent(
                x,
                url=url,
                source_lang_code=source_language_code,
                target_lang_code=target_language_code,
                keys_to_be_translated=translate_keys,
            ),
            batched=True,
            num_proc=num_proc,
        )
        print(f"[{split}] One example translated {ds[0]=}")
        print(f"[{split}] took {time.time() - split_time:.4f} seconds")

        translation_path = os.path.join(output_dir, dataset_name, f"{source_language_code}_to_{target_language_code}",
                                        checkpoint_str, template_name, date)
        Path(translation_path).mkdir(exist_ok=True, parents=True)

        ds.to_csv(
            os.path.join(
                translation_path,
                f"{split}.csv",
            ),
            index=False,
        )
        ds.to_json(os.path.join(
            translation_path,
            f"{split}.jsonl",
        ))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")


def cloud_translate(example: Dict[str, str],
                    target_lang_code: str,
                    keys_to_be_translated: List[str],
                    max_tries: int = 5) -> Dict[str, str]:
    """Translates the input example batch of texts using Google Cloud Translate
    API.

    Args:
        example (Dict[str, str]): A batch of inputs from the dataset for translation. Keys are the column names, values are the batch of text inputs
        target_lang_code (str): Language code for the target language
        keys_to_be_translated (List[str]): The keys/columns for the texts you want translated.
        max_tries (int, optional): _description_. Defaults to 5.

    Returns:
        Dict[str, str]: Translated outputs based on the example Dict
    """
    translate_client = translate_v2.Client()

    tries = 0

    while tries < max_tries:
        tries += 1
        try:
            for key in keys_to_be_translated:
                results = translate_client.translate(example[key], target_language=target_lang_code)
                example[key] = [result["translatedText"] for result in results]
                time.sleep(random.uniform(0.8, 1.5))
        except Exception as e:
            print(e)
            time.sleep(random.uniform(2, 5))

    return example


def translate_dataset_via_cloud_translate(
    dataset: DatasetDict,
    dataset_name: str,
    template_name: str,
    splits: List[str],
    translate_keys: List[str],
    target_language: str,
    output_dir: str = "./datasets",
    source_language: str = "English",
    checkpoint: str = "google_cloud_translate",
    num_proc: int = 1,
) -> None:
    """This function takes an DatasetDict object and translates it via the
    translation inference server API. The function then ouputs the translations
    in both json and csv formats into a output directory under the following
    naming convention:

       <output_dir>/<dataset_name>/<source_language_code>_to_<target_language_code>/<checkpoint>/<template_name>/<date>/<split>.<file_type>

    Args:
        dataset (DatasetDict): A DatasetDict object of the original text dataset. Needs to have at least one split.
        dataset_name (str): Name of the dataset for storing output.
        template_name (str): Name of the template for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        target_language (str): the language you want translation to.
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 1.
    """

    date = datetime.today().strftime('%Y-%m-%d')

    source_language_code = cloud_translate_lang_name_to_code[source_language]
    target_language_code = cloud_translate_lang_name_to_code[target_language]

    checkpoint_str = checkpoint.replace("/", "-")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for split in splits:
        split_time = time.time()
        ds = dataset[split]
        print(f"[{split}] {len(ds)=}")
        ds = ds.map(
            lambda x: cloud_translate(
                x,
                target_lang_code=target_language_code,
                keys_to_be_translated=translate_keys,
            ),
            batched=True,
            batch_size=40,  # translate api has limit of 204800 bytes max at a time
            num_proc=num_proc,
        )
        print(f"[{split}] One example translated {ds[0]=}")
        print(f"[{split}] took {time.time() - split_time:.4f} seconds")

        translation_path = os.path.join(output_dir, dataset_name, f"{source_language_code}_to_{target_language_code}",
                                        checkpoint_str, template_name, date)
        Path(translation_path).mkdir(exist_ok=True, parents=True)

        ds.to_csv(
            os.path.join(
                translation_path,
                f"{split}.csv",
            ),
            index=False,
        )
        ds.to_json(os.path.join(
            translation_path,
            f"{split}.jsonl",
        ))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")


def translate_dataset_from_huggingface_hub(dataset_name: str,
                                           template_name: str,
                                           splits: List[str],
                                           translate_keys: List[str],
                                           repo_id: str = "bigscience/xP3",
                                           train_set: List[str] = [],
                                           validation_set: List[str] = [],
                                           test_set: List[str] = [],
                                           url: str = "http://localhost:8000/translate",
                                           output_dir: str = "./datasets",
                                           source_language: str = "English",
                                           checkpoint: str = "facebook/nllb-200-3.3B",
                                           num_proc: int = 1,
                                           file_ext: str = "json",
                                           translation_lang_codes: List[str] = T5_LANG_CODES,
                                           exclude_languages: Set[str] = {"English"},
                                           from_local: bool = False) -> None:
    """A wrapper for using translate_dataset_via_api specifically on dataset
    repos from HuggingFace hub. The default repo is bigscience/xP3.

    Args:
        dataset_name (str): Name of the dataset for storing output.
        template_name (str): Name of the template for storing output.
        splits (List[str]): Split names in the dataset you want translated.
        translate_keys (List[str]): The keys/columns for the texts you want translated.
        repo_id (str, optional): Name of the dataset repo on Huggingface. Defaults to "bigscience/xP3".
        train_set (List[str], optional): List of training set jsonl files for the dataset. Defaults to [].
        validation_set (List[str], optional): List of validation set jsonl files for the dataset. Defaults to [].
        test_set (List[str], optional): List of test set jsonl files for the dataset. Defaults to [].
        url (str, optional): The URL of the inference API server. Defaults to "http://localhost:8000/translate".
        output_dir (str, optional): Root directory of all datasets. Defaults to "./datasets".
        source_language (str, optional): Languague of the original text. Defaults to "English".
        checkpoint (str, optional): Name of the checkpoint used for naming. Defaults to "facebook/nllb-200-3.3B".
        num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to 1.
        file_ext (str, optional): file extension for the downloaded dataset files. Defaults to "json".
        translation_lang_codes (List[str], optional): List of Flores-200 language codes to translate to. Defaults to T5_LANG_CODES.
        exclude_languages (Set[str], optional): Set of languages to exclude. Defaults to {"English"}.
        from_local: (bool, optional): Load source files from local instead of HuggingFace. Defaults to False.
    """
    assert len(train_set) > 0 or len(validation_set) > 0 or len(
        test_set) > 0, "Error: one of train/validation/test sets has to have a path"

    dataset_splits = {"train": train_set, "validation": validation_set, "test": test_set}

    dataset_template = defaultdict(list)

    temp_root = "temp_datasets"
    temp_dir = f"{temp_root}/{dataset_name}"
    Path(temp_dir).mkdir(exist_ok=True, parents=True)
    
    if not from_local:
        for split, files in dataset_splits.items():
            if len(files) > 0:
                temp_split_dir = f"{temp_root}/{dataset_name}/{split}"
                Path(temp_split_dir).mkdir(exist_ok=True, parents=True)
                for f in files:

                    hf_hub_download(repo_id=repo_id,
                                    local_dir=temp_split_dir,
                                    filename=f,
                                    repo_type="dataset",
                                    local_dir_use_symlinks=False)

                    pth = os.path.join(temp_split_dir, f)
                    dataset_template[split].append(pth)
    else:
        for split, files in dataset_splits.items():
            if len(files) > 0:
                for f in files:
                    dataset_template[split].append(f)


    dataset = load_dataset(file_ext, data_files=dataset_template)
    columns_to_remove = set()
    for split in dataset.column_names:
        for col in set(dataset[split].column_names) - set(translate_keys):
            columns_to_remove.add(col)

    columns_to_remove = list(columns_to_remove)
    dataset = dataset.remove_columns(columns_to_remove)

    # Make a copy of the source dataset inside translated datasets as well
    date = datetime.today().strftime('%Y-%m-%d')
    if checkpoint == "google_cloud_translate":
        source_language_code = cloud_translate_lang_name_to_code[source_language]
    else:
        source_language_code = lang_name_to_code[source_language]
    checkpoint_str = checkpoint.replace("/", "-")
    translation_path = os.path.join(output_dir, dataset_name, f"{source_language_code}_to_{source_language_code}",
                                    checkpoint_str, template_name, date)
    Path(translation_path).mkdir(exist_ok=True, parents=True)
    for s in dataset.keys():
        dataset[s].to_csv(
            os.path.join(
                translation_path,
                f"{s}.csv",
            ),
            index=False,
        )
        dataset[s].to_json(os.path.join(
            translation_path,
            f"{s}.jsonl",
        ))

    if checkpoint == "google_cloud_translate":
        for code in translation_lang_codes:
            l = cloud_translate_lang_code_to_name[code]
            if l not in exclude_languages:
                print(f"Currently translating: {l}")
                translate_dataset_via_cloud_translate(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    splits=splits,
                    translate_keys=translate_keys,
                    target_language=l,
                    output_dir=output_dir,
                    source_language=source_language,
                    checkpoint=checkpoint,
                    num_proc=num_proc,
                )
    else:
        for code in translation_lang_codes:
            l = lang_code_to_name[code]
            if l not in exclude_languages:
                print(f"Currently translating: {l}")
                translate_dataset_via_inference_api(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    template_name=template_name,
                    splits=splits,
                    translate_keys=translate_keys,
                    target_language=l,
                    url=url,
                    output_dir=output_dir,
                    source_language=source_language,
                    checkpoint=checkpoint,
                    num_proc=num_proc,
                )
