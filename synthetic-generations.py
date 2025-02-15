import json
import random
import fire
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from instructmultilingual.translate_datasets import translate_dataset_split_via_inference_api
from concurrent.futures import ThreadPoolExecutor, as_completed

##################################
## SERVER + GENERATION SETTINGS
##################################

NUM_GENERATIONS_BY_LANGUAGE = {
    "pes_Arab": 10000,
    "jpn_Jpan": 10000,
    "vie_Latn": 10000,
    "ron_Latn": 10000,
    "ind_Latn": 10000,
    "tur_Latn": 10000,
    "spa_Latn": 10000,
    "ces_Latn": 10000,
    "pol_Latn": 10000,
    "por_Latn": 10000,
    "nld_Latn": 10000,
    "kor_Hang": 10000,
    "fra_Latn": 10000,
    "ukr_Cyrl": 10000,
    "hin_Deva": 10000,
    "heb_Hebr": 10000,
    "ell_Grek": 10000,
    "ita_Latn": 10000,
    "zho_Hans": 10000,
    "rus_Cyrl": 10000,
    "arb_Arab": 10000,
    "zho_Hant": 10000,
    "deu_Latn": 10000
}

SERVERS = [f"http://localhost:{8000 + i}/translate" for i in range(64)]  # 64 servers

##################################
## CHUNKING + TRANSLATION WRAPPER
##################################

def process_chunk(language, chunk, server, columns_to_translate):
    """
    Processes a single chunk of data using the inference APIs
    """
    # The API expects the dataset in this format
    dataset = DatasetDict({"sampled": Dataset.from_list(chunk)})
    
    translated_chunk = translate_dataset_split_via_inference_api(
        dataset=dataset,
        split="sampled",
        translate_keys=columns_to_translate,
        url=server,
        source_language_code="eng_Latn",
        target_language_code=language,
    )
    return translated_chunk

def process_language(language, sampled_examples, columns_to_translate):
    """
    Process a single language by splitting its dataset into chunks and assigning them to servers.
    """
    # We want to process each dataset across all the servers
    chunks = [c.tolist() for c in np.array_split(sampled_examples, len(SERVERS))]
    futures = []
    translated_data = []

    # Submit tasks for all servers
    with ThreadPoolExecutor(max_workers=len(SERVERS)) as executor:
        for i, server in enumerate(SERVERS):
            futures.append(executor.submit(process_chunk, language, chunks[i], server, columns_to_translate))

        # Collect results as chunks are completed
        for future in as_completed(futures):
            try:
                translated_data.extend(future.result())
            except Exception as e:
                print(f"Error processing chunk for language {language}: {e}")

    print(f"Finished processing {language}. Total examples processed: {len(translated_data)}")
    return translated_data


def generate_synthetic_data(source_prompts_path, output_dir, columns_to_translate=["prompt"]):
    """Generates multilingual synthetic data
    Specify the language codes and desired amounts in the NUM_GENERATIONS_BY_LANGUAGE config
    This function will sample from the specified dataset in the input path,
        and translate the specified column to the desired language

    Args:
        source_prompts_path (str): Path to the dataset to sample from
        output_dir (str): Directory to output the results
        columns_to_translate (List[str]): List of columns to translate

    Returns:
        List[str]: List of translated text
    """
    with open(source_prompts_path, "r", encoding="utf-8") as f:
        english_example_pool = []
        for line in f:
            example = json.loads(line)
            example['source_prompt'] = example['prompt']
            english_example_pool.append(example)

    for language, num_examples in NUM_GENERATIONS_BY_LANGUAGE.items():
        sampled_examples = [random.choice(english_example_pool) for _ in range(num_examples)]

        print(f'Currently synthesizing {language} data.')
        translation_results = process_language(language, sampled_examples, columns_to_translate)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/{language}_translations.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in translation_results:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Saved final results for {language} to {output_path}\n\n")

if __name__ == "__main__":
    fire.Fire(generate_synthetic_data)
