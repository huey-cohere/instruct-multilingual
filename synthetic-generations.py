import json
import random
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from instructmultilingual.translate_datasets import translate_dataset_split_via_inference_api
from concurrent.futures import ThreadPoolExecutor, as_completed

##################################
## SERVER + GENERATION SETTINGS
##################################

NUM_GENERATIONS_BY_LANGUAGE = {
    "pes_Arab": 512,
    "jpn_Jpan": 512,
    "vie_Latn": 512,
    "ron_Latn": 512,
    "ind_Latn": 512,
    "tur_Latn": 512,
    "spa_Latn": 512,
    "ces_Latn": 512,
    "pol_Latn": 512,
    "por_Latn": 512,
    "nld_Latn": 512,
    "kor_Hang": 512,
    "fra_Latn": 512,
    "ukr_Cyrl": 512,
    "hin_Deva": 512,
    "heb_Hebr": 512,
    "ell_Grek": 512,
    "ita_Latn": 512,
    "zho_Hans": 512,
    "rus_Cyrl": 512,
    "arb_Arab": 512,
    "zho_Hant": 512,
    "deu_Latn": 512
}

SERVERS = [f"http://localhost:{8000 + i}/translate" for i in range(64)]  # 64 servers

##################################
## CHUNKING + TRANSLATION WRAPPER
##################################

def process_chunk(language, chunk, server):
    """
    Simulate processing a single chunk on a specific server.
    Replace this with actual translation logic.
    """
    # Only want to translate the prompt column
    columns_to_translate = ["prompt"]
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


def process_language(language, sampled_examples):
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
            futures.append(executor.submit(process_chunk, language, chunks[i], server))

        # Collect results as chunks are completed
        for future in as_completed(futures):
            # try:
            translated_data.extend(future.result())
            # except Exception as e:
            #     print(f"Error processing chunk for language {language}: {e}")

    print(f"Finished processing {language}. Total examples processed: {len(translated_data)}")
    return translated_data


def generate_synthetic_data(source_path, output_dir):
    with open(source_path, "r", encoding="utf-8") as f:
        english_example_pool = [json.loads(line) for line in f]

    for language, num_examples in NUM_GENERATIONS_BY_LANGUAGE.items():
        sampled_examples = [random.choice(english_example_pool) for _ in range(num_examples)]

        print(f'Currently synthesizing {language} data.')
        translation_results = process_language(language, sampled_examples)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/{language}_translations.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for example in translation_results:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Saved final results for {language} to {output_path}")


if __name__ == "__main__":
    source_path = './synth-test.jsonl'
    output_dir = './huey-translations'
    generate_synthetic_data(source_path, output_dir)
