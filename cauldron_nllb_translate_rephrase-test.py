import cohere

client = cohere.ClientV2("MfZwS1plvJfM7vARPs92RbCScEwRniTcCXfmfAdU", base_url="https://stg.api.cohere.ai")

PROMPT = """
Original Sentence: {raw_sentene}\n
Translated Sentence: {translated_sentene}\n

Rephrase the translated sentence to enhance its quality, ensuring it aligns closely with the original in meaning, structure, tone, and style.
Ensure that the rephrased sentence conveys the same meaning as the original sentence but avoid altering the core message or introducing new information. 
Correct any grammatical errors present in the translated sentence.
Maintain a structure similar to the original sentence. 
Match the tone and style of the original sentence. 
Preserve any stylistic elements such as enumeration, punctuation or capitalization.

The output must strictly follow this format:\n
Rephrase Translated Sentence: <sentence>"""

import json
import os
import random
key = "RecapCauldronTextcaps_translation"

eng_Latn = []
with open(f"/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/{key}/eng_Latn/train.jsonl", "r") as f:
    i = 0
    for line in f:
        eng_Latn.append(json.loads(line))
        i += 1
        if i == 10:
            break
    # eng_Latn = [json.loads(line) for line in f]

# eng_Latn_sample = random.sample(eng_Latn, 10)


samples = []
for sample in eng_Latn_sample:
    example = {"id": sample["command_id"], "User":[], "Chatbot":[]}
    for turn in sample["turns"]:
        if turn["role"] == "User":
            for content in turn["content"]:
                if 'text' in content:
                    example["User"].append({"text": content["text"], "language": "eng_Latn", "source": "raw"})
        if turn["role"] == "Chatbot":
            example["Chatbot"].append({"text": turn["content"], "language": "eng_Latn", "source": "raw-gpt_recap"})
    samples.append(example)


samples_dict = {sample["id"]: sample for sample in samples}


for lang in os.listdir(f"/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/{key}/"):
    if lang == "eng_Latn":
        continue
    with open(f"/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/{key}/{lang}/train.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if data['command_id'] in samples_dict:
                for turn in data["turns"]:
                    if turn["role"] == "User":
                        for content in turn["content"]:
                            if 'text' in content:
                                samples_dict[data['command_id']]['User'].append({"text": content["text"], "language": lang, "source": "raw-nllb_transl"})
                    if turn["role"] == "Chatbot":
                        samples_dict[data['command_id']]["Chatbot"].append({"text": turn["content"], "language": lang, "source": "raw-gpt_recap-nllb_transl"})

samples = list(samples_dict.values())

import re
dataset = []
from tqdm import tqdm
def match(output):
    return re.search(r'Rephrase Translated Sentence:\s*(.+)', output)

for sample in samples:
    for user in sample['User']:
        if user["language"] == "eng_Latn":
            eng_Latn_user = user["text"]
            break
    for chatbot in sample['Chatbot']:
        if chatbot["language"] == "eng_Latn":
            eng_Latn_chatbot = chatbot["text"]
            break
    
    
    for i, user in tqdm(enumerate(sample['User'])):
        example = {}
        example['index'] = i
        language = user["language"]
        if language != "eng_Latn":
            
            formatted_prompt = PROMPT.format(raw_sentene=eng_Latn_user, translated_sentene=user["text"])
            example['User'] = eng_Latn_user
            example['User-Translated'] = user["text"]
            
            response = client.chat(
                model="command-r-plus",
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature = 0.7,
                p = 0.9,
            )

            output = match(response.message.content[0].text)
            if output:
                example['User-Translated-Rephrase'] = output.group(1).strip()
            else:
                example['User-Translated-Rephrase'] = response.message.content[0].text

            for bot in sample['Chatbot']:
                if bot["language"] == language:
                    formatted_prompt = PROMPT.format(raw_sentene=eng_Latn_chatbot, translated_sentene=bot["text"])
                    example['Chatbot'] = eng_Latn_chatbot
                    example['Chatbot-Translated'] = bot["text"]
                    response = client.chat(
                        model="command-r-plus",
                        messages=[
                            {
                                "role": "user",
                                "content": formatted_prompt
                            }
                        ],
                        temperature = 0.7,
                        p = 0.9,
                    )

                    output = match(response.message.content[0].text)
                    if output:
                        example['Chatbot-Translated-Rephrase'] = output.group(1).strip()
                    else:
                        example['Chatbot-Translated-Rephrase'] = response.message.content[0].text
                    break
            example['language'] = user["language"]
            dataset.append(example)
    # break

import datasets

datasets.Dataset.from_list(dataset).push_to_hub("olivernan/cauldron_nllb_translate_rephrase", key, split="train")