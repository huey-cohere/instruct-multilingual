# import cohere

# # client = cohere.ClientV2("MfZwS1plvJfM7vARPs92RbCScEwRniTcCXfmfAdU", base_url="https://stg.api.cohere.ai")
# client = cohere.ClientV2("Ymj9fYrCh1uXyttUrxFppAPCzFKkB3jz6ELJcx2i")

# PROMPT = """
# Original Sentence: {raw_sentene}\n
# Translated Sentence: {translated_sentene}\n

# Rephrase the translated sentence to enhance its quality, ensuring it aligns closely with the original in meaning, structure, tone, and style.
# Ensure that the rephrased sentence conveys the same meaning as the original sentence but avoid altering the core message or introducing new information. 
# Correct any grammatical errors present in the translated sentence.
# Maintain a structure similar to the original sentence. 
# Match the tone and style of the original sentence. 
# Preserve any stylistic elements such as enumeration, punctuation or capitalization.

# The output must strictly follow this format:\n
# Rephrase Translated Sentence: <sentence>"""

# import json
# import os
# import random
# key = "RecapCauldronLocalized_narratives_translation"

# eng_Latn_sample = []
# with open(f"/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/{key}/eng_Latn/train.jsonl", "r") as f:
#     i = 0
#     for line in f:
#         eng_Latn_sample.append(json.loads(line))
#         i += 1
#         if i == 10:
#             break
#     # eng_Latn = [json.loads(line) for line in f]

# # eng_Latn_sample = random.sample(eng_Latn, 10)


# samples = []
# for sample in eng_Latn_sample:
#     example = {"id": sample["command_id"], "User":[], "Chatbot":[]}
#     for turn in sample["turns"]:
#         if turn["role"] == "User":
#             for content in turn["content"]:
#                 if 'text' in content:
#                     example["User"].append({"text": content["text"], "language": "eng_Latn", "source": "raw"})
#         if turn["role"] == "Chatbot":
#             example["Chatbot"].append({"text": turn["content"], "language": "eng_Latn", "source": "raw-gpt_recap"})
#     samples.append(example)


# samples_dict = {sample["id"]: sample for sample in samples}


# for lang in os.listdir(f"/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/{key}/"):
#     if lang == "eng_Latn":
#         continue
#     with open(f"/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/{key}/{lang}/train.jsonl", "r") as f:
#         for line in f:
#             data = json.loads(line)
#             if data['command_id'] in samples_dict:
#                 for turn in data["turns"]:
#                     if turn["role"] == "User":
#                         for content in turn["content"]:
#                             if 'text' in content:
#                                 samples_dict[data['command_id']]['User'].append({"text": content["text"], "language": lang, "source": "raw-nllb_transl"})
#                     if turn["role"] == "Chatbot":
#                         samples_dict[data['command_id']]["Chatbot"].append({"text": turn["content"], "language": lang, "source": "raw-gpt_recap-nllb_transl"})

# samples = list(samples_dict.values())

# import re
# dataset = []
# from tqdm import tqdm
# def match(output):
#     return re.search(r'Rephrase Translated Sentence:\s*(.+)', output)

# for sample in samples:
#     for user in sample['User']:
#         if user["language"] == "eng_Latn":
#             eng_Latn_user = user["text"]
#             break
#     for chatbot in sample['Chatbot']:
#         if chatbot["language"] == "eng_Latn":
#             eng_Latn_chatbot = chatbot["text"]
#             break
    
    
#     for i, user in tqdm(enumerate(sample['User'])):
#         example = {}
#         example['index'] = i
#         language = user["language"]
#         if language != "eng_Latn":
            
#             formatted_prompt = PROMPT.format(raw_sentene=eng_Latn_user, translated_sentene=user["text"])
#             example['User'] = eng_Latn_user
#             example['User-Translated'] = user["text"]
            
#             response = client.chat(
#                 model="command-r-plus",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": formatted_prompt
#                     }
#                 ],
#                 temperature = 0.7,
#                 p = 0.9,
#             )

#             output = match(response.message.content[0].text)
#             if output:
#                 example['User-Translated-Rephrase'] = output.group(1).strip()
#             else:
#                 example['User-Translated-Rephrase'] = response.message.content[0].text

#             for bot in sample['Chatbot']:
#                 if bot["language"] == language:
#                     formatted_prompt = PROMPT.format(raw_sentene=eng_Latn_chatbot, translated_sentene=bot["text"])
#                     example['Chatbot'] = eng_Latn_chatbot
#                     example['Chatbot-Translated'] = bot["text"]
#                     response = client.chat(
#                         model="command-r-plus",
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": formatted_prompt
#                             }
#                         ],
#                         temperature = 0.7,
#                         p = 0.9,
#                     )

#                     output = match(response.message.content[0].text)
#                     if output:
#                         example['Chatbot-Translated-Rephrase'] = output.group(1).strip()
#                     else:
#                         example['Chatbot-Translated-Rephrase'] = response.message.content[0].text
#                     break
#             example['language'] = user["language"]
#             dataset.append(example)
#     # break

# import datasets

# datasets.Dataset.from_list(dataset).push_to_hub("olivernan/cauldron_nllb_translate_rephrase", key, split="train")


import datasets

import cohere

# client = cohere.ClientV2("MfZwS1plvJfM7vARPs92RbCScEwRniTcCXfmfAdU", base_url="https://stg.api.cohere.ai")
client = cohere.ClientV2("Ymj9fYrCh1uXyttUrxFppAPCzFKkB3jz6ELJcx2i")

PROMPT_TEMPLATE = """Original Sentence: {raw_sentene}\n
Translated Sentence: {translated_sentene}\n
Rephrase the translation to improve its quality, ensuring it aligns closely with the original input in meaning, structure, tone, and style.
Maintain the core message without introducing new information. 
Include all sentences from the original input.
Correct any grammatical errors in the translated sentence.
Preserve the original sentence structure as much as possible. 
Match the tone and style of the original sentence. 
Retain any stylistic elements such as enumeration, punctuation, or capitalization.\n
The output must strictly follow this format:
Rephrase Translated Sentence: <sentence>"""



from tqdm import tqdm
import re
import time

def main(dataset_name):
    dataset = datasets.load_dataset("olivernan/cauldron_nllb_translate_rephrase", dataset_name)['train']



    new_dataset = []

    def make_request(prompt):
        i = 0
        while i < 20:
            try:
                response = client.chat(
                    model="command-r-plus",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature = 0.7,
                    p = 0.9,
                )
                return response.message.content[0].text
            except Exception as e:
                i += 1
                print(e)
                time.sleep(2)
                continue

        # return response.message.content[0].text

        # output= re.search(r'Rephrase Translated Sentence:\s*(.+)', response.message.content[0].text) 
        # output_chatbot = re.search(r'Rephrase Translated Sentence:\s*(.+)', response_chatbot.message.content[0].text)
        
        #     except Exception as e:
        #         i += 1
        #         print(e)
        #         print(response)
        #         continue
        # return None
        
    for data in tqdm(dataset):
        formatted_input_user = PROMPT_TEMPLATE.format(raw_sentene=data['User'], translated_sentene=data['User-Translated'])
        formatted_input_chatbot = PROMPT_TEMPLATE.format(raw_sentene=data['Chatbot'], translated_sentene=data['Chatbot-Translated'])
        i = 0
        output_user = make_request(formatted_input_user)
        output_chatbot = make_request(formatted_input_chatbot)   
        while not re.search(r'Rephrase Translated Sentence:\s*(.+)', output_user):
            output_user = make_request(formatted_input_user)
            i += 1
            print(f"User retry: {i}")
            if i == 1:
                print(output_user)
            if i == 20:
                print("User failed")
                break
        i = 0
        while not re.search(r'Rephrase Translated Sentence:\s*(.+)', output_chatbot):
            output_chatbot = make_request(formatted_input_chatbot)
            i += 1
            print(f"Chatbot retry: {i}")
            if i == 1:
                print(output_chatbot)
            if i == 20:
                print("Chatbot failed")
                break

        data['User-Translated-Rephrase-2'] = re.search(r'Rephrase Translated Sentence:\s*(.+)', output_user).group(1).strip() 
        data['Chatbot-Translated-Rephrase-2'] = re.search(r'Rephrase Translated Sentence:\s*(.+)', output_chatbot).group(1).strip()
        # print(data['User-Translated-Rephrase-2'])
        # print(data['Chatbot-Translated-Rephrase-2'])
        # break
        new_dataset.append(data)
            
    new_dataset = datasets.Dataset.from_list(new_dataset)

    column_reorder = ['index', 'language', 'User', 'User-Translated', 'User-Translated-Rephrase','User-Translated-Rephrase-2', 'Chatbot', 'Chatbot-Translated', 'Chatbot-Translated-Rephrase', 'Chatbot-Translated-Rephrase-2']
    new_dataset = new_dataset.select_columns(column_reorder)

    new_dataset.push_to_hub("olivernan/cauldron_nllb_translate_rephrase", dataset_name, split="train")


if __name__ == "__main__":
    # for dataset_name in ['RecapCauldronCocoqa_translation', 'RecapCauldronLocalized_narratives_translation', 'RecapCauldronSpot_the_diff_translation', 'RecapCauldronSt_vqa_translation', 'RecapCauldronTextcaps_translation']:
    for dataset_name in ['RecapCauldronTextcaps_translation']:
        main(dataset_name)