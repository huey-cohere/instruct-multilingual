# import json
# import os
# from tqdm import tqdm
# import multiprocessing as mp
# import re
# import io
# import base64
# import dataframe_image as dfi
# import pandas as pd
# import base64
# import io
# from PIL import Image
# import random

# FOLDER = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated/RecapMultiHiertt_translation"
# OUTPUT_BASE = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated_rendered_table/RecapMultiHiertt_translation"
# os.makedirs(OUTPUT_BASE, exist_ok=True)

# HEADER_COLORS = ['lightgreen', 'green', 'lightsteelblue', 'powderblue', 'sandybrown', 'lightsalmon', 'lightskyblue', 'lightgray', 'greenyellow', 'lightseagreen', 'lightslategray', ]
# BACKGROUND_COLORS = ['lightblue', 'aqua', 'cyan', 'honeydew', 'ivory', 'lemonchiffon', 'ghostwhite', 'gainsboro', 'mistyrose', 'powderblue', 'snow', 'whitesmoke', 'lime', 'lightskyblue','khaki', 'mediumaquamarine']  



# def get_dir_list():
#     dir_list = []
#     for lang in os.listdir(FOLDER):
#         dir_list.append((lang, os.path.join(FOLDER, lang, "train.jsonl")))
#     return dir_list


# def covert_to_table_image(table):

#     df = pd.DataFrame(table)
    
#     styled_df = (
#         df.style
#         .hide(axis="index")
#         .hide(axis="columns")
#         .set_table_styles([
#             {'selector': 'tbody tr:nth-child(n+2)', 'props': [('background-color', random.choice(BACKGROUND_COLORS))]},
#             {'selector': 'tbody tr:nth-child(1)', 'props': [('background-color', random.choice(HEADER_COLORS))]},
#             {'selector': 'table', 'props': [
#                 ('border', '1px solid white'),
#             ]},
#             {'selector': 'td', 'props': [
#                 ('min-width', '150px'), 
#                 ('max-width', '450px'),
#                 ('padding', '15px'),
#             ]}
#         ])
#         .set_properties(**{
#             'text-align': 'center',
#             'font-size': '12px',
#         })
#     )

#     # dfi.export(styled_df,f"finqa_translated_images/1.jpeg")

#     # with open(f'finqa_translated_images/1.jpeg', "rb") as image_file:
#     #     print(image_file.read())
#     #     encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

#     with io.BytesIO() as buffer:
#         dfi.export(styled_df, buffer)
#         buffer.seek(0)  
#         encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

#     return f"data:image/jpeg;base64,{encoded_image}"


# def process_file(args):
#     lang, path = args
#     output_folder = os.path.join(OUTPUT_BASE, lang)
#     os.makedirs(output_folder, exist_ok=True)
#     output_file_path = os.path.join(output_folder, "train.jsonl")

#     with open(path, "r") as input_file, open(output_file_path, "a") as output_file:
#         for i, input_line in enumerate(input_file):
#             data = json.loads(input_line)
#             translated_tables = data["Translated_Table"]
#             translated_table_images = []
#             for table in translated_tables:
#                 translated_table_images.append(covert_to_table_image(table)) 
                
#             data["Translated_Image"] = translated_table_images

#             output_file.write(json.dumps(data, ensure_ascii=False) + "\n")

#     # print(count)
            

# if __name__ == "__main__":

#     # Use all available CPU cores
#     dir_list = get_dir_list()
#     print(f"Processing {len(dir_list)} files...")
    
#     with mp.Pool(mp.cpu_count()) as pool:
#         list(tqdm(pool.imap_unordered(process_file, dir_list), total=len(dir_list)))

# print("Processing complete.")


from concurrent.futures import ProcessPoolExecutor
import json
import os
import pandas as pd
import random
import base64
import io
import dataframe_image as dfi
from tqdm import tqdm
import concurrent
import math
import argparse
# Define constants
import time

# FOLDER = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated/RecapMultiHiertt_translation"
# OUTPUT_BASE = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated_rendered_table/RecapMultiHiertt_translation"
# os.makedirs(OUTPUT_BASE, exist_ok=True)

HEADER_COLORS = ['lightgreen', 'green', 'lightsteelblue', 'powderblue', 'sandybrown', 'lightsalmon', 'lightskyblue', 'lightgray', 'greenyellow', 'lightseagreen', 'lightslategray', ]
BACKGROUND_COLORS = ['lightblue', 'aqua', 'cyan', 'honeydew', 'ivory', 'lemonchiffon', 'ghostwhite', 'gainsboro', 'mistyrose', 'powderblue', 'snow', 'whitesmoke', 'lime', 'lightskyblue','khaki', 'mediumaquamarine']  

def convert_to_table_image(data):
    
    tables = data["Translated_Table"]

    Translated_Images = []

    for table in tables:
        df = pd.DataFrame(table)

        styled_df = (
            df.style
            .hide(axis="index")
            .hide(axis="columns")
            .set_table_styles([
                {'selector': 'tbody tr:nth-child(n+2)', 'props': [('background-color', random.choice(BACKGROUND_COLORS))]},
                {'selector': 'tbody tr:nth-child(1)', 'props': [('background-color', random.choice(HEADER_COLORS))]},
                {'selector': 'table', 'props': [
                    ('border', '1px solid white'),
                ]},
                {'selector': 'td', 'props': [
                    ('min-width', '150px'), 
                    ('max-width', '450px'),
                    ('padding', '15px'),
                ]}
            ])
            .set_properties(**{
                'text-align': 'center',
                'font-size': '12px',
            })
        )

        with io.BytesIO() as buffer:
            dfi.export(styled_df, buffer)
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        Translated_Images.append(f"data:image/jpeg;base64,{encoded_image}")

    data['Translated_Image'] = Translated_Images

    return data


def process_single_file(input_path, output_path):

    with open(input_path, "r") as input_file:
        dataset = [json.loads(line) for line in input_file]
    
    # dataset = dataset[:200]

    print(f"Processing {input_path.split('/')[-2]} file with {len(dataset)} samples...")
    
    # start_time = time.time()

    # batch_size = 96
    # with concurrent.futures.ProcessPoolExecutor(max_workers=96) as executor:
    #     new_dataset = []
    #     for batch in range(0, len(dataset), batch_size):
    #         batch_data = dataset[batch:batch+batch_size]
    #         new_dataset += list(executor.map(convert_to_table_image, batch_data))

    # end_time = time.time()
    # print(f"Processing completed in {end_time - start_time:.2f} seconds")

    with concurrent.futures.ProcessPoolExecutor(max_workers=96) as executor:

        futures = [
            executor.submit(
                convert_to_table_image, (data)
            )
            for data in dataset
        ]

        new_dataset = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            new_dataset.append(result)

    # new_dataset = []
    # for data in tqdm(dataset, desc="Rendering Dataset"):
    #     covert_to_table_image(data)
    #     new_dataset.append(data)

    # print(f"Processing time: {time.time() - start_time}")

    # print(new_dataset[0])

    print(f"new dataset size: {len(new_dataset)}")

    with open(output_path, "w+") as f:
        for data in new_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")



if __name__ == "__main__":

    argparse  = argparse.ArgumentParser()
    argparse.add_argument("--input_path", type=str, default='/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated/RecapMultiHiertt_translation/zho_Hans/train.jsonl')
    argparse.add_argument("--output_path", type=str, default='/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated_rendered_table/RecapMultiHiertt_translation/zho_Hans/train.jsonl')
    args = argparse.parse_args()
    # dir_list = get_dir_list()
    # print(f"Processing {len(dir_list)} files...")

    # # Process files one by one, using multiprocessing within each file for table processing
    process_single_file(args.input_path, args.output_path)

    print("Processing complete.")
