import imgkit
import pandas as pd
from io import StringIO
from tqdm import tqdm
import json
import imgkit
import pandas as pd
import re
import cohere
import random
import base64
from io import BytesIO
import io
from PIL import Image

import dataframe_image as dfi


def covert_to_table_image(html, name):

    header_colors = ['lightgreen', 'green', 'lightsteelblue', 'powderblue', 'sandybrown', 'lightsalmon', 'lightskyblue', 'lightgray', 'greenyellow', 'lightseagreen', 'lightslategray','forestgreen', 'mediumspringgreen', 'steelblue', 'mediumpurple' ]
    background_colors = ['lightblue', 'aqua', 'cyan', 'honeydew', 'ivory', 'lemonchiffon', 'ghostwhite', 'gainsboro', 'mistyrose', 'powderblue', 'snow', 'whitesmoke', 'lime', 'lightskyblue','khaki', 'mediumaquamarine', 'lightcyan', 'transparent', 'wheat']  

    df = pd.read_html(StringIO(html))[0].fillna('') 

    styled_df = (
        df.style
        .hide(axis="index")
        .hide(axis="columns")
        .set_table_styles([
            {'selector': 'tbody tr:nth-child(n+2)', 'props': [('background-color', random.choice(background_colors))]},
            {'selector': 'tbody tr:nth-child(1)', 'props': [('background-color', random.choice(header_colors))]},
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

    dfi.export(styled_df,f"MultiHiertt_images/{name}.jpeg")

    with open(f'MultiHiertt_images/{name}.jpeg', "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded_image}"


def convert_expression(program):
    operation_map = {
        'add': '+',
        'subtract': '-',
        'multiply': '*',
        'divide': '/',
        'exp': '**',
        'greater': '>',
        'table_average': 'table_average',
        'table_sum': 'table_sum',
        'table_max': 'table_max',
        'table_min': 'table_min',
    }
    operation_names = "|".join(map(re.escape, operation_map.keys()))
    pattern = rf"({operation_names})\((.*?)\)"

    split_operations = re.findall(pattern, program)
    
    # print(split_operations)
    for i, operation in enumerate(split_operations):
        op = operation[0]
        if op in ['table_average', 'table_sum', 'table_max', 'table_min']:
            split_operations[i] = f"({operation[0]}({operation[1]})"
        else:
            op_sym = operation_map[op]
            args = operation[1].split(',') #re.findall(r'\(([^()]+)\)', operation)[0].split(',')
            assert len(args) == 2
            arg1 = args[0].strip()
            arg2 = args[1].strip()
            if "#" in arg1:
                index = int(arg1.replace("#", ""))
                arg1 = split_operations[index]
            if "#" in arg2:
                index = int(arg2.replace("#", ""))
                arg2 = split_operations[index]

            split_operations[i] = f"({arg1} {op_sym} {arg2})"

    return split_operations[-1]

with open('table_data/MultiHiertt_train.json') as f:
    MultiHiertt = json.load(f)

with open('table_data/MultiHiertt_processed.json' , 'w') as f:
    for i, data in enumerate(tqdm(MultiHiertt)):

        sample = {'Table':[], 'User': [], 'Chatbot': [], 'Image': []}
        
        sample['Table'].append(data['tables'])

        User = data['qa']['question']

        sample['User'].append({'text': User, "language": "eng_Latn", "source": "raw"})
        sample['User'].append({'text': User, "language": "eng_Latn", "source": "raw-processed"})

        Chatbot = "" 
        if data['qa']['program'] != "":
            computation = convert_expression(data['qa']['program'])

            if "const" in computation:
                computation = computation.replace("const_", "")
                if "m1" in computation:
                    computation = computation.replace("m1", "-1")
                
            Chatbot += "Computations: " +  computation  + "\n"

        Chatbot += "Answer: " + str(data['qa']['answer'])

        sample['Chatbot'].append({'text': Chatbot, "language": "eng_Latn", "source": "raw"})

        for j, table in enumerate(data['tables']):
            sample['Image'].append(covert_to_table_image(table, f"{i}_{j}"))
        
        sample['command_id'] = f"MultiHiertt-{i}"
        sample['metadata'] = {"source": "MultiHiertt"}
        sample['index'] = i

        f.write(json.dumps(sample, ensure_ascii=False) + "\n")