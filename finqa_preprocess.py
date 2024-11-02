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

# def to_bytes(image):
#     img_byte_arr = io.BytesIO()
#     image.convert("RGB").save(img_byte_arr, format="jpeg")
#     return img_byte_arr.getvalue()
import dataframe_image as dfi
def covert_to_table_image(table, name):

    header_colors = ['lightgreen', 'green', 'lightsteelblue', 'powderblue', 'sandybrown', 'lightsalmon', 'lightskyblue', 'lightgray', 'greenyellow', 'lightseagreen', 'lightslategray','forestgreen', 'mediumspringgreen', 'steelblue', 'mediumpurple' ]
    background_colors = ['lightblue', 'aqua', 'cyan', 'honeydew', 'ivory', 'lemonchiffon', 'ghostwhite', 'gainsboro', 'mistyrose', 'powderblue', 'snow', 'whitesmoke', 'lime', 'lightskyblue','khaki', 'mediumaquamarine', 'lightcyan', 'transparent', 'wheat']  

    df = pd.DataFrame(table)

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

    dfi.export(styled_df,f"finqa_images/{name}.jpeg")

    with open(f'finqa_images/{name}.jpeg', "rb") as image_file:
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


from tqdm import tqdm
with open('table_data/finqa_train.json') as f:
    finqa = json.load(f)

with open('table_data/finqa_processed.json' , 'w') as f:
    for i, data in enumerate(tqdm(finqa)):

        sample = {'Table':[], 'User': [], 'Chatbot': [], 'Image': []}
        
        sample['Table'].append(data['table'])

        User = ""
        for text in data['pre_text']:
            if text != ".":
                User += text.capitalize() + "\n"

        for text in data['post_text']:
            if text != ".":
                User += text.capitalize() + "\n"

        User += data['qa']['question'].capitalize()

        sample['User'].append({'text': User, "language": "eng_Latn", "source": "raw"})
        sample['User'].append({'text': User, "language": "eng_Latn", "source": "raw-processed"})

        Chatbot = "" 
        if data['qa']['explanation'] != "":
            Chatbot += "Rationale:" + data['qa']['explanation'].capitalize() + "\n"
        
        # computation = convert_expression(data['qa']['program_re'])
        
        computation = convert_expression(data['qa']['program'])

        if "const" in computation:
            computation = computation.replace("const_", "")
            if "m1" in computation:
                computation = computation.replace("m1", "-1")
            
        Chatbot += "Computations: " +  computation  + "\n"
        
        answer = data['qa']['exe_ans']
        try:
            if data['qa']['answer'] != "":
                if ('%' in data['qa']['answer']) and (type(answer) == float):
                    if round(float(data['qa']['answer'].replace('%', ''))/100,2) == round(answer, 2):
                        answer = data['qa']['answer']
        except:
            print(f"Error in {i}")
            print(data['qa']['answer'])
            print(answer)
            pass

        Chatbot += "Answer: " + str(answer)

        sample['Chatbot'].append({'text': Chatbot, "language": "eng_Latn", "source": "raw"})

        sample['Image'].append(covert_to_table_image(data['table'], i))

        sample['command_id'] = f"FINQA-{i}"
        sample['metadata'] = {"source": "FINQA"}
        sample['index'] = i

        f.write(json.dumps(sample, ensure_ascii=False) + "\n") 