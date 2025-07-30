import json
import os
data_path = "D:\\github\\rl_sft_data\\src\\TMATH-master\\TMATH-master"

result=[]
def read_json_files_in_directory(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        content = json.load(f)
                        data.append(content)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}")
    return      data


result = read_json_files_in_directory(data_path)

# for data in enumerate(result):
#     q_prompts=data['problem']
print("")
