import os

import openai
import re
import json
import datetime
import prompts as prompts

task_name="safe_agent_bench"
api_key = 'sk-rMwveLlHvxcADL8vcekIG49bxKnzUZcAhdc7kkJfjpNDYw2L'
base_url = 'https://yibuapi.com/v1'

output_path = f"/data/rl_sft_data/{task_name}/"
system_prompt =prompts.system_prompt

questions=prompts.safe_agent_bench

def get_openai_response(model_name, question, api_key, base_url):
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"请求失败：{str(e)}"


if __name__ == "__main__":
    case_number = 1
    #model_name = "gpt-4o-mini"
    model_name = "grok-3"

    base_filename = f"{task_name}_{model_name}"

for q_idx,q in enumerate(questions):
    for case_i in range(case_number):
        print(f"Case  {case_i + 1}")
        response = get_openai_response(model_name, q, api_key, base_url)
        print(response)

        filename = base_filename + f"_{q_idx}.txt"
        prompt_filename = base_filename + f"_{q_idx}_prompt.txt"
        # 将 response 追加写入 txt 文件

        # with open(output_path + filename, 'w+', encoding='utf-8') as f:
        #     f.write(response)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # 确保输出文件夹存在
        # 将 response 写入文件

        if not os.path.exists(output_path + prompt_filename):
            with open(output_path + prompt_filename, 'w+', encoding='utf-8') as f:
                f.write(system_prompt+"\n"+q)
        if not os.path.exists(output_path + filename):
            with open(output_path + filename, 'w+', encoding='utf-8') as f:
                f.write(response)
        else:
            # 如果文件已存在，则追加内容
            with open(output_path + filename, 'a', encoding='utf-8') as f:
                f.write(response)


