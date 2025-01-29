from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import time
from jinja2 import Template
import re
import pandas as pd

# Replace 'model_name' with your chosen model's name and 'dataset_name' with your chosen dataset's name
model_names = ['roberta-base', 'roberta-large', 'google/flan-t5-small', 'facebook/bart-large-mnli', 'Qwen/Qwen1.5-0.5B-Chat', 'google/flan-t5-large', 'microsoft/phi-2', 'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-6.9b', 'Qwen/Qwen1.5-7B-Chat', 'mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Meta-Llama-3-70B-Instruct']
dataset_names = [['pkavumba/balanced-copa'], ['SetFit/mrpc'], ['nightingal3/fig-qa'], ['mteb/amazon_polarity'], ['gimmaru/story_cloze-2016'], ['yahoo_answers_topics'], ['dair-ai/emotion', 'split']]
prompt_template_strs = ['{{ premise }} because {{ choice1 }}', "Sentence 1: {{ text1 }}\nSentence 2: {{ text2 }}\nQuestion: Do both sentences mean the same thing?\nAnswer:", "{{ startphrase }} therefore {{ ending1 }}", "{{ text }}\nQuestion: Is this sentence positive or negative?\nAnswer:", "{{[input_sentence_1, input_sentence_2, input_sentence_3, input_sentence_4, sentence_quiz1]|join(' ')}}", "{{ question_title }}:\n{{ question_content }}\nWhat is the topic of the text (Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, or Politics & Government?)\nAnswer: {{ topic }}", "{{ text }}\nWhat emotion does the text convey (sadness, joy, love, anger, fear, or surprise?)\nAnswer:"]
splits = ['test', 'validation', 'test', 'test', 'test', 'test', 'test']

# Prepare the dataset for the model using the custom prompt template
def encode(example):
    template = Template(prompt_template_str)
    prompt = template.render({column_name: example[column_name] for column_name in column_names})
    outputs = tokenizer(prompt, truncation=True, padding='max_length')
    return outputs

# Measure the inference speed and log probabilities
def measure_inference_speed_and_log_probabilities(model, dataset, rob, times):
    model.eval()
    start_time = time.time()
    total_examples = 0

    with torch.no_grad():
        for batch in dataset:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            if rob:
                for i in range(times):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            total_examples += inputs['input_ids'].size(0)
    
    print(total_examples)

    end_time = time.time()
    time_taken = end_time - start_time
    examples_per_second = total_examples / time_taken
    return examples_per_second

results = {}
num = 0
for i, model_name in enumerate(model_names[num:num+1]):
    #skip qwen0.5
    # if i == 2:
    #     continue
    results[model_name] = {}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='auto', cache_dir="/scratch/afz225/.cache")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 't5' not in model_name and 'bart' not in model_name:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    for j, dataset_name in enumerate(dataset_names):
        # if j != 0:
        #     continue
        torch.cuda.empty_cache()
        # Load the dataset
        dataset = load_dataset(*dataset_name, cache_dir="/scratch/afz225/.cache")[splits[j]].train_test_split(train_size=20)['train']
        
        prompt_template_str = prompt_template_strs[j]
        
        # Use regular expressions to extract the column names from the template
        column_names = re.findall(r'\{\{\s*(\w+)\s*\}\}', prompt_template_str)
        encoded_dataset = dataset.map(encode, batched=False)
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        batch_size = 1  # You can adjust the batch size
        data_loader = DataLoader(encoded_dataset, batch_size=batch_size)
        
        # Calculate the average examples per second
        print(model_name, dataset_name)
        if "yahoo" in dataset_name[0]:
            times = 10
        elif "emotion" in dataset_name[0]:
            times = 6
        else:
            times = 2
        print(("roberta" in model_name), dataset_name, times)
        results[model_name][dataset_name[0]] = avg_examples_per_second = measure_inference_speed_and_log_probabilities(model, data_loader, rob=("roberta" in model_name), times=times)
        print(f'Average examples processed per second: {avg_examples_per_second}')
        speed_results = pd.DataFrame(results).T
        save_name = model_name.split("/")[-1]
        speed_results.to_csv(f"decoder-speeds-{save_name}.csv")