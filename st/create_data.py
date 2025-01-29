from datasets import load_dataset, Dataset
import random
import numpy as np
import torch
from sklearn.utils import resample
import pandas as pd
import re
from datasets import Dataset, concatenate_datasets

def fill_template(templates, values):
    temp = random.sample(templates,1)[0]
    for i in range(len(values)):
        temp = temp.replace("${"+str(i+1)+"}", values[i])
    return temp


def create_statement_dataset_sent_comparsion(dataset, templates, columns, label_column, SPC, cache_dir, num_statements=10000, prop_negative=0.5, negative_templates=None,splits=["train"]):
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    def create_statements_labels(batch):
        return {"statement":[fill_template([template], [batch[column][example] for column in columns]) for example in range(len(batch[label_column])) for template in templates] + [fill_template([template], [batch[column][example] for column in columns]) for example in range(len(batch[label_column])) for template in negative_templates], "is_true":[example for example in batch[label_column] for template in templates]+[1-example for example in batch[label_column] for template in negative_templates]}
    
    updated_data = [split.map(create_statements_labels,batched=True, remove_columns=col_names).to_pandas() for split in data]
    updated_data = [Dataset.from_dict(resample(data,n_samples=min(SPC, len(data)), replace=False, stratify=data['is_true'])) for data in updated_data]
    return updated_data

def create_statement_dataset_multiple_choice(dataset, templates, question, answers, label_column, SPC, cache_dir, label_offset=0, num_statements=10000,splits=["train"], replace=False):
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    def create_statements_labels(batch):
        answer_choice = random.choices(range(len(answers)), k=len(batch[label_column])*len(templates))
#         example['statement'] = fill_template(templates, [example[question], example[answers[answer_choice]]])
#         example['is_true'] = int(str(answer_choice) == (str(int(example[label_column])-label_offset)))
#         statements = [fill_template([template], [batch[question][example], batch[answers[answer_choice[example]]][example]]) for example in range(len(batch[label_column])) for template in templates]
        statements = []
        for example in range(len(batch[label_column])):
            for template in templates:
                statements.append(fill_template([template], [batch[question][example], batch[answers[answer_choice[example]]][example]]))
            if replace:
                statements.append(batch[question][example].replace("_", batch[answers[answer_choice[example]]][example]))
        
        if replace:
            truth = [int(str(int(example)-label_offset)==str(answer_choice[i])) for i, example in enumerate(batch[label_column]) for template in range(len(templates)+1)]
        else:
            truth = [int(str(int(example)-label_offset)==str(answer_choice[i])) for i, example in enumerate(batch[label_column]) for template in templates]
        return {"statement":statements, "is_true":truth}
    updated_data = [split.map(create_statements_labels, batched=True, remove_columns=data[0].column_names).to_pandas() for split in data]
#     downsample = [True if 50000<len(data) else False for data in updated_data]
    updated_data = [Dataset.from_dict(resample(data,n_samples=min(SPC, len(data)), replace=False, stratify=data['is_true'])) for data in updated_data]
    return updated_data

def create_statements_labels_mnli(batch, label_column, templates, columns):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    template_choices = [templates[int(batch[label_column][i])] if example_type == 'positive' else templates[(int(batch[label_column][i])+1)%len(templates)] for i, example_type in enumerate(example_types)]
    statements = [fill_template([template], [batch[text][example] for text in columns]) for example in range(len(batch[label_column])) for template in template_choices[example]]
    is_true = [1 if choice=='positive' else 0 for i, choice in enumerate(example_types) for template in template_choices[i]]
    return {'statement':statements, 'is_true':is_true}

def create_statements_labels_mintaka(batch, templates, label_column, question):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    statements = [fill_template([template], [batch[question][example], batch[label_column][example] if example_types[example] == 'positive' else batch[label_column][(example+1)%len(batch)]]) for example in range(len(batch[label_column])) for template in templates]
    is_true = [1 if example_type == 'positive' else 0 for example_type in example_types for template in templates]
    return {'statement':statements, 'is_true':is_true}

def create_statements_labels_yelp(batch, label_column, templates, columns):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    template_choices = [templates[int(batch[label_column][i])] if example_type == 'positive' else templates[(int(batch[label_column][i])+1)%len(templates)] for i, example_type in enumerate(example_types)]
    statements = [fill_template([template], [batch[text][example] for text in columns]) for example in range(len(batch[label_column])) for template in template_choices[example]]
    is_true = []
    for i, example in enumerate(example_types):
        for template in template_choices[i]:
            is_true.append(1 if example =='positive' else 0)
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_wikilingua(batch, label_column, templates, question):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    statements = [fill_template([template], [batch[question][example], batch[label_column][example] if example_types[example] == 'positive' else batch[label_column][(example+1)%len(batch)]]) for example in range(len(batch[label_column])) for template in templates]
    is_true = [1 if example_type == 'positive' else 0 for example_type in example_types for template in templates]
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_squad(batch, label_column, templates, context, question):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    random_span = []
    for example in range(len(batch[label_column])):
        if example_types[example] == 'positive':
            random_span.append(None)
        else:
            random_span_length = random.randint(0, len(batch[context][example])//4)
            random_span_start = random.randint(0, len(batch[context][example])-random_span_length-1)
            random_span.append((random_span_length, random_span_start))
    statements = [fill_template([template], [batch[context][example], batch[question][example], batch[label_column][example]['text'][0] if example_types[example] == 'positive' else batch[context][example][random_span[example][1]:random_span[example][1]+random_span[example][0]]]) for example in range(len(batch[label_column])) for template in templates]
    is_true = [1 if example_type == 'positive' else 0 for example_type in example_types for template in templates]
    return {'statement':statements, 'is_true':is_true}

def create_statements_labels_offensive(batch, label_column, templates, question, class_names):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    statements = []
    is_true = []
    for i, example in enumerate(example_types):
        if example == 'positive':
            statements += [fill_template([template], [batch[question][i], class_names[batch[label_column][i]]]) for template in templates]
            is_true += [1 for template in templates]
        else:
            statements += [fill_template([template], [batch[question][i], class_names[1-batch[label_column][i]]]) for template in templates]
            is_true += [0 for template in templates]
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_massive(batch, label_column, templates, question, class_names):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    statements = []
    is_true = []
    for i, example in enumerate(example_types):
        if example == 'positive':
            statements+=[fill_template([template], [batch[question][i], class_names[batch[label_column][i]]]) for template in templates]
            is_true += [1 for template in templates]
        else:
            cp = class_names.copy()
            del cp[batch[label_column][i]]
            wrong = random.choice(cp)
            statements += [fill_template([template], [batch[question][i], wrong]) for template in templates]
            is_true += [0 for template in templates]
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_dpr(batch, templates):
    example_types = random.choices(['positive', 'negative'], k=len(batch['label']))
    statements = []
    is_true = []
    for i, example in enumerate(example_types):
        if example == 'positive':
            statements+=[fill_template([template], [batch["sentence"][i], batch["pronoun"][i], batch["candidates"][i][batch["label"][i]]]) for template in templates]
            statements.append(re.sub(r"\b"+batch["pronoun"][i]+r"\b", batch["candidates"][i][batch["label"][i]], batch["sentence"][i], flags=re.IGNORECASE))
            is_true += [1 for template in range(len(templates)+1)]
        else:
            statements+=[fill_template([template], [batch["sentence"][i], batch["pronoun"][i], batch["candidates"][i][1-batch["label"][i]]]) for template in templates]
            statements.append(re.sub(r"\b"+batch["pronoun"][i]+r"\b", batch["candidates"][i][1-batch["label"][i]], batch["sentence"][i], flags=re.IGNORECASE))
            is_true += [0 for template in range(len(templates)+1)]
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_qasc(batch, template_1, template_2, template_3, template_4):
    example_types = random.choices(['positive', 'negative'], k=len(batch['answerKey']))
    statements = []
    is_true = []
    for i, example in enumerate(example_types):
        if example == 'positive':
            statements += [fill_template([template], [batch['formatted_question'][i], batch["answerKey"][i]]) for template in template_1]
            statements += [fill_template([template], [batch['formatted_question'][i], batch["choices"][i]["text"][(ord(batch["answerKey"][i])-ord("A"))]]) for template in template_2]
            statements += [fill_template([template], [batch['combinedfact'][i], batch['question'][i], batch["choices"][i]["text"][(ord(batch["answerKey"][i])-ord("A"))]]) for template in template_3]
            statements += [fill_template([template], [batch['combinedfact'][i], batch['formatted_question'][i], batch["answerKey"][i]]) for template in template_4]
            is_true += [1]*(len(template_1)+len(template_2)+len(template_3)+len(template_4))
        else:
            rand_letter = random.choice([chr(j) for j in range(ord("A"), ord("H")+1) if chr(j) != batch["answerKey"][i]])
            
            statements += [fill_template([template], [batch['formatted_question'][i], rand_letter]) for template in template_1]
            statements += [fill_template([template], [batch['formatted_question'][i], batch["choices"][i]["text"][(ord(rand_letter)-ord("A"))]]) for template in template_2]
            statements += [fill_template([template], [batch['combinedfact'][i], batch['question'][i], batch["choices"][i]["text"][(ord(rand_letter)-ord("A"))]]) for template in template_3]
            statements += [fill_template([template], [batch['combinedfact'][i], batch['formatted_question'][i], rand_letter]) for template in template_4]
            is_true += [0]*(len(template_1)+len(template_2)+len(template_3)+len(template_4))
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_sciq(batch, template_1, template_2):
    example_types = random.choices(['positive', 'negative'], k=len(batch['correct_answer']))
    statements = []
    is_true = []
    for i, example in enumerate(example_types):
        if example == 'positive':
            statements += [fill_template([template], [batch['question'][i], batch["correct_answer"][i]]) for template in template_1]
            statements += [fill_template([template], [batch['support'][i], batch['question'][i], batch["correct_answer"][i]]) for template in template_2]
            
            is_true += [1]*(len(template_1)+len(template_2))
        else:
            rand_letter = random.choice(["distractor1", "distractor2", "distractor3"])
            
            statements += [fill_template([template], [batch['question'][i], batch[rand_letter][i]]) for template in template_1]
            statements += [fill_template([template], [batch['support'][i], batch['question'][i], batch[rand_letter][i]]) for template in template_2]
            
            is_true += [0]*(len(template_1)+len(template_2))
    return {'statement':statements, 'is_true':is_true}
def filter_data(example):
    if " _" not in example["question"]:
        return False
    return True
def create_statements_labels_race(batch, template_1):
    example_types = random.choices(['positive', 'negative'], k=len(batch['answer']))
    statements = []
    is_true = []
    for i, example in enumerate(example_types):
        if example == 'positive':
            q = batch['question'][i].replace("_", batch['options'][i][ord(batch['answer'][i])-ord("A")])
            statements += [fill_template([template], [batch['article'][i], q]) for template in template_1]
            
            is_true += [1]*(len(template_1))
        else:
            rand_letter = random.choice([chr(j) for j in range(ord("A"), ord("D")+1) if chr(j) != batch["answer"][i]])
            q = batch['question'][i].replace("_", batch['options'][i][ord(rand_letter)-ord("A")])
            statements += [fill_template([template], [batch['article'][i], q]) for template in template_1]
            
            is_true += [0]*(len(template_1))
    return {'statement':statements, 'is_true':is_true}
def create_statements_labels_samsum(batch, label_column, templates, question):
    example_types = random.choices(['positive', 'negative'], k=len(batch[label_column]))
    statements = [fill_template([template], [batch[question][example], batch[label_column][example] if example_types[example] == 'positive' else batch[label_column][(example+1)%len(batch)]]) for example in range(len(batch[label_column])) for template in templates]
    is_true = [1 if example_type == 'positive' else 0 for example_type in example_types for template in templates]
    return {'statement':statements, 'is_true':is_true}



def generate_train_data(excluded_data, ppc, spc, cache_dir):
    SPC = spc
    PPC = ppc
    train_datasets = []

    print(excluded_data)
    
    dataset = ["SetFit/qqp"]
    templates = ["\"${1}\" is a duplicate of \"${2}\"", "\"${1}\" duplicates \"${2}\"", "\"${1}\" is the same as \"${2}\"", "\"${1}\" can be stated as \"${2}\"", "\"${1}\" is a paraphrase of \"${2}\""]
    negative_templates = ["\"${1}\" is not a duplicate of \"${2}\"", "\"${1}\" does not duplicate \"${2}\"", "\"${1}\" doesn't duplicate \"${2}\"", "\"${1}\" is not the same as \"${2}\"", "\"${1}\" is unrelated to \"${2}\"", "\"${1}\" can't be stated as \"${2}\"", "\"${1}\" can not be stated as \"${2}\"", "\"${1}\" is not a paraphrase of \"${2}\"", "\"${1}\" isn't a paraphrase of \"${2}\""]
    columns = ['text1', 'text2']
    label_column = 'label'
    
    templates = random.sample(templates, k=min(len(templates),PPC))
    negative_templates = random.sample(negative_templates, k=min(len(negative_templates),PPC))

    if "qqp" not in excluded_data:
        qqp_statements = create_statement_dataset_sent_comparsion(dataset, templates, columns, label_column, SPC=SPC, cache_dir=cache_dir, negative_templates=negative_templates)[0]
        train_datasets.append(qqp_statements)

    dataset = ["winogrande", 'winogrande_xl']
    templates = ["In \"${1}\", _ is: ${2}", "Q: \"${1}\", A: ${2}", "The missing word in: \"${1}\" is ${2}", "_ in: \"${1}\" is ${2}", "\"${1}\", _ is: ${2}"]
    templates = random.sample(templates, k=min(len(templates),PPC))
    question = 'sentence'
    answers = ['option1','option2']
    label_column = 'answer'
    if "winogrande" not in excluded_data:
        winogrande_statements = create_statement_dataset_multiple_choice(dataset,templates, question, answers, label_column, SPC=SPC, cache_dir=cache_dir, label_offset=1, replace=True)[0]
        train_datasets.append(winogrande_statements)

    dataset = ["piqa"]
    templates = ["${1} ${2}", "Goal:${1}, Solution: ${2}", "If the goal is: ${1}, then the solution is: ${2}", "Problem: ${1}, Solution: ${2} "]
    templates = random.sample(templates, k=min(len(templates),PPC))
    question = 'goal'
    answers = ['sol1','sol2']
    label_column = 'label'
    if "piqa" not in excluded_data:
        piqa_statements = create_statement_dataset_multiple_choice(dataset,templates, question, answers, label_column, SPC=SPC, cache_dir=cache_dir)[0]
        train_datasets.append(piqa_statements)

    dataset = ["SetFit/mnli"]
    templates = [["\"${1}\" entails \"${2}\"", "${1}? yes, ${2}", "Premise: ${1}, Hypothesis: ${2}, label: Entailment"], ["\"${1}\" is neutral with regards to \"${2}\"", "${1}? maybe, ${2}", "Premise: ${1}, Hypothesis: ${2}, label: Neutral"], ["\"${1}\" contradicts \"${2}\"", "${1}? no, ${2}", "Premise: ${1}, Hypothesis: ${2}, label: Contradiction"]]
    templates = [random.sample(template, k=min(len(template),PPC)) for template in templates]
    num_statements = 10000
    splits = ['train']
    label_column = 'label'
    columns = ['text1', 'text2']
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    if "mnli" not in excluded_data:
        mnli_statements = [split.map(create_statements_labels_mnli, fn_kwargs={"label_column" : label_column, "templates":templates, "columns": columns}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        mnli_statements = Dataset.from_dict(resample(mnli_statements,n_samples=min(SPC, len(mnli_statements)), replace=False, stratify=mnli_statements['is_true']))
        train_datasets.append(mnli_statements)
    dataset = ["snli"]
    templates = [["\"${1}\" entails \"${2}\"", "${1}? yes, ${2}", "Premise: ${1}, Hypothesis: ${2}, label: Entailment"], ["\"${1}\" is neutral with regards to \"${2}\"", "${1}? maybe, ${2}", "Premise: ${1}, Hypothesis: ${2}, label: Neutral"], ["\"${1}\" contradicts \"${2}\"", "${1}? no, ${2}", "Premise: ${1}, Hypothesis: ${2}, label: Contradiction"]]
    templates = [random.sample(template, k=min(len(template),PPC)) for template in templates]
    num_statements = 10000
    splits = ['train']
    label_column = 'label'
    columns = ['premise', 'hypothesis']
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    if "snli" not in excluded_data:
        snli_statements = [split.map(create_statements_labels_mnli, fn_kwargs={"label_column" : label_column, "templates":templates, "columns": columns}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        snli_statements = Dataset.from_dict(resample(snli_statements,n_samples=min(SPC, len(snli_statements)), replace=False, stratify=snli_statements['is_true']))
        train_datasets.append(snli_statements)

    dataset = ["AmazonScience/mintaka", 'en']
    templates = ["Q: ${1}, A: ${2}", "${1} ${2}", "Question: ${1}, Answer: ${2}", "The answer of ${1} is ${2}"]
    templates = random.sample(templates, k=min(len(templates),PPC))
    num_statements = 10000
    splits = ['train']
    label_column = 'answerText'
    question = 'question'
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    if "mintaka" not in excluded_data:
        mintaka_statements = [split.map(create_statements_labels_mintaka, fn_kwargs={"label_column" : label_column, "templates":templates, "question": question}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        mintaka_statements = Dataset.from_dict(resample(mintaka_statements,n_samples=min(SPC, len(mintaka_statements)), replace=False, stratify=mintaka_statements['is_true']))
        train_datasets.append(mintaka_statements)

    dataset = ["yelp_polarity"]
    templates = [["\"${1}\" has negative sentiment", "Statement: ${1}, Sentiment: Negative", "${1} It was terrible", "The sentiment in \"${1}\" is negative", "The emotions conveyed in \"${1}\" are negative"], ["\"${1}\" has positive sentiment", "Statement: ${1}, Sentiment: Positive", "${1} It was great", "The sentiment in \"${1}\" is positive", "The emotions conveyed in \"${1}\" are positive"]]
    templates = [random.sample(template, k=min(len(template),PPC)) for template in templates]
    num_statements = 10000
    split = ['train']
    label_column = 'label'
    columns = ['text']
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    
    if "yelp_polarity" not in excluded_data:
        yelp_statements = [split.map(create_statements_labels_yelp, fn_kwargs={"label_column":label_column, "templates":templates, "columns":columns}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        yelp_statements = Dataset.from_dict(resample(yelp_statements,n_samples=min(SPC, len(yelp_statements)), replace=False, stratify=yelp_statements['is_true']))
        train_datasets.append(yelp_statements)

    dataset = ["GEM/wiki_lingua", 'en']
    templates = ["Passage: ${1}, Summary: ${2}", "The summary of \"${1}\" is ${2}", "Context: ${1}, Summary: ${2}", "Q: Summarize the following: ${1}, A: ${2}", "The answer of \"Summarize the following ${1}\" is ${2}"]
    templates = random.sample(templates, k=min(len(templates),PPC))
    num_statements = 10000
    splits = ['train']
    label_column = 'target'
    question = 'source'
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    if "wikilingua" not in excluded_data:
        wikilingua_statements = [split.map(create_statements_labels_wikilingua, fn_kwargs={"label_column":label_column, "templates":templates, "question":question}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        wikilingua_statements = Dataset.from_dict(resample(wikilingua_statements,n_samples=min(SPC, len(wikilingua_statements)), replace=False, stratify=wikilingua_statements['is_true']))
        train_datasets.append(wikilingua_statements)

    dataset = ["squad"]
    templates = ["Context: ${1}\n Question: ${2}\n Answer: ${3}", "${1}\n According to the passage above, the answer of ${2} is ${3}", "Passage: ${1}\n Question: ${2}\n Answer: ${3}", "${1}\n Q: ${2}\n A:${3}"]
    templates = random.sample(templates, k=min(len(templates),PPC))
    num_statements = 10000
    splits = ['train']
    label_column = 'answers'
    question = 'question'
    context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    if "squad" not in excluded_data:
        squad_statements = [split.map(create_statements_labels_squad, fn_kwargs={"label_column": label_column, "templates":templates, "context":context, "question":question}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        squad_statements = Dataset.from_dict(resample(squad_statements,n_samples=min(SPC, len(squad_statements)), replace=False, stratify=squad_statements['is_true']))
        train_datasets.append(squad_statements)
    dataset = ["tweet_eval", "offensive"]
    templates = ["\"${1}\" The tweet is ${2}.", "This tweet \"${1}\" is considered ${2}.", "Tweet: \"${1}\". Label: ${2}.", "\"${1}\". This text is ${2}.", "The text \"${1}\" is ${2}."]
    templates = random.sample(templates, k=min(len(templates),PPC))
    num_statements = 10000
    splits = ['train']
    label_column = 'label'
    question = 'text'
    # context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    class_names = data[0].features['label'].names
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    if "offensive" not in excluded_data:
        offensive_statements = [split.map(create_statements_labels_offensive, fn_kwargs={"label_column": label_column, "templates":templates, "question":question, "class_names":class_names}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        offensive_statements = Dataset.from_dict(resample(offensive_statements,n_samples=min(SPC, len(offensive_statements)), replace=False, stratify=offensive_statements['is_true']))
        train_datasets.append(offensive_statements)
    dataset = ["AmazonScience/massive", "en-US"]
    templates = ["The utterance \"${1}\" is under the ${2} scenario.", "Utterance: \"${1}\"\nScenario: ${2}", "User: \"${1}\". The best scenario for the user query is ${2}.", "The scenario of user's utterance \"${1}\" is ${2}."]
    templates = random.sample(templates, k=min(len(templates),PPC))
    num_statements = 10000
    splits = ['train']
    label_column = 'scenario'
    question = 'utt'
    # context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    class_names = data[0].features['scenario'].names
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    if "massive" not in excluded_data:
        massive_statements = [split.map(create_statements_labels_massive, fn_kwargs={"label_column":label_column, "templates":templates, "question":question, "class_names":class_names}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        massive_statements = Dataset.from_dict(resample(massive_statements,n_samples=min(SPC, len(massive_statements)), replace=False, stratify=massive_statements['is_true']))
        train_datasets.append(massive_statements)

    dataset = ["definite_pronoun_resolution"]
    templates = ["${1} Based on the sentence, ${2} refers to ${3}.", "The pronoun ${2} in \"${1}\" is referring to ${3}.", "${1}\n'${2}' refers to ${3}."]
    templates = random.sample(templates, k=min(len(templates),PPC))
    num_statements = 10000
    splits = ['train']
    # label_column = 'label'
    # question = 'utt'
    # context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    # class_names = data[0].features['scenario'].names
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    if "dpr" not in excluded_data:
        dpr_statements = [split.map(create_statements_labels_dpr, fn_kwargs={"templates":templates}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        dpr_statements = Dataset.from_dict(resample(dpr_statements,n_samples=min(SPC, len(dpr_statements)), replace=False, stratify=dpr_statements['is_true']))
        train_datasets.append(dpr_statements)
        
    dataset = ["allenai/qasc"]
    template_1 = ["${1}. Answer: ${2}", "Q: \"${1}\".\nA: ${2}", "${1}.\nThe answer is ${2}"]
    template_2 = ["Question: \"${1}.\" Answer: ${2}"]
    template_3 = ["Context: ${1} Question: ${2} Answer: ${3}", "${2} Based on the passage \"${1}\", the answer if the question is \"${3}\".", "${1} ${2} ${3}"]
    template_4 = ["Context: ${1}\nQuestion: ${2}. \nAnswer: ${3}"]
    
    template_1 = random.sample(template_1, k=min(len(template_1),PPC))
    template_2 = random.sample(template_2, k=min(len(template_2),PPC))
    template_3 = random.sample(template_3, k=min(len(template_3),PPC))
    template_4 = random.sample(template_4, k=min(len(template_4),PPC))
    
    num_statements = 10000
    splits = ['train']
    # label_column = 'label'
    # question = 'utt'
    # context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    # class_names = data[0].features['scenario'].names
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    if "qasc" not in excluded_data:
        qasc_statements = [split.map(create_statements_labels_qasc, fn_kwargs={"template_1" : template_1, "template_2" : template_2, "template_3" : template_3, "template_4" : template_4}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        qasc_statements = Dataset.from_dict(resample(qasc_statements,n_samples=min(SPC, len(qasc_statements)), replace=False, stratify=qasc_statements['is_true']))
        train_datasets.append(qasc_statements)
    dataset = ["allenai/sciq"]
    template_1 = ["${1} ${2}", "Question: ${1}\nAnswer: ${2}"]
    template_2 = ["${1}\n\nQuestion: ${2}\nAnswer: ${3}", "${1}\n\nAccording to the information, ${2}.\nAnswer: ${3}.", "The answer to the question ${2}, according to \"${1}\" is ${3}."]
    
    template_1 = random.sample(template_1, k=min(len(template_1),PPC))
    template_2 = random.sample(template_2, k=min(len(template_2),PPC))
    
    num_statements = 10000
    splits = ['train']
    # label_column = 'label'
    # question = 'utt'
    # context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    # class_names = data[0].features['scenario'].names
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names

    if "sciq" not in excluded_data:
        sciq_statements = [split.map(create_statements_labels_sciq, fn_kwargs={"template_1" : template_1, "template_2" : template_2}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        sciq_statements = Dataset.from_dict(resample(sciq_statements,n_samples=min(SPC, len(sciq_statements)), replace=False, stratify=sciq_statements['is_true']))
        train_datasets.append(sciq_statements)
    dataset = ["race", "all"]
    template_1 = ["${1} ${2}"]
    
    num_statements = 10000
    splits = ['train']
    # label_column = 'label'
    # question = 'utt'
    # context = "context"
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    def filter_data(example):
        if " _" not in example["question"]:
            return False
        return True
    data = [d.filter(filter_data) for d in data]
    
    # class_names = data[0].features['scenario'].names
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    if "race" not in excluded_data:
        race_statements = [split.map(create_statements_labels_race, fn_kwargs={"template_1":template_1}, batched=True, remove_columns=col_names).to_pandas() for split in data][0]
        race_statements = Dataset.from_dict(resample(race_statements,n_samples=min(SPC, len(race_statements)), replace=False, stratify=race_statements['is_true']))
        train_datasets.append(race_statements)
        
    dataset = ["samsum"]
    templates = ["Passage: ${1}, Summary: ${2}", "The summary of \"${1}\" is ${2}", "Context: ${1}, Summary: ${2}", "Q: Summarize the following: ${1}, A: ${2}", "The answer of \"Summarize the following ${1}\" is ${2}"]
    num_statements = 10000
    splits = ['train']
    label_column = 'summary'
    question = 'dialogue'
    templates = random.sample(templates, k=min(len(templates),PPC))
    
    data = load_dataset(*dataset, split=splits, cache_dir=cache_dir, trust_remote_code=True)
    downsample = [True if num_statements<len(split) else False for split in data]
    new_data = []
    for i, split in enumerate(data):
        split = pd.DataFrame(split)
        if downsample[i]:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements, replace=False)))
        else:
            new_data.append(Dataset.from_pandas(resample(split, n_samples=num_statements)))
    data = new_data
    col_names = data[0].column_names
    if "samsum" not in excluded_data:
        samsum_statements = [split.map(create_statements_labels_samsum, fn_kwargs={"label_column":label_column, "templates": templates, "question":question}, batched=True, remove_columns=col_names ).to_pandas() for split in data][0]
        samsum_statements = Dataset.from_dict(resample(samsum_statements,n_samples=min(SPC, len(samsum_statements)), replace=False, stratify=samsum_statements['is_true']))
        train_datasets.append(samsum_statements)
    statement_data = {}
    statement_data['train'] = concatenate_datasets(train_datasets)
    return statement_data