from datasets import load_dataset, get_dataset_config_names, Dataset
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
import evaluate
import random
import copy
from sklearn.utils import resample
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from datetime import datetime


def shuffle_words(s):
    # Split the string into a list of words
    words = s.split()
    # Use random.shuffle to shuffle the list of words
    random.shuffle(words)
    # Join the shuffled list back into a string
    return ' '.join(words)

def fill_template(templates, values):
    temp = random.sample(templates,1)[0]
    for i in range(len(values)):
        if shuffle:
            temp = temp.replace("${"+str(i+1)+"}", shuffle_words(values[i]))
        else:
            temp = temp.replace("${"+str(i+1)+"}", values[i])
    return temp

def measure_time(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        end_time = time.time()
        return (end_time-start_time), out
    return inner
        

def train_few_shot_two_statements(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8, patience=5):
    nshot_model = copy.deepcopy(model)
    optimizer = AdamW(nshot_model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    loss_fn = CrossEntropyLoss()
    loss_values = []
    best_val_loss = math.inf
    best_model = None
    print('total steps per epoch: ',  len(train_dataloader) / batch_size)
    for epoch_i in range(0, epochs):
        
        print('training on epoch: ', epoch_i)
        # set start time 
        t0 = time.time()
        # reset total loss
        total_loss = 0
        total_val_loss = 0
        # model in training 
        model.train()
        # loop through batch 
        for step, batch in enumerate(train_dataloader):
            # Progress update every 10 steps
            if step % 10 == 0 and not step == 0:
                total_val_loss = 0
                for step, batch in enumerate(test_dataloader):
                    nshot_model.eval()
                    tok1 = tokenizer(batch['statement1'], return_tensors='pt', padding=True).to(device)
                    tok2 = tokenizer(batch['statement2'], return_tensors='pt', padding=True).to(device)
                    labels = batch['label'].to(device)
                    # get outputs
                    prob1 = nshot_model(input_ids=tok1['input_ids'], attention_mask=tok1['attention_mask']).logits[:,1]
                    prob2 = nshot_model(input_ids=tok2['input_ids'], attention_mask=tok2['attention_mask']).logits[:,1]
                    preds = F.softmax(torch.stack([prob1, prob2],dim=-1),dim=-1).to(device)
                    # get loss
                    loss = loss_fn(preds, labels)
                    total_val_loss += loss.item()
                del tok1
                del tok2
                del labels
                del prob1
                del prob2
                del preds
                torch.cuda.empty_cache()
                avg_val_loss = total_val_loss/len(test_dataloader)
                print("average val loss: {0:.2f}".format(avg_val_loss))
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = copy.deepcopy(nshot_model)
                    
            nshot_model.train()
            tok1 = tokenizer(batch['statement1'], return_tensors='pt', padding=True).to(device)
            tok2 = tokenizer(batch['statement2'], return_tensors='pt', padding=True).to(device)
            labels = batch['label'].to(device)
            # clear any previously calculated gradients 
            nshot_model.zero_grad()
            # get outputs
            prob1 = nshot_model(input_ids=tok1['input_ids'], attention_mask=tok1['attention_mask']).logits[:,1]
            prob2 = nshot_model(input_ids=tok2['input_ids'], attention_mask=tok2['attention_mask']).logits[:,1]
            preds = F.softmax(torch.stack([prob1, prob2],dim=-1),dim=-1).to(device)
            # get loss
            loss = loss_fn(preds, labels)
    
    #         loss = loss_fn(prob1, pred1) + loss_fn(prob2, pred2)
            loss.backward()
            # total loss
            total_loss += loss.item()
            # clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(nshot_model.parameters(), 1.0)
            # update optimizer
            optimizer.step()
            # update learning rate 
            scheduler.step()
            del tok1
            del tok2
            del labels
            del prob1
            del prob2
            del preds
            torch.cuda.empty_cache()
            
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("average training loss: {0:.2f}".format(avg_train_loss))
    
    nshot_model = best_model
    return nshot_model

def train_few_shot_one_statement(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8, patience=5):
    nshot_model = copy.deepcopy(model)
    optimizer = AdamW(nshot_model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    loss_fn = CrossEntropyLoss()
    loss_values = []
    best_val_loss = math.inf
    best_model = None
    print('total steps per epoch: ',  len(train_dataloader) / batch_size)
    for epoch_i in range(0, epochs):
        
        print('training on epoch: ', epoch_i)
        # set start time 
        t0 = time.time()
        # reset total loss
        total_loss = 0
        total_val_loss = 0
        # model in training 
        model.train()
        # loop through batch 
        for step, batch in enumerate(train_dataloader):
            # Progress update every 10 steps
            if step % 10 == 0 and not step == 0:
                total_val_loss = 0
                for step, batch in enumerate(test_dataloader):
                    nshot_model.eval()
                    tok = tokenizer(batch['statement'], return_tensors='pt', padding=True).to(device)
                    labels = batch['label'].to(device)
                    # clear any previously calculated gradients 
                    # get outputs
                    preds = nshot_model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits
                    # get loss
                    loss = loss_fn(preds, labels)
                    total_val_loss += loss.item()
                del tok
                del labels
                del preds
                torch.cuda.empty_cache()
                avg_val_loss = total_val_loss/len(test_dataloader)
                print("average val loss: {0:.2f}".format(avg_val_loss))
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = copy.deepcopy(nshot_model)
                    
            nshot_model.train()
            tok = tokenizer(batch['statement'], return_tensors='pt', padding=True).to(device)
            labels = batch['label'].to(device)
            # clear any previously calculated gradients 
            nshot_model.zero_grad()
            # get outputs
            preds = nshot_model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits
            # get loss
            loss = loss_fn(preds, labels)
    
    #         loss = loss_fn(prob1, pred1) + loss_fn(prob2, pred2)
            loss.backward()
            # total loss
            total_loss += loss.item()
            # clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(nshot_model.parameters(), 1.0)
            # update optimizer
            optimizer.step()
            # update learning rate 
            scheduler.step()
            del tok
            del labels
            del preds
            torch.cuda.empty_cache()
            
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("average training loss: {0:.2f}".format(avg_train_loss))
    
    nshot_model = best_model
    return nshot_model

@measure_time
def compute_accuracy_two_statements(test_dataloader, label_column, model, tokenizer, device, clf_metrics):
    predictions = []
    actual_labels = []
    for batch in test_dataloader:
        tok1 = tokenizer(batch['statement1'], return_tensors='pt', padding=True).to(device)
        tok2 = tokenizer(batch['statement2'], return_tensors='pt', padding=True).to(device)
        labels = batch[label_column]
        prob1 = F.softmax(model(input_ids=tok1['input_ids'], attention_mask=tok1['attention_mask']).logits, dim=-1)[:,1]
        prob2 = F.softmax(model(input_ids=tok2['input_ids'], attention_mask=tok2['attention_mask']).logits, dim=-1)[:,1]
        preds = torch.argmax(torch.stack([prob1, prob2],dim=-1),dim=-1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    return clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy']
@measure_time
def compute_accuracy_one_statement(test_dataloader, model, tokenizer, device, clf_metrics):
    predictions = []
    actual_labels = []
    for batch in test_dataloader:
        tok = tokenizer(batch['statement'], return_tensors='pt', padding=True).to(device)
        labels = batch['label']
        _, preds = torch.max(model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits, dim=-1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    del tok
    del labels
    del preds
    torch.cuda.empty_cache()
    return clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy']
@measure_time
def compute_accuracy_yahoo(model,tokenizer, device,temp, cache_dir, clf_metrics):
    def create_statements_labels_yahooanswers(example):
        for i, topic in enumerate(topics):
            example['statement'+str(i)] = fill_template([temp], [example['question_title'], example['question_content'], topic])
    
        example['label'] = example[label_column]
        return example
    dataset = ["yahoo_answers_topics"]
    templates = ["\"${1} ${2}\". the topic is ${3}", "Text: \"${1} ${2}\". Topic: ${3}", "This text: \"${1} ${2}\" is about ${3}"]
    split = ['test']
    data = load_dataset(*dataset, split=split, cache_dir=cache_dir)
    topics = data[0].features['topic'].names
    data = [Dataset.from_dict(data[0][:1000])]
    label_column = 'topic'
    col_names = copy.copy(data[0].column_names)
    yahooanswers_statements = [split.map(create_statements_labels_yahooanswers, remove_columns=col_names) for split in data][0]
    yahooanswers_dataloader = DataLoader(yahooanswers_statements, batch_size=2, shuffle=False)
    predictions = []
    actual_labels = []
    for batch in yahooanswers_dataloader:
        torch.cuda.empty_cache()
        toks = [tokenizer(batch['statement'+str(i)], return_tensors='pt', padding=True, truncation=True).to(device) for i in range(len(topics))]
        labels = batch['label']
        probs = [F.softmax(model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits, dim=-1)[:,1] for tok in toks]
        preds = torch.argmax(torch.stack(probs,dim=-1),dim=-1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    del toks
    del labels
    del probs
    del preds
    torch.cuda.empty_cache()
    return clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy']
@measure_time
def compute_accuracy_emotion(model,tokenizer, device,temp, cache_dir, clf_metrics):
    dataset = ["dair-ai/emotion"]
    templates = ["\"${1}\". The emotion conveyed in the text is ${2}", "Text: \"${1}\". Emotion: ${2}", "The text \"${1}\" is ${2}"]
    split = ['test']
    data = load_dataset(*dataset, split=split, cache_dir=cache_dir)
    emotions = data[0].features['label'].names
    label_column = 'label'
    col_names = copy.copy(data[0].column_names)
    col_names.remove(label_column)
    def create_statements_labels_emotion(example):
        for i, emotion in enumerate(emotions):
            example['statement'+str(i)] = fill_template([temp], [example['text'], emotion])
        return example
    
    emotion_statements = [split.map(create_statements_labels_emotion, remove_columns=col_names) for split in data][0]
    emotion_dataloader = DataLoader(emotion_statements, batch_size=2, shuffle=False)
    predictions = []
    actual_labels = []
    for batch in emotion_dataloader:
        torch.cuda.empty_cache()
        toks = [tokenizer(batch['statement'+str(i)], return_tensors='pt', padding=True, truncation=True).to(device) for i in range(len(emotions))]
        labels = batch['label']
        probs = [F.softmax(model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).logits, dim=-1)[:,1] for tok in toks]
        preds = torch.argmax(torch.stack(probs,dim=-1),dim=-1)
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    del toks
    del labels
    del preds
    del probs
    torch.cuda.empty_cache()
    return clf_metrics.compute(predictions=predictions, references=actual_labels)['accuracy']
    
def run_eval(tokenizer, model, batch_size, cache_dir, shuffled=False, seed=42, n_shot='full'):
    
    data_accuracies = {}
    speed = {}

    global shuffle
    shuffle=shuffled
    
    dataset = "pkavumba/balanced-copa"
    templates = [["The cause of \"${1}\" is that \"${2}\"", "\"${1}\" because \"${2}\"", "\"${1}\" due to \"${2}\""], ["The effect of \"${1}\" is that \"${2}\"", "\"${1}\" therefore \"${2}\"", "\"${1}\", so \"${2}\""]]
    split = ['train', 'test']
    label_column = 'label'
    question = 'premise'
    choices = ['choice1', 'choice2']
    data = load_dataset(dataset, split=split, cache_dir=cache_dir)
    col_names = copy.copy(data[0].column_names)
    col_names.remove(label_column)
    temps = [random.choice(templates[0]), random.choice(templates[1])]
    def create_statements_labels_copa(example):
        temp = temps[0] if example['question'] == 'cause' else temps[1]
        # temp = random.choice(template)
        example['statement1'] = fill_template([temp], [example[question], example[choices[0]]])
        example['statement2'] = fill_template([temp], [example[question], example[choices[1]]])
        return example
    
    copa_statements = [split.map(create_statements_labels_copa, remove_columns=col_names) for split in data]

    train = copa_statements[0]
    if n_shot > 0 and n_shot != 'full':
        train = Dataset.from_dict(train_test_split(train.to_pandas(), train_size=min(n_shot, len(train)-2), stratify=train['label'])[0], random_state=seed)
    test = copa_statements[1]

    n_examples = len(test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    current_time = datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')

    clf_metrics = evaluate.load('accuracy', experiment_id=formatted_time)

    if n_shot != 0:
        train_dataloader = DataLoader(train, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    eval_model = model if n_shot == 0 else train_few_shot_two_statements(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
    eval_model.to(device)
    run_time, data_accuracies['copa'] = compute_accuracy_two_statements(test_dataloader, label_column, eval_model,tokenizer, device,clf_metrics)
    speed['copa'] = n_examples/run_time
    dataset = ["SetFit/mrpc"]
    split=['train', 'test']
    templates = ["\"${1}\" is a paraphrase of \"${2}\"", "\"${1}\"\n In other words: \"${2}\"", "${1}? yes, ${2}",  "\"${1}\" can be stated as \"${2}\"", "\"${1}\" is the same as saying \"${2}\""]
    columns = ['text1', 'text2']
    label_column = 'label'
    
    data = load_dataset(*dataset, split=split)
    col_names = copy.copy(data[0].column_names)
    col_names.remove(label_column)
    temp = random.choice(templates)
    def create_statements_labels_mrpc(example):
        example['statement'] = fill_template([temp], [example[columns[0]], example[columns[1]]])
        return example
    
    mrpc_statements = [split.map(create_statements_labels_mrpc, remove_columns=col_names) for split in data]

    train = mrpc_statements[0]
    if n_shot > 0 and n_shot != 'full':
        train = Dataset.from_dict(train_test_split(train.to_pandas(), train_size=min(n_shot, len(train)-2), stratify=train['label'], random_state=seed)[0])
    test = mrpc_statements[1]
    n_examples = len(test)
    
    
    if n_shot != 0:
        train_dataloader = DataLoader(train, batch_size=batch_size)
        
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)


    eval_model = model if n_shot == 0 else train_few_shot_one_statement(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
    eval_model.to(device)
    run_time, data_accuracies['mrpc'] = compute_accuracy_one_statement(test_dataloader, eval_model, tokenizer, device, clf_metrics)
    speed['mrpc'] = n_examples/run_time

    dataset = ["amazon_polarity"]
    templates = [["\"Title: ${1}, Content: ${2}\" has negative sentiment", "${1} ${2} has negative sentiment", "\"Title: ${1}, Content: ${2}\", Sentiment: Negative", "${1} ${2} It was terrible", "The sentiment in \"${1} ${2}\" is negative",  "The emotions conveyed in \"${1} ${2}\" are negative"], ["\"Title: ${1}, Content: ${2}\" has positive sentiment", "${1} ${2} has positive sentiment", "\"Title: ${1}, Content: ${2}\", Sentiment: Positive", "${1} ${2} It was great", "The sentiment in \"${1} ${2}\" is positive",  "The emotions conveyed in \"${1} ${2}\" are positive"]]
    splits = ['train','test']
    label_column = 'label'
    columns = ['title','content']
    sizes = [3000, 1000]
    
    data = [Dataset.from_dict(load_dataset(*dataset, split=split, cache_dir=cache_dir)[:sizes[i]]) for i, split in enumerate(splits)]
    # data = load_dataset(*dataset, split=split, cache_dir='/scratch/afz225/.cache')
    col_names = copy.copy(data[0].column_names)
    col_names.remove(label_column)
    negative_template = random.choice(templates[0])
    positive_template = random.choice(templates[1])
    def create_statements_labels_amazon(example):
        example['statement1'] = fill_template([negative_template], [example[text] for text in columns])
        example['statement2'] = fill_template([positive_template], [example[text] for text in columns])
        return example
    
    amazon_statements = [split.map(create_statements_labels_amazon, remove_columns=col_names) for split in data]
    
    
    train = amazon_statements[0]
    if n_shot > 0 and n_shot != 'full':
        train = Dataset.from_dict(train_test_split(train.to_pandas(), train_size=min(n_shot, len(train)-2), stratify=train['label'], random_state=seed)[0])
    test = amazon_statements[1]
    n_examples = len(test)
    
    
    if n_shot != 0:
        train_dataloader = DataLoader(train, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    eval_model = model if n_shot == 0 else train_few_shot_two_statements(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
    eval_model.to(device)
    run_time, data_accuracies['amazon_p'] = compute_accuracy_two_statements(test_dataloader, label_column, eval_model,tokenizer, device, clf_metrics)
    speed['amazon_p'] = n_examples/run_time
    dataset = ["nightingal3/fig-qa"]
    templates = ["${1} ${2}", "${1} therefore ${2}", "startphrase: ${1}, ending: ${2}", "if ${1} then ${2}", "${1} means ${2}"]
    split = ['train', 'validation']
    label_column = 'labels'
    question = 'startphrase'
    choices = ['ending1', 'ending2']
    
    data = load_dataset(*dataset, split=split, cache_dir=cache_dir)
    col_names = copy.copy(data[0].column_names)
    col_names.remove(label_column)
    temp = random.choice(templates)
    def create_statements_labels_figqa(example):
        example['statement1'] = fill_template([temp], [example[question], example[choices[0]]])
        example['statement2'] = fill_template([temp], [example[question], example[choices[1]]])
        return example
        
    figqa_statements = [split.map(create_statements_labels_figqa, remove_columns=col_names) for split in data]

    train = figqa_statements[0]
    if n_shot > 0 and n_shot != 'full':
        train = Dataset.from_dict(train_test_split(train.to_pandas(), train_size=min(n_shot, len(train)-2), stratify=train['label'], random_state=seed)[0])
    test = figqa_statements[1]
    n_examples = len(test)
    
    
    if n_shot != 0:
        train_dataloader = DataLoader(train, batch_size=batch_size)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    eval_model = model if n_shot == 0 else train_few_shot_two_statements(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
    eval_model.to(device)
    run_time, data_accuracies['figqa'] = compute_accuracy_two_statements(test_dataloader, label_column, eval_model,tokenizer, device, clf_metrics)
    speed['figqa'] = n_examples/run_time

    if n_shot != 'full' and n_shot <= 32:
        dataset = ["gimmaru/story_cloze-2016"]
        templates = ["${1} ${2} ${3} ${4} ${5}", "story: ${1} ${2} ${3} ${4} ending: ${5}", "story: \"${1} ${2} ${3} ${4}\" ending: \"${5}\""]
        split = ['test']
        label_column = 'answer_right_ending'
        question = 'startphrase'
        choices = ['sentence_quiz1', 'sentence_quiz2']
        
        data = load_dataset(*dataset, split=split, cache_dir=cache_dir)
        col_names = copy.copy(data[0].column_names)
        temp = random.choice(templates)
        # col_names.remove(label_column)
        def create_statements_labels_storycloze(example):
            example['statement1'] = fill_template([temp], [example['input_sentence_1'], example['input_sentence_2'], example['input_sentence_3'], example['input_sentence_4'], example[choices[0]]])
            example['statement2'] = fill_template([temp], [example['input_sentence_1'], example['input_sentence_2'], example['input_sentence_3'], example['input_sentence_4'], example[choices[1]]])
            example['label'] = example[label_column]-1
            return example
        storycloze_statements = [split.map(create_statements_labels_storycloze, remove_columns=col_names) for split in data][0]
        train, test = train_test_split(storycloze_statements.to_pandas(), train_size=32, stratify=storycloze_statements['label'], random_state=seed)
        train = Dataset.from_dict(train)
        test = Dataset.from_dict(test)
        n_examples = len(test)
    
        if n_shot != 0:
            train_dataloader = DataLoader(train, batch_size=batch_size)
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
        eval_model = model if n_shot == 0 else train_few_shot_two_statements(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
        eval_model.to(device)
        label_column = 'label'
        run_time, data_accuracies['storycloze'] = compute_accuracy_two_statements(test_dataloader,label_column, eval_model,tokenizer, device, clf_metrics)
        speed['storycloze'] = n_examples/run_time
    if n_shot != "full" and n_shot <= 200:
        yahoo_dataset = load_dataset('yahoo_answers_topics', cache_dir=cache_dir)
        templates = ["\"${1} ${2}\". the topic is ${3}", "Text: \"${1} ${2}\". Topic: ${3}", "This text: \"${1} ${2}\" is about ${3}"]
        topics = yahoo_dataset['train'].features['topic'].names
        temp = random.choice(templates)
        def create_statements_labels_yahoo(examples):
            statements = [fill_template([temp], [examples['question_title'][i], examples['question_content'][i], topic]) for i in range(len(examples['topic'])) for topic in topics]
            labels = [int(i == label) for label in examples['topic'] for i, topic in enumerate(topics)]
            examples = {}
            examples['statement'] = statements
            examples['label'] = labels
            return examples

        if n_shot > 0:
            train = yahoo_dataset['train']
            train = Dataset.from_dict(train_test_split(train.to_pandas(), train_size=min(n_shot, len(train)-2), stratify=train['topic'])[0], random_state=seed)
            train = train.map(create_statements_labels_yahoo, remove_columns=['topic', 'question_title', 'question_content', 'id', 'best_answer'],batched=True).filter(lambda example: len(tokenizer(example['statement'])['input_ids']) < 514)
        test = yahoo_dataset['test']
        test = Dataset.from_dict(train_test_split(test.to_pandas(), train_size=min(2000, len(test)-2), stratify=test['topic'])[0], random_state=seed)

        test = test.map(create_statements_labels_yahoo, remove_columns=['topic', 'question_title', 'question_content', 'id', 'best_answer'],batched=True).filter(lambda example: len(tokenizer(example['statement'])['input_ids']) < 514)
        n_examples = len(test)
    
        if n_shot != 0:
            train_dataloader = DataLoader(train, batch_size=batch_size//8)
        test_dataloader = DataLoader(test, batch_size=batch_size//8, shuffle=False)
        eval_model.to(device)
        eval_model = model if n_shot == 0 else train_few_shot_one_statement(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
        
        run_time, data_accuracies['yahoo'] = compute_accuracy_yahoo(eval_model,tokenizer, device, temp, cache_dir, clf_metrics)
        speed['yahoo'] = n_examples/run_time
    if n_shot != "full" and n_shot <= 200:
        emotion_dataset = load_dataset('dair-ai/emotion', 'split', cache_dir=cache_dir)
        templates = ["\"${1}\". The emotion conveyed in the text is ${2}", "Text: \"${1}\". Emotion: ${2}", "The text \"${1}\" is ${2}"]
        topics = emotion_dataset['train'].features['label'].names
        temp = random.choice(templates)
        def create_statements_labels_emotion(examples):
            statements = [fill_template([temp], [examples['text'][i], topic]) for i in range(len(examples['label'])) for topic in topics]
            labels = [int(i == label) for label in examples['label'] for i, topic in enumerate(topics)]
            examples = {}
            examples['statement'] = statements
            examples['label'] = labels
            return examples

        if n_shot > 0:
            train = emotion_dataset['train']
            train = Dataset.from_dict(train_test_split(train.to_pandas(), train_size=min(n_shot, len(train)-2), stratify=train['label'])[0], random_state=seed)
            train = train.map(create_statements_labels_emotion, remove_columns=['text'],batched=True).filter(lambda example: len(tokenizer(example['statement'])['input_ids']) < 514)
        test = emotion_dataset['test']
        test = Dataset.from_dict(train_test_split(test.to_pandas(), train_size=min(2000, len(test)-6), stratify=test['label'])[0], random_state=seed)

        test = test.map(create_statements_labels_emotion, remove_columns=['text'],batched=True).filter(lambda example: len(tokenizer(example['statement'])['input_ids']) < 514)
        n_examples = len(test)
    
        if n_shot != 0:
            train_dataloader = DataLoader(train, batch_size=batch_size//8)
        test_dataloader = DataLoader(test, batch_size=batch_size//8, shuffle=False)
        eval_model.to(device)
        eval_model = model if n_shot == 0 else train_few_shot_one_statement(train_dataloader, test_dataloader, model,tokenizer, device, epochs = 8, batch_size = 8)
        
        run_time, data_accuracies['emotion'] = compute_accuracy_emotion(eval_model, tokenizer, device, temp, cache_dir, clf_metrics)
        speed['emotion'] = n_examples/run_time
        return speed, data_accuracies
        