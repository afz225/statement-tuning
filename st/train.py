from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback, IntervalStrategy
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import evaluate
import wandb
import os
from datetime import datetime

class StatementDataset(torch.utils.data.Dataset):
    def __init__(self, statements, labels, tokenizer):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.statements[idx], truncation=True, padding=True)
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = int(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    predictions, labels = eval_pred
    # Decode generated summaries, which is in ids into text
    _, predictions = torch.max(torch.tensor(predictions), dim=1)
    return clf_metrics.compute(predictions=predictions, references=labels)

def train_st(model_name, tolerance, data, batch_size, lr, patience, n_epochs, context_len=514) -> Trainer:
    TRANSFORMER=model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = data['train'].filter(lambda example: example["is_true"] is not None).filter(lambda example: len(tokenizer(example['statement'])['input_ids']) < context_len+tolerance)
    train_statements, val_statements, train_labels, val_labels = train_test_split(train['statement'], train['is_true'], test_size=.1)
    
    train_dataset = StatementDataset(train_statements, train_labels, tokenizer)
    val_dataset = StatementDataset(val_statements, val_labels, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    current_time = datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')
    
    training_args = TrainingArguments(
        output_dir=f'./outputs/{TRANSFORMER}-{formatted_time}',          # output directory
        num_train_epochs=n_epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        warmup_steps=200,                # number of warmup steps for learning rate scheduler
        learning_rate=lr,
        weight_decay=0.01,               # strength of weight decay
        logging_dir=f'./logs/{TRANSFORMER}-{formatted_time}',            # directory for storing logs
        logging_steps=500,
        save_steps=500,
        eval_steps = 500,
        evaluation_strategy='steps',
        save_total_limit=2,
        load_best_model_at_end= True,
        metric_for_best_model='f1',
        report_to="wandb",
    )

    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,            # evaluation dataset
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    
    
    trainer.train()
    
    return tokenizer, trainer.model, trainer