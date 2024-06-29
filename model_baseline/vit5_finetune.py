import os
import warnings
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["inputs"], max_length=1024, truncation=True, padding=True
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["labels"], max_length=256, truncation=True, padding=True
        )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs


if __name__ == "__main__":
    train_dataset = pd.read_csv("../dataset/train_dataset_clean.csv")
    test_dataset = pd.read_csv("../dataset/test_dataset_clean.csv")
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")  
    dict_obj = {'inputs': ["summarization: " + t for t in train_dataset["context"].tolist()], 'labels': train_dataset["summarization"].tolist()}
    dataset = Dataset.from_dict(dict_obj)
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=10)
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    model.to('cuda')
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    training_args = Seq2SeqTrainingArguments("checkpoint_baseline/",
                                            do_train=True,
                                            do_eval=False,
                                            num_train_epochs=30,
                                            learning_rate=1e-5,
                                            warmup_ratio=0.05,
                                            weight_decay=0.01,
                                            per_device_train_batch_size=8,
                                            per_device_eval_batch_size=8,
                                            logging_dir='./log_baseline/',
                                            group_by_length=True,
                                            save_strategy="epoch",
                                            save_total_limit=3,
                                            report_to="none",
                                            #eval_steps=1,
                                            #evaluation_strategy="steps",
                                            # evaluation_strategy="no",
                                            fp16=True,
                                            )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )
    trainer.train()
