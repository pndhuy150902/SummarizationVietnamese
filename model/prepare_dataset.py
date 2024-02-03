import warnings
import pandas as pd
from configuration import prepare_tokenizer
from datasets import Dataset, DatasetDict

warnings.filterwarnings('ignore')


def prepare_prompt(i, df):
    context = df.iloc[i]['context']
    summarization = df.iloc[i]['summarization']
    prompt = f"""<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm tắt ngắn gọn nội dung sau bằng tiếng Việt:
{context} [/INST] {summarization}</s>"""
    return prompt


def read_dataset(config):
    train_data = pd.read_csv(config.processed_data.train_data)
    valid_data = pd.read_csv(config.processed_data.valid_data)
    test_data = pd.read_csv(config.processed_data.test_data)
    return train_data, valid_data, test_data


def generate_and_tokenize_prompt(data_point, tokenizer, max_length):
    tokenized_full_prompt = tokenizer(data_point['text'], padding=True, truncation=True, max_length=max_length)
    return tokenized_full_prompt


def prepare_dataset(config):
    train_data, valid_data, test_data = read_dataset(config)
    train_prompts = [prepare_prompt(i, train_data) for i in range(len(train_data))]
    valid_prompts = [prepare_prompt(i, valid_data) for i in range(len(valid_data))]
    test_prompts = [prepare_prompt(i, test_data) for i in range(len(test_data))]
    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': train_prompts}),
        'valid': Dataset.from_dict({'text': valid_prompts}),
        'test': Dataset.from_dict({'text': test_prompts})
    })
    tokenizer = prepare_tokenizer(config.model_name)
    dataset = dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, config.length.text), batched=True)
    return dataset
