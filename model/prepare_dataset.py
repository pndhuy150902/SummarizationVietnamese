import warnings
import pandas as pd
from configuration import prepare_tokenizer
from datasets import Dataset, DatasetDict

warnings.filterwarnings('ignore')


def prepare_prompt(i, df):
    context = df.iloc[i]['context']
    summarization = df.iloc[i]['summarization']
    prompt = f"""<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm lược ngắn gọn nội dung sau bằng tiếng Việt:
{context} [/INST] {summarization}</s>"""
    return prompt


def prepare_prompt_for_title(i, df):
    title = df.iloc[i]['title']
    context = df.iloc[i]['context']
    summarization = df.iloc[i]['summarization']
    prompt = f"""<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm lược ngắn gọn nội dung sau bằng tiếng Việt biết rằng tiêu đề của nội dung là "{title}":
{context} [/INST] {summarization}</s>"""
    return prompt


def read_dataset(config):
    train_data = pd.read_csv(config.processed_data.train_data)
    valid_data = pd.read_csv(config.processed_data.valid_data)
    test_data = pd.read_csv(config.processed_data.test_data)
    return train_data, valid_data, test_data


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
    return dataset
