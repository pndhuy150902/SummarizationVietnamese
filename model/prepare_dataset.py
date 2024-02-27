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
    train_data_no_title = pd.read_csv(config.processed_data.train_data)
    # valid_data = pd.read_csv(config.processed_data.valid_data)
    test_data = pd.read_csv(config.processed_data.test_data)
    train_data_with_title = pd.read_csv(config.processed_data.train_data_with_title)
    return train_data_no_title, train_data_with_title, test_data


def prepare_dataset(config):
    train_data_no_title, train_data_with_title, test_data = read_dataset(config)
    train_prompts_no_title = [prepare_prompt(i, train_data_no_title) for i in range(len(train_data_no_title))]
    train_prompts_with_title = [prepare_prompt_for_title(i, train_data_with_title) for i in range(len(train_data_with_title))]
    test_prompts = [prepare_prompt(i, test_data) for i in range(len(test_data))]
    train_prompts = train_prompts_with_title.extend(train_prompts_no_title)
    train_prompts = sorted(train_prompts, key=len, reverse=False)
    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': train_prompts}),
        'test': Dataset.from_dict({'text': test_prompts})
    })
    return dataset
