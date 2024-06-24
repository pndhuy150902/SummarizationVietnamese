import warnings
import random
import pandas as pd
from datasets import Dataset, DatasetDict

warnings.filterwarnings('ignore')


def prepare_prompt(i, df):
    context = df.iloc[i]['article']
    summarization = df.iloc[i]['abstract']
    prompt = f"""<s>[INST] Bạn là một trợ lí AI tiếng Việt hữu ích. Bạn hãy tóm lược ngắn gọn nội dung chính của văn bản sau:
{context} [/INST] {summarization}</s>"""
    return prompt


def prepare_prompt_for_title(i, df):
    title = df.iloc[i]['title']
    context = df.iloc[i]['article']
    summarization = df.iloc[i]['abstract']
    prompt = f"""<s>[INST] Bạn là một trợ lí AI tiếng Việt hữu ích. Bạn hãy tóm lược ngắn gọn nội dung chính của văn bản sau, biết rằng tiêu đề của văn bản này là "{title}":
{context} [/INST] {summarization}</s>"""
    return prompt


def prepare_prompt_questions(i, df):
    context = df.iloc[i]['context']
    summarization = df.iloc[i]['summarization']
    question = df.iloc[i]['questions']
    answer = df.iloc[i]['answers']
    prompt = f"""<s>[INST] Bạn là một trợ lí AI. Bạn được giao một nhiệm vụ là tóm tắt nội dung sau bằng tiếng Việt, dựa vào tất cả các thông tin dưới đây:
Nội dung cần tóm tắt: {context}
Câu hỏi: {question}
Câu trả lời: {answer}
Hãy đảm bảo rằng nội dung tóm tắt của bạn phải nắm bắt được thông tin quan trọng từ nội dung cần tóm tắt và bỏ qua những thông tin không liên quan. [/INST] {summarization}</s>"""
    return prompt


def prepare_prompt_for_title_questions(i, df):
    title = df.iloc[i]['title']
    context = df.iloc[i]['context']
    summarization = df.iloc[i]['summarization']
    question = df.iloc[i]['questions']
    answer = df.iloc[i]['answers']
    prompt = f"""<s>[INST] Bạn là một trợ lí AI. Bạn được giao một nhiệm vụ là tóm tắt nội dung sau bằng tiếng Việt, dựa vào tất cả các thông tin dưới đây:
Nội dung cần tóm tắt: {context}
Tiêu đề: {title}
Câu hỏi: {question}
Câu trả lời: {answer}
Hãy đảm bảo rằng nội dung tóm tắt của bạn phải nắm bắt được thông tin quan trọng từ nội dung cần tóm tắt và bỏ qua những thông tin không liên quan. [/INST] {summarization}</s>"""
    return prompt


def read_dataset(config):
    train_data_no_title = pd.read_csv(config.processed_data.train_data)
    # valid_data = pd.read_csv(config.processed_data.valid_data)
    test_data = pd.read_csv(config.processed_data.test_data)
    # train_data_with_title = pd.read_csv(config.processed_data.train_data_with_title)
    return train_data_no_title, test_data


def prepare_dataset(config):
    random.seed(42)
    train_data_no_title, test_data = read_dataset(config)
    train_prompts_no_title = [prepare_prompt(i, train_data_no_title) for i in range(len(train_data_no_title))]
    # train_prompts_with_title = [prepare_prompt_for_title(i, train_data_with_title) for i in range(len(train_data_with_title))]
    test_prompts = [prepare_prompt(i, test_data) for i in range(len(test_data))]
    # train_prompts = train_prompts_with_title + train_prompts_no_title
    test_prompts = test_prompts[11000:15000]
    random.shuffle(test_prompts)
    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': test_prompts}),
        'test': Dataset.from_dict({'text': test_prompts})
    })
    return dataset
