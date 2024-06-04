import warnings
import random
import pandas as pd
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
    train_data_with_title = pd.read_csv(config.processed_data.train_data_with_title)
    return train_data_no_title, train_data_with_title, test_data


def prepare_dataset(config):
    random.seed(42)
    train_data_no_title, train_data_with_title, test_data = read_dataset(config)
    train_prompts_no_title = [prepare_prompt(i, train_data_no_title) for i in range(len(train_data_no_title))]
    train_prompts_with_title = [prepare_prompt_for_title(i, train_data_with_title) for i in range(len(train_data_with_title))]
    test_prompts = [prepare_prompt(i, test_data) for i in range(len(test_data))]
    train_prompts = train_prompts_with_title[:40] + train_prompts_no_title[:40]
    train_prompts = random.shuffle(train_prompts)
    dataset = DatasetDict({
        'train': Dataset.from_dict({'text': train_prompts}),
        'test': Dataset.from_dict({'text': test_prompts})
    })
    return dataset
