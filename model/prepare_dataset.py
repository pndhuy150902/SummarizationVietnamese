import warnings
import pandas as pd
from datasets import Dataset, DatasetDict

warnings.filterwarnings('ignore')


def prepare_prompt(i, df):
    context = df.iloc[i]['context']
    output = df.iloc[i]['summarization']
    prompt = f"""
Nội dung văn bản:
{context}

[INST]Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm tắt ngắn gọn nội dung của văn bản trên[/INST]
Kết quả: {output} </s>"""
    return prompt


def prepare_dataset():
    data_summarization = {
        'context': [],
        'summarization': []
    }
