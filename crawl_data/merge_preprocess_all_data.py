import warnings
import re
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def read_and_merge_data_crawled():
    news_thanhnien = pd.read_csv('../dataset/news_crawled_data/crawled_data_thanhnien.csv')
    news_tuoitre = pd.read_csv('../dataset/news_crawled_data/crawled_data_tuoitre.csv')
    news_dantri = pd.read_csv('../dataset/news_crawled_data/crawled_data_dantri.csv')
    full_news_crawled = pd.concat([news_thanhnien, news_tuoitre, news_dantri], axis=0)
    full_news_crawled = full_news_crawled[~(full_news_crawled['context'] == '')]
    full_news_crawled = full_news_crawled[~(full_news_crawled['summarization'] == '')]
    full_news_crawled.dropna(inplace=True)
    full_news_crawled.reset_index(inplace=True, drop=True)
    return full_news_crawled


def read_data_vslp():
    structure_data = {
        'context': [],
        'summarization': []
    }
    vlsp_path = {
        'vlsp_train': '../dataset/vlsp_data/vlsp_2022_abmusu_train_data.jsonl',
        'vlsp_val': '../dataset/vlsp_data/vlsp_2022_abmusu_validation_data.jsonl',
        'vlsp_test': '../dataset/vlsp_data/vlsp_abmusu_test_data.jsonl'
    }
    for key in vlsp_path.keys():
        vlsp_data = pd.read_json(vlsp_path[key], lines=True)
        for docs in vlsp_data["single_documents"]:
            for doc in docs:
                structure_doc = dict(doc)
                structure_data['context'].append(structure_doc["raw_text"])
                structure_data['summarization'].append(structure_doc["anchor_text"])
    vlsp_data = pd.DataFrame(structure_data)
    vlsp_data = vlsp_data[~(vlsp_data['context'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['summarization'] == '')]
    vlsp_data.dropna(inplace=True)
    vlsp_data.reset_index(inplace=True, drop=True)
    return vlsp_data


def read_data_vietgpt():
    train_vietgpt_data = pd.read_parquet('../dataset/news_vietgpt_data/train_news_summarization_vi_old_vietgpt.parquet')
    test_vietgpt_data = pd.read_parquet('../dataset/news_vietgpt_data/test_news_summarization_vi_old_vietgpt.parquet')
    vietgpt_data = pd.concat([train_vietgpt_data, test_vietgpt_data], axis=0)
    vietgpt_data = vietgpt_data[~(vietgpt_data['content'] == '')]
    vietgpt_data = vietgpt_data[~(vietgpt_data['summary'] == '')]
    vietgpt_data.dropna(inplace=True)
    vietgpt_data.reset_index(inplace=True, drop=True)
    vietgpt_data.rename(columns={'content': 'context', 'summary': 'summarization'}, inplace=True)
    return vietgpt_data


def preprocessing_data(df):
    df['context'] = df['context'].apply(lambda x: re.sub(r'\... ...', ', ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\>> ', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\Ảnh: [\w+\s+\/+]+.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\Ảnh [\w+\s+\/+]+.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\Ảnh: [\w+\.]+.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\Ảnh [\w+\.]+.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\Ảnh: [\w+\s+\-\w+]+.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\Ảnh [\w+\s+\-\w+]+.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\s+\/+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\s+\/+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\.]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\.]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\s+\-\w+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\s+\-\w+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\s+\/+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\s+\/+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\.]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\.]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\s+\-\w+]+\).', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\s+\-\w+]+\).', '', x))
    return df


def merge_and_preprocess_and_split_all_data():
    crawled_data = read_and_merge_data_crawled()
    vlsp_data = read_data_vslp()
    vietgpt_data = read_data_vietgpt()
    full_data = pd.concat([crawled_data, vlsp_data, vietgpt_data], axis=0)
    full_data = full_data.sample(frac=1, random_state=42)
    full_data.drop_duplicates(inplace=True)
    full_data.reset_index(inplace=True, drop=True)
    full_data = preprocessing_data(full_data)
    train_data, tmp_data = train_test_split(full_data, test_size=0.1, random_state=42)
    valid_data, test_data = train_test_split(tmp_data, test_size=0.8, random_state=42)
    train_data.to_csv('../dataset/full_train_data_summarization.csv', index=False)
    valid_data.to_csv('../dataset/full_validation_data_summarization.csv', index=False)
    test_data.to_csv('../dataset/full_test_data_summarization.csv', index=False)
    
    
if __name__ == '__main__':
    merge_and_preprocess_and_split_all_data()