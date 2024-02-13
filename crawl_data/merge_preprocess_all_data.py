import warnings
import re
import pickle
import pandas as pd
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def remove_longer_text(df):
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    prefix = '<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm lược ngắn gọn nội dung sau bằng tiếng Việt: '
    infix = ' [/INST] '
    suffix = '</s>'
    df['length_prompt'] = (prefix + df['context'] + infix + df['summarization'] + suffix).apply(lambda x: len(tokenizer.tokenize(str(x))))
    df = df[~(df['length_prompt'] > 4096)]
    df.drop(columns=['length_prompt'], axis=1, inplace=True)
    return df


def read_and_merge_data_crawled():
    news_thanhnien = pd.read_csv('../dataset/news_crawled_data/crawled_data_thanhnien.csv')
    news_tuoitre = pd.read_csv('../dataset/news_crawled_data/crawled_data_tuoitre.csv')
    news_dantri = pd.read_csv('../dataset/news_crawled_data/crawled_data_dantri.csv')
    full_news_crawled = pd.concat([news_thanhnien, news_tuoitre, news_dantri], axis=0)
    full_news_crawled = full_news_crawled[~(full_news_crawled['context'] == '')]
    full_news_crawled = full_news_crawled[~(full_news_crawled['summarization'] == '')]
    full_news_crawled = remove_longer_text(full_news_crawled)
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
    vlsp_data = remove_longer_text(vlsp_data)
    vlsp_data.dropna(inplace=True)
    vlsp_data.reset_index(inplace=True, drop=True)
    return vlsp_data


def read_data_vietgpt():
    train_vietgpt_data = pd.read_parquet('../dataset/news_vietgpt_data/train_news_summarization_vi_old_vietgpt.parquet')
    test_vietgpt_data = pd.read_parquet('../dataset/news_vietgpt_data/test_news_summarization_vi_old_vietgpt.parquet')
    vietgpt_data = pd.concat([train_vietgpt_data, test_vietgpt_data], axis=0)
    vietgpt_data = vietgpt_data[~(vietgpt_data['content'] == '')]
    vietgpt_data = vietgpt_data[~(vietgpt_data['summary'] == '')]
    vietgpt_data.rename(columns={'content': 'context', 'summary': 'summarization'}, inplace=True)
    vietgpt_data = remove_longer_text(vietgpt_data)
    vietgpt_data.dropna(inplace=True)
    vietgpt_data.reset_index(inplace=True, drop=True)
    return vietgpt_data


def read_data_wikilingual():
    structure_data = {
        'context': [],
        'summarization': []
    }
    with open('../dataset/wikilingual_vietnamese_data/vietnamese.pkl', mode='rb') as file:
        obj = pickle.load(file)
    for subject in obj.items():
        for news in subject[1].items():
            structure_data['context'].append(news[1]['document'])
            structure_data['summarization'].append(news[1]['summary'])
    wikilingual_data = pd.DataFrame(structure_data)
    wikilingual_data = remove_longer_text(wikilingual_data)
    wikilingual_data.dropna(inplace=True)
    wikilingual_data.reset_index(inplace=True, drop=True)
    return wikilingual_data


def preprocessing_data(df):
    df['context'] = df['context'].apply(lambda x: re.sub(r'\⋯', 'dấu ba chấm', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'{.*}', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\.\.\. \.\.\.', ', ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\>> ', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh: [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh: [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh: [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn: [\w+\s+\/+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn: [\w+\.]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn: [\w+\s+\-\w+]+\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(ảnh minh họa\)', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'Ảnh minh họa.', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\.+\n\.+', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+,\s+', ', ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\.\s+', '. ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r' +', ' ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\⋯', 'dấu ba chấm', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'{.*}', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\.\.\. \.\.\.', ', ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\>> ', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\(ảnh minh họa\)', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'Ảnh minh họa.', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+,\s+', ', ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\.\s+', '. ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r' +', ' ', x))
    df['context'] = df['context'].apply(lambda x: x.strip().strip('\n'))
    df['summarization'] = df['summarization'].apply(lambda x: x.strip().strip('\n'))
    return df


def merge_and_preprocess_and_split_all_data():
    crawled_data = read_and_merge_data_crawled()
    vlsp_data = read_data_vslp()
    vietgpt_data = read_data_vietgpt()
    wikilingual_data = read_data_wikilingual()
    crawled_data = preprocessing_data(crawled_data)
    vlsp_data = preprocessing_data(vlsp_data)
    vietgpt_data = preprocessing_data(vietgpt_data)
    wikilingual_data = preprocessing_data(wikilingual_data)
    crawled_data = crawled_data.loc[::-1].reset_index(drop=True)
    crawled_data = crawled_data[~(crawled_data['context'] == '')]
    crawled_data = crawled_data[~(crawled_data['summarization'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['context'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['summarization'] == '')]
    wikilingual_data = wikilingual_data[~(wikilingual_data['context'] == '')]
    wikilingual_data = wikilingual_data[~(wikilingual_data['summarization'] == '')]
    vietgpt_data = vietgpt_data[~(vietgpt_data['context'] == '')]
    vietgpt_data = vietgpt_data[~(vietgpt_data['summarization'] == '')]
    crawled_data.drop_duplicates(inplace=True)
    vlsp_data.drop_duplicates(inplace=True)
    wikilingual_data.drop_duplicates(inplace=True)
    vietgpt_data.drop_duplicates(inplace=True)
    train_data = pd.concat([
        crawled_data[:int(0.85 * len(crawled_data))],
        vlsp_data[:int(0.8 * len(vlsp_data))],
        wikilingual_data[:int(0.8 * len(wikilingual_data))],
        vietgpt_data[:int(0.24 * len(vietgpt_data))]
    ], axis=0)
    valid_data = pd.concat([
        crawled_data[int(0.85 * len(crawled_data)):int(0.9 * len(crawled_data))],
        vlsp_data[int(0.8 * len(vlsp_data)):int(0.85 * len(vlsp_data))],
        wikilingual_data[int(0.8 * len(wikilingual_data)):int(0.85 * len(wikilingual_data))],
        vietgpt_data[int(0.24*len(vietgpt_data)):int(0.25*len(vietgpt_data))]
    ], axis=0)
    test_data = pd.concat([
        crawled_data[int(0.9 * len(crawled_data)):],
        vlsp_data[int(0.85 * len(vlsp_data)):],
        wikilingual_data[int(0.85 * len(wikilingual_data)):],
        vietgpt_data[int(0.25*len(vietgpt_data)):int(0.28*len(vietgpt_data))]
    ], axis=0)
    train_data.reset_index(inplace=True, drop=True)
    valid_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    train_data.to_csv('../dataset/full_train_data_summarization.csv', index=False)
    valid_data.to_csv('../dataset/full_validation_data_summarization.csv', index=False)
    test_data.to_csv('../dataset/full_test_data_summarization.csv', index=False)

    
if __name__ == '__main__':
    merge_and_preprocess_and_split_all_data()
