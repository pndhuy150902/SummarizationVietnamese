import warnings
import os
import re
import pickle
import pandas as pd
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def remove_longer_text(df):
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    prefix = '<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm lược ngắn gọn nội dung sau bằng tiếng Việt: '
    infix = ' [/INST]'
    suffix = '</s>'
    df['length_prompt'] = (prefix + df['context'] + infix + df['summarization'] + suffix).apply(lambda x: len(tokenizer.tokenize(str(x))))
    df = df[~((df['length_prompt'] > 4096) | (df['length_prompt'] < 768))]
    df.sort_values(by=['length_prompt'], ascending=False, inplace=True)
    df.drop(columns=['length_prompt'], axis=1, inplace=True)
    return df


def remove_longer_text_with_title(df):
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    prefix = '<s>[INST] Bạn là một trợ lý AI. Bạn sẽ được giao một nhiệm vụ. Hãy tóm lược ngắn gọn nội dung sau bằng tiếng Việt biết rằng tiêu đề của nội dung là "'
    infix = ' [/INST]'
    suffix = '</s>'
    df['length_prompt'] = (prefix + df['title'] + '": ' + df['context'] + infix + df['summarization'] + suffix).apply(lambda x: len(tokenizer.tokenize(str(x))))
    df = df[~((df['length_prompt'] > 4096) | (df['length_prompt'] < 768))]
    df.sort_values(by=['length_prompt'], ascending=False, inplace=True)
    df.drop(columns=['length_prompt'], axis=1, inplace=True)
    return df


def read_and_merge_data_crawled():
    news_thanhnien = pd.read_csv('../dataset/news_crawled_data/crawled_data_thanhnien.csv')
    news_thanhnien_1 = pd.read_csv('../dataset/news_crawled_data/crawled_data_thanhnien_1.csv')
    news_tuoitre = pd.read_csv('../dataset/news_crawled_data/crawled_data_tuoitre.csv')
    news_tuoitre_1 = pd.read_csv('../dataset/news_crawled_data/crawled_data_tuoitre_1.csv')
    news_dantri = pd.read_csv('../dataset/news_crawled_data/crawled_data_dantri.csv')
    news_dantri_1 = pd.read_csv('../dataset/news_crawled_data/crawled_data_dantri_1.csv')
    full_news_crawled = pd.concat([news_thanhnien_1, news_tuoitre_1, news_dantri_1, news_thanhnien, news_tuoitre, news_dantri], axis=0)
    full_news_crawled = full_news_crawled[~(full_news_crawled['context'] == '')]
    full_news_crawled = full_news_crawled[~(full_news_crawled['summarization'] == '')]
    full_news_crawled = remove_longer_text(full_news_crawled)
    full_news_crawled.drop_duplicates(inplace=True)
    full_news_crawled.dropna(inplace=True)
    full_news_crawled.reset_index(inplace=True, drop=True)
    return full_news_crawled


def read_data_vslp():
    structure_data = {
        'title': [],
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
                structure_data['title'].append(structure_doc["title"].strip())
                structure_data['context'].append(structure_doc["raw_text"].strip())
                structure_data['summarization'].append(structure_doc["anchor_text"].strip())
    vlsp_data = pd.DataFrame(structure_data)
    vlsp_data = vlsp_data[~(vlsp_data['title'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['context'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['summarization'] == '')]
    vlsp_data.drop_duplicates(inplace=True)
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
    vietgpt_data.drop_duplicates(inplace=True)
    vietgpt_data.dropna(inplace=True)
    vietgpt_data.reset_index(inplace=True, drop=True)
    return vietgpt_data


def read_data_wikilingual():
    structure_data = {
        'title': [],
        'context': [],
        'summarization': []
    }
    with open('../dataset/wikilingual_vietnamese_data/vietnamese.pkl', mode='rb') as file:
        obj = pickle.load(file)
    for subject in obj.items():
        for news in subject[1].items():
            structure_data['title'].append(news[0].strip())
            structure_data['context'].append(news[1]['document'].strip())
            structure_data['summarization'].append(news[1]['summary'].strip())
    wikilingual_data = pd.DataFrame(structure_data)
    wikilingual_data = wikilingual_data[wikilingual_data['context'].str.len() > 100]
    wikilingual_data.drop_duplicates(inplace=True)
    wikilingual_data.dropna(inplace=True)
    wikilingual_data.reset_index(inplace=True, drop=True)
    return wikilingual_data


def read_data_vims():
    vims_structure = {
        'title': [],
        'context': [],
        'summarization': []
    }
    lst_folder_cluster = os.listdir('../dataset/ViMs/original/')
    for i, folder in enumerate(lst_folder_cluster):
        if i == 0:
            continue
        lst_file = os.listdir('../dataset/ViMs/original/' + folder + '/original/')
        for file in lst_file:
            with open('../dataset/ViMs/original/' + folder + '/original/' + file, 'r', encoding='utf-8') as f:
                content = f.readlines()
                try:
                    title = content[0].strip('\n').split('Title: ')[1].strip().strip('.').strip('/')
                    summarization = content[6].strip('\n').split('Summary: ')[1].strip().strip('.').strip('/')
                    context = ''.join(content[8:]).strip().strip('.').strip('/')
                    vims_structure['title'].append(title)
                    vims_structure['summarization'].append(summarization)
                    vims_structure['context'].append(context)
                except:
                    continue
    vims_data = pd.DataFrame(vims_structure)
    vims_data.drop_duplicates(inplace=True)
    vims_data.dropna(inplace=True)
    vims_data.reset_index(inplace=True, drop=True)
    return vims_data


def read_data_vietnews():
    vietnews_structure = {
        'title': [],
        'context': [],
        'summarization': []
    }
    train_vietnews_data = pd.read_parquet('../dataset/vietnews/train-00000-of-00001-2a54892d3f45697b.parquet')
    validation_vietnews_data = pd.read_parquet('../dataset/vietnews/validation-00000-of-00001-2a6248b18b5da97d.parquet')
    test_vietnews_data = pd.read_parquet('../dataset/vietnews/test-00000-of-00001-f435108aee3e1334.parquet')
    vietnews_structure['title'].extend(train_vietnews_data['title'].tolist())
    vietnews_structure['title'].extend(validation_vietnews_data['title'].tolist())
    vietnews_structure['title'].extend(test_vietnews_data['title'].tolist())
    vietnews_structure['context'].extend(train_vietnews_data['article'].tolist())
    vietnews_structure['context'].extend(validation_vietnews_data['article'].tolist())
    vietnews_structure['context'].extend(test_vietnews_data['article'].tolist())
    vietnews_structure['summarization'].extend(train_vietnews_data['abstract'].tolist())
    vietnews_structure['summarization'].extend(validation_vietnews_data['abstract'].tolist())
    vietnews_structure['summarization'].extend(test_vietnews_data['abstract'].tolist())
    vietnews_data = pd.DataFrame(vietnews_structure)
    vietnews_data = vietnews_data.sample(frac=1, random_state=42)
    vietnews_data.drop_duplicates(inplace=True)
    vietnews_data.dropna(inplace=True)
    vietnews_data.reset_index(inplace=True, drop=True)
    return vietnews_data


def preprocessing_data(df):
    df['context'] = df['context'].apply(lambda x: re.sub(r'\n+\s+', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\n+\.+', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\n+\-+', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\n+\,+', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s*'\s*(.*?)\s*'\s*", r" '\1' ", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'(\d+)\s+([A-Za-z])\b', r'\1\2', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s*-\s*|\s*\.\s*', lambda y: '-' if '-' in y.group() else '.', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\(\s+", " (", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\)\s+", ") ", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\[\s+", " [", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\]\s+", "] ", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\{\s+", " {", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\}\s+", "} ", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\…', '...', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\⋯', 'dấu ba chấm', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'{.*}', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\.\.\. \.\.\.', ', ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\>> ', '', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\Ảnh: [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(Nguồn: [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh: [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh: [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(\ảnh: [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn: [\w+\s+\/+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn: [\w+\.]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(nguồn: [\w+\s+\-\w+]+\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\(ảnh minh họa\)', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'Ảnh minh họa.', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\.+\n\.+', ' ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\,\s+', ', ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\;\s+', '; ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\:\s+', ': ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\?\s+', '? ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\!\s+', '! ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\đ\s+', 'đ ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\$\s+', '$ ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\/\s+', '/', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\.\.\.\s+', '... ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\.\s+', '. ', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r"\s+\'\s+", "'", x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'\s+\.', '.', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r' +', ' ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s*'\s*(.*?)\s*'\s*", r" '\1' ", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'(\d+)\s+([A-Za-z])\b', r'\1\2', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s*-\s*|\s*\.\s*', lambda y: '-' if '-' in y.group() else '.', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\(\s+", " (", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\)\s+", ") ", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\[\s+", " [", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\]\s+", "] ", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\{\s+", " {", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\}\s+", "} ", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\…', '...', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\⋯', 'dấu ba chấm', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'{.*}', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\.\.\. \.\.\.', ', ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\>> ', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\(ảnh minh họa\)', ' ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'Ảnh minh họa.', '', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\,\s+', ', ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\;\s+', '; ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\:\s+', ': ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\?\s+', '? ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\!\s+', '! ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\đ\s+', 'đ ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\$\s+', '$ ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\/\s+', '/', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\.\.\.\s+', '... ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\.\s+', '. ', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r"\s+\'\s+", "'", x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'\s+\.', '.', x))
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r' +', ' ', x))
    df['context'] = df['context'].apply(lambda x: x.strip().strip('\n'))
    df['summarization'] = df['summarization'].apply(lambda x: x.strip().strip('\n'))
    if 'title' in df.columns:
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s*'\s*(.*?)\s*'\s*", r" '\1' ", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'(\d+)\s+([A-Za-z])\b', r'\1\2', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s*-\s*|\s*\.\s*', lambda y: '-' if '-' in y.group() else '.', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\(\s+", " (", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\)\s+", ") ", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\[\s+", " [", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\]\s+", "] ", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\{\s+", " {", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\}\s+", "} ", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\…', '...', x)) 
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\,\s+', ', ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\;\s+', '; ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\:\s+', ': ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\?\s+', '? ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\!\s+', '! ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\đ\s+', 'đ ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\$\s+', '$ ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\/\s+', '/', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\.\.\.\s+', '... ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\.\s+', '. ', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r"\s+\'\s+", "'", x))
        df['title'] = df['title'].apply(lambda x: re.sub(r'\s+\.', '.', x))
        df['title'] = df['title'].apply(lambda x: re.sub(r' +', ' ', x))
        df['title'] = df['title'].apply(lambda x: x.strip().strip('\n'))
    return df


def merge_and_preprocess_and_split_all_data():
    crawled_data = read_and_merge_data_crawled()
    vlsp_data = read_data_vslp()
    vietgpt_data = read_data_vietgpt()
    wikilingual_data = read_data_wikilingual()
    vims_data = read_data_vims()
    vietnews_data = read_data_vietnews()
    crawled_data = preprocessing_data(crawled_data)
    vlsp_data = preprocessing_data(vlsp_data)
    vietgpt_data = preprocessing_data(vietgpt_data)
    wikilingual_data = preprocessing_data(wikilingual_data)
    vims_data = preprocessing_data(vims_data)
    vietnews_data = preprocessing_data(vietnews_data)
    crawled_data = crawled_data.loc[::-1].reset_index(drop=True)
    crawled_data = crawled_data[~(crawled_data['context'] == '')]
    crawled_data = crawled_data[~(crawled_data['summarization'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['title'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['context'] == '')]
    vlsp_data = vlsp_data[~(vlsp_data['summarization'] == '')]
    wikilingual_data = wikilingual_data[~(wikilingual_data['title'] == '')]
    wikilingual_data = wikilingual_data[~(wikilingual_data['context'] == '')]
    wikilingual_data = wikilingual_data[~(wikilingual_data['summarization'] == '')]
    vietgpt_data = vietgpt_data[~(vietgpt_data['context'] == '')]
    vietgpt_data = vietgpt_data[~(vietgpt_data['summarization'] == '')]
    vims_data = vims_data[~(vims_data['title'] == '')]
    vims_data = vims_data[~(vims_data['context'] == '')]
    vims_data = vims_data[~(vims_data['summarization'] == '')]
    vietnews_data = vietnews_data[~(vietnews_data['title'] == '')]
    vietnews_data = vietnews_data[~(vietnews_data['context'] == '')]
    vietnews_data = vietnews_data[~(vietnews_data['summarization'] == '')]
    crawled_data.drop_duplicates(inplace=True)
    vlsp_data.drop_duplicates(inplace=True)
    wikilingual_data.drop_duplicates(inplace=True)
    vietgpt_data.drop_duplicates(inplace=True)
    vims_data.drop_duplicates(inplace=True)
    vietnews_data.drop_duplicates(inplace=True)
    wikilingual_data_with_title = wikilingual_data[:5000]
    wikilingual_data_no_title = wikilingual_data[5000:][['context', 'summarization']]
    wikilingual_data_with_title = remove_longer_text_with_title(wikilingual_data_with_title)
    wikilingual_data_no_title = remove_longer_text(wikilingual_data_no_title)
    vlsp_data_with_title = vlsp_data[:625]
    vlsp_data_no_title = vlsp_data[625:][['context', 'summarization']]
    vlsp_data_with_title = remove_longer_text_with_title(vlsp_data_with_title)
    vlsp_data_no_title = remove_longer_text(vlsp_data_no_title)
    vietnews_data_with_title = vietnews_data[:21000]
    vietnews_data_no_title = vietnews_data[21000:44000][['context', 'summarization']]
    vietnews_data_with_title = remove_longer_text_with_title(vietnews_data_with_title)
    vietnews_data_no_title = remove_longer_text(vietnews_data_no_title)
    train_data = pd.concat([
        vietgpt_data[:int(0.53 * len(vietgpt_data))],
        crawled_data[:int(0.9 * len(crawled_data))],
        vietnews_data_no_title[:int(0.8 * len(vietnews_data_no_title))],
        vlsp_data_no_title[:int(0.8 * len(vlsp_data_no_title))],
    ], axis=0)
    train_data_title = pd.concat([
        remove_longer_text_with_title(vims_data[:1600]),
        vietnews_data_with_title,
        vlsp_data_with_title,
    ])
    test_data = pd.concat([
        vietgpt_data[int(0.53*len(vietgpt_data)):int(0.58*len(vietgpt_data))],
        crawled_data[int(0.9 * len(crawled_data)):],
        vims_data[1600:][['context', 'summarization']],
        vietnews_data_no_title[int(0.8 * len(vietnews_data_no_title)):],
        vlsp_data_no_title[int(0.8 * len(vlsp_data_no_title)):],
    ], axis=0)
    train_data.drop_duplicates(inplace=True)
    test_data.drop_duplicates(inplace=True)
    train_data_title.drop_duplicates(inplace=True)
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    train_data_title.reset_index(inplace=True, drop=True)
    print(vietgpt_data[:int(0.50 * len(vietgpt_data))].info())
    print(crawled_data[:int(0.9 * len(crawled_data))].info())
    print(vietnews_data_no_title[:int(0.8 * len(vietnews_data_no_title))].info())
    print(vlsp_data_no_title[:int(0.8 * len(vlsp_data_no_title))].info())
    print("------------------------------------------------------------------------------------")
    print(remove_longer_text_with_title(vims_data[:1600]).info())
    print(vietnews_data_with_title.info())
    print(vlsp_data_with_title.info())
    print("------------------------------------------------------------------------------------")
    print(vietgpt_data[int(0.50*len(vietgpt_data)):int(0.55*len(vietgpt_data))].info())
    print(crawled_data[int(0.9 * len(crawled_data)):].info())
    print(vims_data[1600:][['context', 'summarization']].info())
    print(vietnews_data_no_title[int(0.8 * len(vietnews_data_no_title)):].info())
    print(vlsp_data_no_title[int(0.8 * len(vlsp_data_no_title)):].info())
    print("------------------------------------------------------------------------------------")
    print(train_data.info())
    print(train_data_title.info())
    print(test_data.info())
    # train_data.to_csv('../dataset/full_train_data_summarization.csv', index=False)
    # test_data.to_csv('../dataset/full_test_data_summarization.csv', index=False)
    # train_data_title.to_csv('../dataset/full_train_data_title_summarization.csv', index=False)

    
if __name__ == '__main__':
    merge_and_preprocess_and_split_all_data()
