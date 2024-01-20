import warnings
import os
import hydra
import crawl_thanhnien_news
import crawl_tuoitre_news
import crawl_dantri_news
from general_methods import save_data, DATA_SUMMARIZATION
from selenium import webdriver
from tqdm import tqdm

warnings.filterwarnings('ignore')
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('--disable-notifications')
DRIVER = webdriver.Chrome(options=OPTIONS)
ACTIONS_MAP = {
    'thanhnien': crawl_thanhnien_news.get_news,
    'tuoitre': crawl_tuoitre_news.get_news,
    'dantri': crawl_dantri_news.get_news,
}


@hydra.main(config_path='../config', config_name='crawlparameters', version_base=None)
def click_websites_and_get_data(config):
    for key, item in tqdm(config.news.items()):
        DRIVER.get(config.news[key])
        DRIVER.maximize_window()
        ACTIONS_MAP[key](DRIVER)
        path_file = os.path.join('../dataset/', f'crawled_data_{key}.csv')
        save_data(DATA_SUMMARIZATION, path_data=path_file)
        DATA_SUMMARIZATION['context'] = []
        DATA_SUMMARIZATION['summarization'] = []


if __name__ == '__main__':
    click_websites_and_get_data()
    