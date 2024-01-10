import warnings
import hydra
import crawl_thanhnien_news
import crawl_tuoitre_news
import crawl_vnexpress_news
from general_methods import scroll_down
from selenium import webdriver

warnings.filterwarnings('ignore')
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('--disable-notifications')
DRIVER = webdriver.Chrome(options=OPTIONS)
ACTIONS_MAP = {
    'thanhnien': crawl_thanhnien_news.get_news,
    'tuoitre': crawl_tuoitre_news.get_news,
    'vnexpress': crawl_vnexpress_news.get_news
}


@hydra.main(config_path='../config', config_name='crawlparameters')
def click_websites_and_get_data(config):
    for key, item in config.news.items():
        DRIVER.get(config.news[key])
        DRIVER.maximize_window()
        ACTIONS_MAP[key](DRIVER)
        break
    # save_data(DATA_SUMMARIZATION, path_data='../data/')


if __name__ == '__main__':
    click_websites_and_get_data()
