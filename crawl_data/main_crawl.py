import warnings
import hydra
import crawl_thanhnien_news
import crawl_tuoitre_news
import crawl_vnexpress_news
from selenium import webdriver

warnings.filterwarnings('ignore')
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('--disable-notifications')
DRIVER = webdriver.Chrome(options=OPTIONS)
ACTIONS_MAP = {
    'thanhnien': crawl_thanhnien_news.get_news(DRIVER),
    'tuoitre': crawl_tuoitre_news.get_news(DRIVER),
    'vnexpress': crawl_vnexpress_news.get_news(DRIVER)
}


def save_data(data):
    pass


@hydra.main(config_path='../config', config_name='crawlparameters')
def click_website_and_get_data(config):
    for key, item in config.news.items():
        # print(key, item)
        DRIVER.get(config.news[key])
        DRIVER.maximize_window()
        # ACTIONS_MAP[key]


if __name__ == '__main__':
    click_website_and_get_data()
