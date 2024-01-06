import warnings
from general_methods import driver_wait_by_xpath, scroll_down
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

warnings.filterwarnings('ignore')


def click_topic(driver):
    # driver_wait_by_xpath(driver, xpath='//ul[@class="menu-nav"')
    scroll_down(driver)


def click_news_from_topic(driver):
    pass


def get_summarization(driver):
    pass


def get_content(driver):
    pass


def get_news(driver):
    pass
