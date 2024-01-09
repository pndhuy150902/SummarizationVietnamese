import warnings
from general_methods import find_element_by_css, find_elements_by_xpath, driver_wait_by_xpath, scroll_down
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

warnings.filterwarnings('ignore')


def click_topic(driver):
    try:
        driver_wait_by_xpath(driver, xpath='//ul[@class="menu-nav"]', seconds=300)
        list_menu_nav = find_elements_by_xpath(driver, xpath='//ul[@class="menu-nav"]/li')
        del list_menu_nav[0]
        for idx in range(len(list_menu_nav)):
            list_menu_nav = find_elements_by_xpath(driver, xpath='//ul[@class="menu-nav"]/li')
            del list_menu_nav[0]
            list_menu_nav[idx].click()
            click_news_from_topic(driver)
            driver.back()
            driver_wait_by_xpath(driver, xpath='//ul[@class="menu-nav"]', seconds=300)
    except Exception as err:
        raise Exception("Have error in click_topic function") from err


def click_news_from_topic(driver):
    driver_wait_by_xpath(driver, xpath='//div[@class="container"]//div[@class="box-category-middle list__main_check"]/div[@class="box-category-item"]', seconds=300)
    # scroll_down(driver)
    list_news = find_elements_by_xpath(driver, xpath='//div[@class="container"]//div[@class="box-category-middle list__main_check"]/div[@class="box-category-item"]')
    original_window = driver.current_window_handle
    for item in list_news:
        link = find_element_by_css(item, css_selector='a').get_attribute('href')
        driver.switch_to.new_window('tab')
        driver.get(link)
        driver.close()
        driver.switch_to.window(original_window)


def get_summarization(driver):
    pass


def get_content(driver):
    pass


def get_news(driver):
    pass
