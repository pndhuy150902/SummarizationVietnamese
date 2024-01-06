import warnings
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

warnings.filterwarnings('ignore')


def find_element_by_xpath(driver, xpath):
    return driver.find_element(by=By.XPATH, value=xpath)


def find_element_by_css(driver, css_selector):
    return driver.find_element(by=By.CSS_SELECTOR, value=css_selector)


def find_elements_by_xpath(driver, xpath):
    return driver.find_elements(by=By.XPATH, value=xpath)


def find_elements_by_css(driver, css_selector):
    return driver.find_elements(by=By.CSS_SELECTOR, value=css_selector)


def scroll_down(driver, scroll_value):
    scroll_height = driver.execute_script("return document.body.scrollHeight;")
    scroll_value += 500
    if scroll_value > scroll_height:
        scroll_value = scroll_height
    driver.execute_script(f"window.scrollTo(0, {scroll_value});")
    time.sleep(3)
    return scroll_value


def driver_wait_by_xpath(driver, xpath, seconds):
    return WebDriverWait(driver, seconds).until(
        ec.presence_of_element_located((By.XPATH, xpath))
    )
