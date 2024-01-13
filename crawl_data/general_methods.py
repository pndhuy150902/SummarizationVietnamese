import time
import re
import warnings
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

warnings.filterwarnings('ignore')
DATA_SUMMARIZATION = {
    'context': [],
    'summarization': []
}


def find_element_by_xpath(driver, xpath):
    return driver.find_element(by=By.XPATH, value=xpath)


def find_element_by_css(driver, css_selector):
    return driver.find_element(by=By.CSS_SELECTOR, value=css_selector)


def find_elements_by_xpath(driver, xpath):
    return driver.find_elements(by=By.XPATH, value=xpath)


def find_elements_by_css(driver, css_selector):
    return driver.find_elements(by=By.CSS_SELECTOR, value=css_selector)


def scroll_down(driver, key):
    reached_page_end = False
    last_height = driver.execute_script("return document.body.scrollHeight;")
    count_btn_tuoitre = 0
    count_btn_thanhnien = 0
    while not reached_page_end:
        try:
            driver.execute_script(f"window.scrollTo(0, {last_height});")
            if key == 'thanhnien':
                btn_more = find_element_by_xpath(driver, xpath='//div[@class="container"]/div[@class="list__stream-flex"]//a[@class="list__center view-more list__viewmore"]')
                count_btn_thanhnien += 1
                if btn_more.value_of_css_property('display') == 'block':
                    driver.execute_script("arguments[0].click();", btn_more)
            elif key == 'tuoitre':
                count_btn_tuoitre += 1
                btn_more = find_element_by_xpath(driver, xpath='//div[@class="container"]/div[@class="list__listing-flex"]//div[@class="box-viewmore"]//a[@class="view-more"]')
                driver.execute_script("arguments[0].click();", btn_more)
            elif key == 'dantri':
                pass
            time.sleep(10)
            scroll_height = driver.execute_script("return document.body.scrollHeight;")
            if (last_height == scroll_height) or (count_btn_tuoitre == 40) or (count_btn_thanhnien == 40):
                reached_page_end = True
            else:
                last_height = scroll_height
            driver.implicitly_wait(10)
        except:
            break


def driver_wait_by_xpath(driver, xpath, seconds):
    WebDriverWait(driver, seconds).until(
        ec.presence_of_element_located((By.XPATH, xpath))
    )
    
    
def preprocessing_data(df):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[~((df['context'].str.len() < 512) | (df['context'].str == '') | (df['summarization'].str.len() < 128) | (df['summarization'].str.contains('kết quả xổ số')))]
    df['summarization'] = df['summarization'].apply(lambda x: re.sub(r'(\d+)\.(\d+)(?![\d%])', r'\1\2', x))
    df['context'] = df['context'].apply(lambda x: re.sub(r'(\d+)\.(\d+)(?![\d%])', r'\1\2', x))
    return df
    
    
def save_data(data, path_data):
    df = pd.DataFrame(data)
    df = preprocessing_data(df)
    df.to_csv(path_data, index=False)
