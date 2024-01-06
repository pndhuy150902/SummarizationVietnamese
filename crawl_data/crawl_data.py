import warnings
import pickle
import hydra
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
warnings.filterwarnings('ignore')
OPTIONS = webdriver.ChromeOptions()
OPTIONS.add_argument('--disable-notifications')
# DRIVER = webdriver.Chrome(options=OPTIONS)


@hydra.main(config_path='../config', config_name='crawlparameters')
def get_website(config):
  for key, item in config.news.items():
    print(key, item)
  # driver.get(config.news.thanhnien)
  # driver.maximize_window()

if __name__ == '__main__':  
  get_website()