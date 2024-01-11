import warnings
import time
from general_methods import find_element_by_css, find_element_by_xpath, find_elements_by_xpath, driver_wait_by_xpath, scroll_down, DATA_SUMMARIZATION

warnings.filterwarnings('ignore')


def get_news(driver):
    try:
        driver_wait_by_xpath(driver, xpath='//section[@class="section wrap-main-nav" and @id="wrap-main-nav"]//ul[@class="parent"]', seconds=60)
        list_menu_nav = find_elements_by_xpath(driver, xpath='//section[@class="section wrap-main-nav" and @id="wrap-main-nav"]//ul[@class="parent"]/li')
        del list_menu_nav[-1]
        del list_menu_nav[1]
        del list_menu_nav[0:5:2]
        del list_menu_nav[2]
        original_window = driver.current_window_handle
        for idx in range(len(list_menu_nav)):
            list_menu_nav = find_elements_by_xpath(driver, xpath='//section[@class="section wrap-main-nav" and @id="wrap-main-nav"]//ul[@class="parent"]/li')
            del list_menu_nav[-1]
            del list_menu_nav[1]
            del list_menu_nav[0:5:2]
            del list_menu_nav[2]
            link = find_element_by_css(list_menu_nav[idx], css_selector='a').get_attribute('href')
            driver.switch_to.new_window('tab')
            driver.get(link)
            click_news_from_topic(driver)
            time.sleep(5)
            driver.close()
            driver.switch_to.window(original_window)
            driver_wait_by_xpath(driver, xpath='//section[@class="section wrap-main-nav" and @id="wrap-main-nav"]//ul[@class="parent"]', seconds=60)
    except Exception as err:
        raise Exception("Have error in get_news function") from err


def click_news_from_topic(driver):
    for i in range(0, 20):
        driver_wait_by_xpath(driver, xpath='//div[@class="container flexbox"]//article[@class="item-news item-news-common thumb-left"]/p[@class="description"]', seconds=60)
        scroll_down(driver, 'dantri')
        list_news = find_elements_by_xpath(driver, xpath='//div[@class="container flexbox"]//article[@class="item-news item-news-common thumb-left"]/p[@class="description"]')
        original_window = driver.current_window_handle
        for item in list_news:
            link = find_element_by_css(item, css_selector='a').get_attribute('href')
            driver.switch_to.new_window('tab')
            driver.get(link)
            summarization = get_summarization(driver)
            content = get_content(driver)
            # DATA_SUMMARIZATION['content'].append(content)
            # DATA_SUMMARIZATION['summarization'].append(summarization)
            print(summarization)
            # print(content)
            driver.close()
            driver.switch_to.window(original_window)
        if i == 19:
            break
        else:
            btn_next = find_element_by_xpath(driver, xpath='//div[@class="width_common pagination flexbox"]//div[@class="button-page flexbox"]/a[@class="btn-page next-page "]')
            driver.execute_script("arguments[0].click();", btn_next)
    

def get_summarization(driver):
    driver_wait_by_xpath(driver, xpath='//div[@class="container"]', seconds=60)
    summarization = find_element_by_xpath(driver, xpath='//div[@class="container"]//p[@class="description"]').text.strip()
    return summarization


def get_content(driver):
    driver_wait_by_xpath(driver,
                         xpath='//div[@class="container"]//article[@class="fck_detail "]',
                         seconds=60)
    lst_content = [item.text.strip() for item in find_elements_by_xpath(driver,
                                                                        xpath='//div[@class="container"]//article[@class="fck_detail "]/p[@class="Normal"]')]
    content = '\n'.join(lst_content)
    return content
