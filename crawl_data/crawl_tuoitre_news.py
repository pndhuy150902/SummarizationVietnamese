import warnings
import time
from general_methods import find_element_by_css, find_element_by_xpath, find_elements_by_xpath, driver_wait_by_xpath, scroll_down, DATA_SUMMARIZATION

warnings.filterwarnings('ignore')


def get_news(driver):
    try:
        driver_wait_by_xpath(driver, xpath='//div[@class="header"]//div[@class="container"]//ul[@class="menu-nav"]', seconds=300)
        list_menu_nav = find_elements_by_xpath(driver, xpath='//div[@class="header"]//div[@class="container"]//ul[@class="menu-nav"]/li')
        del list_menu_nav[0:2]
        for idx in range(len(list_menu_nav)):
            list_menu_nav = find_elements_by_xpath(driver, xpath='//div[@class="header"]//div[@class="container"]//ul[@class="menu-nav"]/li')
            del list_menu_nav[0:2]
            list_menu_nav[idx].click()
            click_news_from_topic(driver)
            time.sleep(5)
            driver.back()
            driver_wait_by_xpath(driver, xpath='//div[@class="header"]//div[@class="container"]//ul[@class="menu-nav"]', seconds=300)
    except Exception as err:
        raise Exception("Have error in get_news function") from err


def click_news_from_topic(driver):
    driver_wait_by_xpath(driver, xpath='//div[@class="list__listing"]/div[@class="container"]//div[@class="box-category-middle" and @id="load-list-news"]/div[@class="box-category-item"]', seconds=300)
    scroll_down(driver, 'tuoitre')
    list_news = find_elements_by_xpath(driver, xpath='//div[@class="list__listing"]/div[@class="container"]//div[@class="box-category-middle" and @id="load-list-news"]/div[@class="box-category-item"]')
    original_window = driver.current_window_handle
    for item in list_news:
        link = find_element_by_css(item, css_selector='a').get_attribute('href')
        driver.switch_to.new_window('tab')
        driver.get(link)
        summarization = get_summarization(driver)
        content = get_content(driver)
        DATA_SUMMARIZATION['content'].append(content)
        DATA_SUMMARIZATION['summarization'].append(summarization)
        driver.close()
        driver.switch_to.window(original_window)


def get_summarization(driver):
    driver_wait_by_xpath(driver, xpath='//div[@class="main" and @id="content"]//div[@class="container"]', seconds=300)
    summarization = find_element_by_xpath(driver, xpath='//div[@class="main" and @id="content"]//div[@class="container"]//h2[@class="detail-sapo"]').text.strip()
    return summarization


def get_content(driver):
    driver_wait_by_xpath(driver,
                         xpath='//div[@class="main" and @id="content"]//div[@class="container"]//div[@class="detail-cmain"]/div[@class="detail-content afcbc-body"]',
                         seconds=300)
    lst_content = [item.text.strip() for item in find_elements_by_xpath(driver,
                                                                        xpath='//div[@class="main" and @id="content"]//div[@class="container"]//div[@class="detail-cmain"]/div[@class="detail-content afcbc-body"]/p')]
    content = '\n'.join(lst_content)
    return content
