import warnings
import time
from general_methods import find_element_by_css, find_element_by_xpath, find_elements_by_xpath, driver_wait_by_xpath, scroll_down, DATA_SUMMARIZATION

warnings.filterwarnings('ignore')


def get_news(driver):
    try:
        driver_wait_by_xpath(driver, xpath='//nav[@class="menu container bg-wrap"]/ol[@class="menu-wrap bg-wrap"]', seconds=120)
        list_menu_nav = find_elements_by_xpath(driver, xpath='//nav[@class="menu container bg-wrap"]/ol[@class="menu-wrap bg-wrap"]/li[@class="has-child"]')
        original_window = driver.current_window_handle
        for idx in range(len(list_menu_nav)):
            list_menu_nav = find_elements_by_xpath(driver, xpath='//nav[@class="menu container bg-wrap"]/ol[@class="menu-wrap bg-wrap"]/li[@class="has-child"]')
            link = find_element_by_css(list_menu_nav[idx], css_selector='a').get_attribute('href')
            driver.switch_to.new_window('tab')
            driver.get(link)
            click_news_from_topic(driver)
            time.sleep(5)
            driver.close()
            driver.switch_to.window(original_window)
            driver_wait_by_xpath(driver, xpath='//nav[@class="menu container bg-wrap"]/ol[@class="menu-wrap bg-wrap"]', seconds=120)
    except Exception as err:
        raise Exception("Have error in get_news function") from err


def click_news_from_topic(driver):
    for i in range(0, 30):
        driver_wait_by_xpath(driver, xpath='//div[@class="grid list" and @id="bai-viet"]//div[@class="article list"]/article[@class="article-item"]', seconds=120)
        scroll_down(driver, 'dantri')
        list_news = find_elements_by_xpath(driver, xpath='//div[@class="grid list" and @id="bai-viet"]//div[@class="article list"]/article[@class="article-item"]/div[@class="article-thumb"]')
        original_window = driver.current_window_handle
        for item in list_news:
            try:
                link = find_element_by_css(item, css_selector='a').get_attribute('href')
                driver.switch_to.new_window('tab')
                driver.get(link)
                try:
                    summarization = get_summarization(driver)
                    content = get_content(driver)
                    DATA_SUMMARIZATION['context'].append(content)
                    DATA_SUMMARIZATION['summarization'].append(summarization)
                except:
                    pass
                driver.close()
                driver.switch_to.window(original_window)
            except:
                pass
        if i == 29:
            break
        else:
            btn_next = find_element_by_xpath(driver, xpath='//div[@class="grid list" and @id="bai-viet"]//div[@class="pagination"]/a[@class="page-item next"]')
            driver.execute_script("arguments[0].click();", btn_next)
    

def get_summarization(driver):
    try:
        driver_wait_by_xpath(driver, xpath='//div[@class="singular-wrap"]/article[@class="singular-container"]', seconds=10)
        summarization = find_element_by_xpath(driver, xpath='//div[@class="singular-wrap"]/article[@class="singular-container"]/h2[@class="singular-sapo"]').text.strip().replace('(Dân trí) -', '').strip()
    except:
        try:
            driver_wait_by_xpath(driver, xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap photo-story"]', seconds=10)
            summarization = find_element_by_xpath(driver, xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap photo-story"]/h2[@class="e-magazine__sapo sapo-top"]').text.strip().replace('(Dân trí) -', '').strip()
        except:
            try:
                driver_wait_by_xpath(driver, xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap d-magazine"]', seconds=10)
                summarization = find_element_by_xpath(driver, xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap d-magazine"]/h2[@class="e-magazine__sapo"]').text.strip().replace('(Dân trí) -', '').strip()
            except:
                driver_wait_by_xpath(driver, xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap infographic"]', seconds=10)
                summarization = find_element_by_xpath(driver, xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap infographic"]/h2[@class="e-magazine__sapo"]').text.strip().replace('(Dân trí) -', '').strip()
    return summarization


def get_content(driver):
    try:
        driver_wait_by_xpath(driver,
                             xpath='//div[@class="singular-wrap"]/article[@class="singular-container"]',
                             seconds=10)
        lst_content = [item.text.strip() for item in find_elements_by_xpath(driver,
                                                                            xpath='//div[@class="singular-wrap"]/article[@class="singular-container"]/div[@class="singular-content"]/p')]
    except:
        try:
            driver_wait_by_xpath(driver,
                                 xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap photo-story"]',
                                 seconds=10)
            lst_content = [item.text.strip() for item in find_elements_by_xpath(driver,
                                                                                xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap photo-story"]/div[@class="e-magazine__body"]/p')]
        except:
            try:
                driver_wait_by_xpath(driver,
                                     xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap d-magazine"]',
                                     seconds=10)
                lst_content = [item.text.strip() for item in find_elements_by_xpath(driver,
                                                                                    xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap d-magazine"]/div[@class="e-magazine__body dnews__body"]/p')]
            except:
                driver_wait_by_xpath(driver,
                                     xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap infographic"]',
                                     seconds=10)
                lst_content = [item.text.strip() for item in find_elements_by_xpath(driver,
                                                                                    xpath='//main[@class="body container"]/article[@class="e-magazine bg-wrap infographic"]/div[@class="e-magazine__body"]/p')]
    content = '\n'.join(lst_content)
    return content
