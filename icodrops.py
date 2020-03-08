import logging
import multiprocessing
import re
import time
from multiprocessing import Pool

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


class IcodropsEnded:
    icodrops_ended_url = 'https://icodrops.com/category/ended-ico/'
    not_available = 'not_available'  # used as standard label for not available information

    def __init__(self, init_icodrops_ended_list_path, init_chromedriver_path, debug=False, ico_list=None):
        """
        Download the full list of ended icos and remove the one already downloaded if passed in the icolist parameter.
        :param debug:       the debug variable limit the retrieved values to facilitate the debug
        :param ico_list:    a list of dictionaries or a pandas dataframe containing
                            the generic information from the ico ended page
        """

        self.icodrops_ended_list_path = init_icodrops_ended_list_path
        self.chromedriver_path = init_chromedriver_path
        self.logger = self.set_logging()
        if debug:
            self.icodrops_ended_last_ico = 'Wirex'
            self.icodrops_exec_mode = 'ICOs\' subset'
        else:
            self.icodrops_ended_last_ico = 'Infinito'
            self.icodrops_exec_mode = 'all ICOs'
        if ico_list is None:
            self.ico_list = self.load_icos()
            self.logger.debug('Downloaded %d ICOs from %s' % (len(self.ico_list), self.icodrops_ended_url))
        else:
            if isinstance(ico_list, pd.DataFrame):
                init_df = pd.read_csv(self.icodrops_ended_list_path, header=0)
                init_df = init_df[['ticker', 'name', 'web_site', 'category_name', 'goal', 'goal_received',
                                   'on_exchanges']]
                init_list_1 = init_df.to_dict('records')
                init_list_2 = load_icos()
                self.ico_list = set(init_list_2) - set(init_list_1)
                self.logger.debug('Loaded %d ICOs from dataframe' % (len(self.ico_list)))
        self.logger.debug('INIT completed')

    def load_icos(self):
        """
        Return a list of dictionaries by retrieving from the icodrops main page and
        the generic information for a given ico.
        :return:                            list of dictionaries with general ico information
        """
        li_start_time = time.time()
        logger = self.logger
        logger.info('Initialized general inofrmation collection for %s' % self.icodrops_exec_mode)
        sleep_load_ico = 2  # used to pause the process and allow the browser to load the page
        i_load_ico = 1  # count the number of refresh(es)
        driver = webdriver.Chrome(self.chromedriver_path)
        driver.get(self.icodrops_ended_url)
        actions = ActionChains(driver)
        elem = driver.find_element_by_class_name('col-md-12')
        while self.icodrops_ended_last_ico not in elem.text:
            elem = driver.find_element_by_class_name('col-md-12')
            logger.debug('Executed {} refresh(es)'.format(i_load_ico))
            actions.send_keys(Keys.SPACE).perform()
            actions.send_keys(Keys.SPACE).perform()
            time.sleep(sleep_load_ico)
            actions.send_keys(Keys.ARROW_UP).perform()
            i_load_ico += 1
        r = driver.page_source
        logger.debug('Extracted page source code')
        soup = BeautifulSoup(r, 'html.parser')
        icos = soup.find_all('div', {'class': 'col-md-12 col-12 a_ico'})
        logger.debug('Retrieved max number of ICOs - {}'.format(len(icos)))
        icos_list = []
        for ico_num, ico in enumerate(icos):
            if ico:
                try:
                    ticker = ico.find('div', id='t_tikcer').text.replace('Ticker: ', '')
                    logger.debug('ticker: %s' % ticker)
                except Exception as e:
                    logger.debug(e)
                    ticker = self.not_available
                try:
                    ticker_full_name = ico.h3.a.text
                    logger.debug('ticker_full_name: %s' % ticker_full_name)
                except Exception as e:
                    logger.debug(e)
                    ticker_full_name = self.not_available
                try:
                    ico_website = ico.h3.find('a', href=True)['href']
                    logger.debug('ico_website: %s' % ico_website)
                except Exception as e:
                    logger.debug(e)
                    ico_website = self.not_available
                try:
                    ico_category_name = ico.find('div', class_='categ_type').text
                    logger.debug('ico_category_name: %s' % ico_category_name)
                except Exception as e:
                    logger.debug(e)
                    ico_category_name = self.not_available
                try:
                    goal = re.sub(r'\s+', '', ico.find('div', id='categ_desctop').text)
                    logger.debug('goal: %s' % goal)
                except Exception as e:
                    logger.debug(e)
                    goal = self.not_available
                try:
                    goal_received = ico.div.find('div', id='new_column_categ_invisted').span.text
                    logger.debug('goal_received: %s' % goal_received)
                except Exception as e:
                    logger.debug(e)
                    goal_received = self.not_available
                try:
                    on_exchanges = ico.find('div', id='t_tikcer')['title']
                    logger.debug('on_exchanges: %s' % on_exchanges)
                except Exception as e:
                    logger.debug(e)
                    on_exchanges = self.not_available
                logger.info('Collected ICO main details for %s' % ticker_full_name)
            else:
                ticker = ticker_full_name = ico_website = ico_category_name = \
                    goal = goal_received = on_exchanges = self.not_available

            ico_dict_load_ico = {'ticker': ticker, 'name': ticker_full_name, 'web_site': ico_website,
                                 'category_name': ico_category_name, 'goal': goal, 'goal_received': goal_received,
                                 'on_exchanges': on_exchanges}
            icos_list.append(ico_dict_load_ico)

        li_end_time = time.time()
        li_hours, li_rem = divmod(li_end_time - li_start_time, 3600)
        li_minutes, li_seconds = divmod(li_rem, 60)
        logger.info('Finalized general information collection for {}'.format(
            self.icodrops_exec_mode) + ' - execution time: {:0>2}:{:0>2}:{:05.2f}'.format(int(li_hours),
                                                                                          int(li_minutes),
                                                                                          li_seconds))
        return icos_list

    def get_details(self, ico_dict):
        """
        Retrieves from the icodrops specific page and returns the information for a given ico.
        :return:                    the ico_dict from init updated with all the detailed information
        """

        ei_start_time = time.time()
        logger = self.logger
        format_ico_dict = """ico_dict_load_ico = {'ticker': ticker, 'name': ticker_full_name, 'web_site': ico_website,
                                 'category_name': ico_category_name, 'goal': goal, 'goal_received': goal_received,
                                 'on_exchanges': on_exchanges}"""
        logger.info('Initialized detail collection for %s' % ico_dict['web_site'])

        gd_token_sale = gd_sale_date = gd_website = gd_whitepaper_link = gd_facebook = gd_reddit = \
            gd_github = gd_twitter = gd_telegram = gd_linkedin = \
            gd_medium = gd_slack = gd_btc = gd_youtube = gd_token_type = gd_ico_token_price = gd_fund_goal = \
            gd_total_tokens = gd_presale = gd_bonus_first = gd_available_for_sale = gd_whitelist = gd_accepts = \
            gd_kyc = gd_token_issue = gd_min_max_cap = gd_cant_participate = gd_bonus_presale = self.not_available

        driver = webdriver.Chrome(self.chromedriver_path)
        driver.get(ico_dict['web_site'])
        r = driver.page_source
        soup = BeautifulSoup(r, 'html.parser')

        right_panes = soup.find_all('div', {'class': 'ico-right-col'})
        for right_pane in right_panes:
            try:
                gd_sale_date = right_pane.find('div', class_='sale-date').text
                logger.debug('sale_date: %s' % gd_sale_date)
            except Exception as e:
                logger.debug(e)
                pass
            links = right_pane.find_all('a')
            for link in links:
                try:
                    if link.div.text == 'WEBSITE':
                        gd_website = link['href']
                        logger.debug('gd_website: %s' % gd_website)
                    elif link.div.text == 'WHITEPAPER':
                        gd_whitepaper_link = link['href']
                        logger.debug('gd_whitepaper_link: %s' % gd_whitepaper_link)
                except AttributeError:
                    if link.i['class'] == ['fa', 'fa-facebook-square']:
                        gd_facebook = link['href']
                        logger.debug('gd_facebook: %s' % gd_facebook)
                    elif link.i['class'] == ['fa', 'fa-reddit-alien']:
                        gd_reddit = link['href']
                        logger.debug('gd_reddit: %s' % gd_reddit)
                    elif link.i['class'] == ['fa', 'fa-github']:
                        gd_github = link['href']
                        logger.debug('gd_github: %s' % gd_github)
                    elif link.i['class'] == ['fa', 'fa-twitter']:
                        gd_twitter = link['href']
                        logger.debug('gd_twitter: %s' % gd_twitter)
                    elif link.i['class'] == ['fa', 'fa-telegram']:
                        gd_telegram = link['href']
                        logger.debug('gd_telegram: %s' % gd_telegram)
                    elif link.i['class'] == ['fa', 'fa-linkedin']:
                        gd_linkedin = link['href']
                        logger.debug('gd_linkedin: %s' % gd_linkedin)
                    elif link.i['class'] == ['fa', 'fa-medium']:
                        gd_medium = link['href']
                        logger.debug('gd_medium: %s' % gd_medium)
                    elif link.i['class'] == ['fa', 'fa-slack']:
                        gd_slack = link['href']
                        logger.debug('gd_slack: %s' % gd_slack)
                    elif link.i['class'] == ['fa', 'fa-btc']:
                        gd_btc = link['href']
                        logger.debug('gd_btc: %s' % gd_btc)
                    elif link.i['class'] == ['fa', 'fa-youtube']:
                        gd_youtube = link['href']
                        logger.debug('gd_youtube: %s' % gd_youtube)
                except Exception as e:
                    logger.debug(e)
                    pass

        ico_page_main = soup.find('div', class_="site-content", id="content")
        gd_white_desks = []
        try:
            gd_white_desks = [desk['class'] for desk in ico_page_main.find_all('div', class_='white-desk ico-desk')]
            logger.debug('gd_white_desks: %s' % gd_white_desks)
        except Exception as e:
            logger.debug(e)
            pass
        for i in range(len(gd_white_desks)):
            gd_white_desks = ico_page_main.find_all('div', class_='white-desk ico-desk')[i]
            titles_temp = [title_temp.text for title_temp in gd_white_desks.find_all('li')]
            token_sale_temp = [token_sale_temp.text for token_sale_temp in gd_white_desks.find_all('h4')]
            for k in token_sale_temp:
                if bool(re.search(r'TokenSale:(.*)', re.sub('\s+', '', k))):
                    try:
                        gd_token_sale = re.search(r'TokenSale:(.*)', re.sub('\s+', '', k)).group(1)
                        logger.debug('gd_token_sale: %s' % gd_token_sale)
                    except Exception as e:
                        logger.debug(e)
                        pass
                try:
                    if titles_temp[0][0:6] == 'Ticker':
                        for item in titles_temp:
                            if bool(re.search(r'Token type: (.*)', item)):
                                try:
                                    gd_token_type = re.search(r'Token type: (.*)', item).group(1)
                                    logger.debug('gd_token_type: %s' % gd_token_type)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'ICO Token Price: (.*)', item)):
                                try:
                                    gd_ico_token_price = re.search(r'ICO Token Price: (.*)', item).group(1)
                                    logger.debug('gd_ico_token_price: %s' % gd_ico_token_price)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Fundraising Goal: (.*)', item)):
                                try:
                                    gd_fund_goal = re.search(r'Fundraising Goal: (.*)', item).group(1)
                                    logger.debug('gd_fund_goal: %s' % gd_fund_goal)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Total Tokens: (.*)', item)):
                                try:
                                    gd_total_tokens = re.search(r'Total Tokens: (.*)', item).group(1)
                                    logger.debug('gd_total_tokens: %s' % gd_total_tokens)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Sold on presale: (.*)', item)):
                                try:
                                    gd_presale = re.search(r'Sold on presale: (.*)', item).group(1)
                                    logger.debug('gd_presale: %s' % gd_presale)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Bonus for the First: (.*)', item)):
                                try:
                                    gd_bonus_first = re.search(r'Bonus for the First: (.*)', item).group(1)
                                    logger.debug('gd_bonus_first: %s' % gd_bonus_first)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Available for Token Sale: (.*)', item)):
                                try:
                                    gd_available_for_sale = re.search(r'Available for Token Sale: (.*)',
                                                                      item).group(1)
                                    logger.debug('gd_available_for_sale: %s' % gd_available_for_sale)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Whitelist: (.*)', item)):
                                try:
                                    gd_whitelist = re.search(r'Whitelist: (.*)', item).group(1)
                                    logger.debug('gd_whitelist: %s' % gd_whitelist)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Accepts: (.*)', item)):
                                try:
                                    gd_accepts = re.search(r'Accepts: (.*)', item).group(1)
                                    logger.debug('gd_accepts: %s' % gd_accepts)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Know Your Customer (KYC): (.*)', item)):
                                try:
                                    gd_kyc = re.search(r'Know Your Customer (KYC): (.*)', item).group(1)
                                    logger.debug('gd_kyc: %s' % gd_kyc)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Token Issue: (.*)', item)):
                                try:
                                    gd_token_issue = re.search(r'Token Issue: (.*)', item).group(1)
                                    logger.debug('gd_token_issue: %s' % gd_token_issue)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Min/Max Personal Cap: (.*)', item)):
                                try:
                                    gd_min_max_cap = re.search(r'Min/Max Personal Cap: (.*)', item).group(1)
                                    logger.debug('gd_min_max_cap: %s' % gd_min_max_cap)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Сan\'t participate: (.*)', item)):
                                try:
                                    gd_cant_participate = re.search(r'Сan\'t participate: (.*)', item).group(1)
                                    logger.debug('gd_cant_participate: %s' % gd_cant_participate)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                            elif bool(re.search(r'Pre-sale Bonus: (.*)', item)):
                                try:
                                    gd_bonus_presale = re.search(r'Pre-sale Bonus: (.*)', item).group(1)
                                    logger.debug('gd_bonus_presale: %s' % gd_bonus_presale)
                                except Exception as e:
                                    logger.debug(e)
                                    pass
                except Exception as e:
                    logger.debug(e)
                    pass
        driver.close()
        gd_ico_dict_details = {'sales_dates': gd_token_sale,
                               'sale_date': gd_sale_date,
                               'details_website': gd_website,
                               'whitepaper_link': gd_whitepaper_link,
                               'facebook': gd_facebook,
                               'reddit': gd_reddit,
                               'github': gd_github,
                               'twitter': gd_twitter,
                               'telegram': gd_telegram,
                               'linkedin': gd_linkedin,
                               'medium': gd_medium,
                               'slack': gd_slack,
                               'btc': gd_btc,
                               'youtube': gd_youtube,
                               'token_type': gd_token_type,
                               'price_in_ico': gd_ico_token_price,
                               'found_goal': gd_fund_goal,
                               'total_tockens': gd_total_tokens,
                               'presale': gd_presale,
                               'bonus_first': gd_bonus_first,
                               'available_for_sale': gd_available_for_sale,
                               'whitelist': gd_whitelist,
                               'accepts': gd_accepts,
                               'kyc': gd_kyc,
                               'token_issue': gd_token_issue,
                               'min_max_cap': gd_min_max_cap,
                               'cant_participate': gd_cant_participate,
                               'bonus_presale': gd_bonus_presale}

        ico_dict.update(gd_ico_dict_details)

        ei_end_time = time.time()
        ei_hours, ei_rem = divmod(ei_end_time - ei_start_time, 3600)
        ei_minutes, ei_seconds = divmod(ei_rem, 60)
        logger.info('Finalized detail collection for {}'.format(
            ico_dict['web_site']) + ' - execution time: {:0>2}:{:0>2}:{:05.2f}'.format(int(ei_hours),
                                                                                       int(ei_minutes),
                                                                                       ei_seconds))

        return gd_ico_dict_details

    def set_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger


if __name__ == '__main__':
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option("display.max_rows", 999)
    chromedriver_path = "C:/chromedriver_win32/chromedriver.exe"
    base_path = 'C:/forecasting_winner_icos/'
    icodrops_path = 'C:/forecasting_winner_icos/icodrops/new/'
    icodrops_ended_list_path = icodrops_path + '01_icodrops_ended_list.csv'
    start_time = time.time()

    icodrops = IcodropsEnded(icodrops_ended_list_path, chromedriver_path, debug=True, ico_list=None)

    n_cores = multiprocessing.cpu_count() - 1
    pool = Pool(n_cores)
    data = pd.DataFrame(pool.map(icodrops.get_details, icodrops.ico_list))
    pool.close()
    pool.join()

    print(data)

    df = pd.DataFrame(data)
    df.to_csv(icodrops_ended_list_path, index_label='index')

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
