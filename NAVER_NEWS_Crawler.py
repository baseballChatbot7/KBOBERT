import pandas as pd
import numpy as np
from tqdm import tqdm
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time

def NAVER_NEWS_Crawler():
    driver = webdriver.Chrome('./chromedriver')

    df = pd.DataFrame(columns = ['title', 'content'])

    for date in tqdm(pd.date_range('20200101','20201231')):

        date = str(date)

        for page in range(1,100):

            url = 'https://sports.news.naver.com/kbaseball/news/index.nhn?date='+date[:4]+date[5:7]+date[8:10]+'&isphoto=N&type=popular&page='+str(page)

            driver.get(url=url)

            soup = bs(driver.page_source, 'html.parser')

            try:
                if int(driver.find_element_by_css_selector('#_pageList > strong').text) != page:
                    break
            except:
                break

            time.sleep(1)

            for article_url in set(soup.select('#_newsList')[0].findAll('a')):

                driver.get(url='https://sports.news.naver.com/news'+article_url.get('href')[20:])

                try:
                    title = driver.find_element_by_css_selector('#content > div > div.content > div > div.news_headline > h4').text
                except:
                    title = ''

                try:
                    content = driver.find_element_by_css_selector('#newsEndContents').text
                    content = content.replace('\n', '')
                except:
                    content = ''

                df = df.append({'title': title, 'content': content}, ignore_index=True)

                time.sleep(0.5)

    df = df.replace('',np.nan)

    df = df.dropna()

    df = df.drop_duplicates()

    df.to_csv('./NAVER_NEWS.csv', index=False, encoding='utf-8-sig')

    driver.close()

if __name__ == "__main__":
    NAVER_NEWS_Crawler()