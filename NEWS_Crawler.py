import pandas as pd
import numpy as np
from tqdm import tqdm
from selenium import webdriver
import time

def NEWS_Crawler():
    driver = webdriver.Chrome('./chromedriver')

    df = pd.DataFrame(columns = ['title', 'content'])

    for num in tqdm(range(46000)):
        url = 'https://www.koreabaseball.com/News/Preview/View.aspx?bdSe='+str(num)
        driver.get(url=url)
        title = driver.find_element_by_css_selector('#cphContents_cphContents_cphContents_lblTitle').text
        content = driver.find_element_by_css_selector('#contents > div.sub-content > div.view > div.detail').text
        content = content.replace('\n','')
        content = content.replace('[Copyright ⓒ KBO 홈페이지 뉴스, 기사, 사진은 KBO 홈페이지 자료 입니다. 무단전재 및 재배포는 금지되어 있으며 무단전재 및 재배포시 법적인 제재를 받을 수 있습니다.]', '')

        df.loc[num] = [title, content]

        time.sleep(0.5)

    df = df.replace('',np.nan)

    df = df.dropna()

    df.to_csv('./NEWS.csv', index=False, encoding='utf-8-sig')

    driver.close()

if __name__ == "__main__":
    NEWS_Crawler()