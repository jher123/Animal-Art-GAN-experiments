import os
import requests
import time
import random
import pathlib
import argparse
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from selenium import webdriver

# Note for this to work geckodriver, selenium and Firefox need to be installed
# and geckodriver needs to be in /usr/local/bin/geckodriver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True, help="image keyword for searching the website")
    opt = parser.parse_args()

    keyword = opt.keyword

    results_folder = "./" + keyword + "/"  # '/Users/joannah/Documents/generative-art/lions1/'
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

    url = f'https://unsplash.com/search/photos/' + keyword
    driver = webdriver.Firefox(executable_path=r'/usr/local/bin/geckodriver')
    driver.get(url)

    # scrolling - inspired by https://stackoverflow.com/a/40779663
    lastHeight = driver.execute_script("return document.body.scrollHeight")
    pause = 0.5
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)
        newHeight = driver.execute_script("return document.body.scrollHeight")
        if newHeight == lastHeight:
            break
        lastHeight = newHeight

    # url = driver.page_source
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # searching using CSS selectors
    imgs = soup.select('#gridMulti img[src^="https://images.unsplash.com/photo"]')
    urls = []
    for i, img in enumerate(imgs):
        print('Downloading image {}'.format(i))
        image_url = str(img.extract()['src'])
        urls.append(image_url)
        image_object = requests.get(image_url)
        image = Image.open(BytesIO(image_object.content))
        image.save(results_folder + keyword + '_' + str(i) + "." + image.format, image.format)

if __name__ == "__main__":
    main()
