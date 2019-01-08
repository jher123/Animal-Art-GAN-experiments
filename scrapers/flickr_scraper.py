# Heavily inspired by https://gist.github.com/yunjey/14e3a069ad2aa3adf72dee93a53117d6
import flickrapi
import urllib
from PIL import Image
import requests
from io import BytesIO
import pathlib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", required=True, help="image keyword for searching the website")
    parser.add_argument("--num_images", type=int, required=True, help="how many images to download")
    opt = parser.parse_args()

    keyword = opt.keyword
    results_folder = "./" + keyword + "/"
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Flickr api access key
    flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

    photos = flickr.walk(text=keyword,
                         tag_mode='all',
                         tags=keyword,
                         extras='url_c',
                         per_page=100,  # can try different numbers
                         sort='relevance')

    urls = []
    for i, photo in enumerate(photos):
        print('Downloading image {}'.format(i))
        url = photo.get('url_c')
        if url is not None:
            urls.append(url)
            image_object = requests.get(url)
            image = Image.open(BytesIO(image_object.content))
            image.save(results_folder + keyword + '_' + str(i) + "." + image.format, image.format)

        if len(urls)> opt.num_images:
            break

if __name__ == "__main__":
    main()
