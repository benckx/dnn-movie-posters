import os
import sys

import movies_dataset as movies

download = '-download' in sys.argv
resize = '-resize' in sys.argv

min_year = 0
for arg in sys.argv:
    if arg.startswith('-min_year='):
        min_year = int(arg.split('=')[1])

ratios = [30, 40, 50, 60, 70]
images_path = 'data/images/'
base_images_path = images_path + '100/'

# create images directory if not exists
if not os.path.isdir(base_images_path):
    os.makedirs(base_images_path)

# download posters
if download:
    movies.download_posters(min_year=min_year)

# resize images (so we can build test models more quickly)
if resize:
    for ratio in ratios:
        directory_path = images_path + str(ratio)
        # directory = os.path.join(os.getcwd(), directory_path)
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
            command = 'mogrify -path "' + directory_path + '/" -resize ' \
                      + str(ratio) + '% ' + base_images_path + '*.jpg -verbose'
            print(command)
            os.system(command)
