import os
import sys
from shutil import copyfile

import movies_dataset as movies

min_year = 0
max_year = 2017
ratio = 50
genres_white_list = []
genres_black_list = []
genres = []
path = 'dcgan_movies_posters'

# parse arguments
for arg in sys.argv:
    if arg.startswith('-min_year='):
        min_year = int(arg.split('=')[1])

    if arg.startswith('-max_year='):
        max_year = int(arg.split('=')[1])

    if arg.startswith('-ratio='):
        ratio = int(arg.split('=')[1])

    if arg.startswith('-include_genres='):
        genres_white_list = arg.split('=')[1].split(',')

    if arg.startswith('-exclude_genres='):
        genres_black_list = arg.split('=')[1].split(',')

# filter genres
if len(genres_white_list) > 0:
    genres = genres_white_list
else:
    genres = list(set(movies.list_genres(14)) - set(genres_black_list))

genres.sort()

# log
print()
print('min_year:', min_year)
print('max_year:', max_year)
print('genres:', genres)
print('ratio:', ratio)
print()

# collect movies
all_movies = []
for year in reversed(range(min_year, max_year + 1)):
    all_movies += movies.list_movies(year=year, genres=genres)

print('movies collected:', len(all_movies))

# create folder if doesn't exist
if not os.path.isdir(path):
    os.makedirs(path)

# delete content if any
for the_file in os.listdir(path):
    file_path = os.path.join(path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

# copy files
for movie in all_movies:
    if movie.poster_file_exists():
        copyfile(movie.poster_file_path(ratio), path + '/' + movie.poster_file_name())

# TODO: not very clean
# convert all to RGB
command = './convert_to_RGB.sh'
print(command)
os.system(command)
