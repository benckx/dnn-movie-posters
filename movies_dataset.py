"""
Manage movies data (extracted from /data/MovieGenre.csv).
"""

import io
import os.path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

images_folder = 'data/images/'
test_data_ratio = 7  # 14.3%
validation_data_ratio = 6  # 14.3%
parsed_movies = []  # cache


class Movie:
    imdb_id = 0
    title = ''
    year = 0
    genres = []
    poster_url = ''

    def poster_file_exists(self) -> bool:
        return os.path.isfile(self.poster_file_name())

    def download_poster(self):
        try:
            response = urllib.request.urlopen(self.poster_url)
            data = response.read()
            file = open(self.poster_file_name(), 'wb')
            file.write(bytearray(data))
            file.close()
            return data
        except:
            print('-> error')

    def poster_file_name(self, size=100) -> str:
        return images_folder + str(size) + "/" + str(self.imdb_id) + '.jpg'

    def is_valid(self) -> bool:
        return self.poster_url.startswith('https://') \
               and 1900 <= self.year <= 2018 \
               and len(self.title) > 1 \
               and len(self.genres) > 1

    def to_rgb_pixels(self, poster_size):
        data = open(images_folder + str(poster_size) + '/' + str(self.imdb_id) + '.jpg', "rb").read()
        image = Image.open(io.BytesIO(data))
        rgb_im = image.convert('RGB')
        pixels = []
        for x in range(image.size[0]):
            row = []
            for y in range(image.size[1]):
                r, g, b = rgb_im.getpixel((x, y))
                pixel = [r / 255, g / 255, b / 255]
                row.append(pixel)
            pixels.append(row)

        return pixels

    def get_genres_vector(self, genres):
        if len(genres) == 1:
            has_genre = self.has_genre(genres[0])
            return [int(has_genre), int(not has_genre)]
        else:
            vector = []
            if self.has_any_genre(genres):
                for genre in genres:
                    vector.append(int(self.has_genre(genre)))

            return vector

    def short_title(self) -> str:
        max_size = 20
        return (self.title[:max_size] + '..') if len(self.title) > max_size else self.title

    def is_test_data(self) -> bool:
        return self.imdb_id % test_data_ratio == 0

    def has_any_genre(self, genres) -> bool:
        return len(set(self.genres).intersection(genres)) > 0

    def has_genre(self, genre) -> bool:
        return genre in self.genres

    def __str__(self):
        return self.short_title() + ' (' + str(self.year) + ')'


def download_posters(min_year=0):
    for movie in list_movies():
        print(str(movie))
        if movie.year >= min_year:
            if not movie.poster_file_exists():
                movie.download_poster()
                if movie.poster_file_exists():
                    print('-> downloaded')
                else:
                    print('-> could not download')
            else:
                print('-> already downloaded')
        else:
            print('-> skip (too old)')


def load_genre_data(min_year, max_year, genres, ratio, data_type, verbose=True):
    xs = []
    ys = []

    for year in reversed(range(min_year, max_year + 1)):
        if verbose:
            print('loading movies', data_type, 'data for', year, '...')

        xs_year, ys_year = _load_genre_data_per_year(year, genres, ratio, data_type)
        _add_to(xs_year, xs)
        _add_to(ys_year, ys)

        if verbose:
            print('->', len(xs_year))

    return np.concatenate(xs), np.concatenate(ys)


def _load_genre_data_per_year(year, genres, poster_ratio, data_type):
    xs = []
    ys = []

    count = 1
    for movie in list_movies(year, genres):
        if movie.poster_file_exists():
            if (data_type == 'train' and not movie.is_test_data() and count % validation_data_ratio != 0) \
                    or (data_type == 'validation' and not movie.is_test_data() and count % validation_data_ratio == 0) \
                    or (data_type == 'test' and movie.is_test_data()):
                x = movie.to_rgb_pixels(poster_ratio)
                y = movie.get_genres_vector(genres)
                xs.append(x)
                ys.append(y)
            count += 1

    xs = np.array(xs, dtype='float32')
    ys = np.array(ys, dtype='uint8')
    return xs, ys


def _add_to(array1d, array2d):
    if len(array1d) > 0:
        array2d.append(array1d)


def list_movies(year=None, genres=None):
    if len(parsed_movies) == 0:
        data = pd.read_csv('data/MovieGenre.csv', encoding='ISO-8859-1')
        for index, row in data.iterrows():
            movie = _parse_movie_row(row)
            if movie.is_valid():
                parsed_movies.append(movie)

        parsed_movies.sort(key=lambda m: m.imdb_id)

    result = parsed_movies

    if year is not None:
        result = [movie for movie in result if movie.year == year]

    if genres is not None:
        result = [movie for movie in result if movie.has_any_genre(genres)]

    return result


def _parse_movie_row(row) -> Movie:
    movie = Movie()
    movie.imdb_id = int(row['imdbId'])
    movie.title = row['Title'][:-7]
    year = row['Title'][-5:-1]
    if year.isdigit() and len(year) == 4:
        movie.year = int(row['Title'][-5:-1])

    url = str(row['Poster'])
    if len(url) > 0:
        movie.poster_url = url.replace('"', '')

    genre_str = str(row['Genre'])
    if len(genre_str) > 0:
        movie.genres = genre_str.split('|')

    return movie


def search_movie(imdb_id=None, title=None) -> Movie:
    movies = list_movies()
    for movie in movies:
        if imdb_id is not None and movie.imdb_id == imdb_id:
            return movie
        if title is not None and movie.title == title:
            return movie


def list_genres(number):
    if number == 3:
        return ['Comedy', 'Drama', 'Action']
    if number == 7:
        return ['Comedy', 'Drama', 'Action', 'Animation', 'Romance', 'Adventure', 'Horror']
