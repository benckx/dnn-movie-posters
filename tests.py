import operator
from os import listdir
from os.path import isfile, join

import numpy as np
from keras.models import load_model

import movies_dataset as movies

saved_models_folder = 'saved_models/'
align = 33
my_format = "{:.0%}"
eval_models = True
print_summary = True
print_test_movies = True
crop_results = 3


class TransferModel:
    min_year = 0
    max_year = 0
    genres = []
    ratio = 0
    epochs = 0
    version = 1
    file_path = ''
    model = None

    def eval(self):
        print('loading test data...')
        x_test, y_test = movies.load_genre_data(self.min_year, self.max_year, self.genres, self.ratio, data_type='test',
                                                verbose=False)
        print('Evaluating model...')
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def predict(self, movie):
        x = [movie.to_rgb_pixels(self.ratio)]
        x = np.array(x, dtype='float32')
        return self.model.predict(x)

    def load(self):
        self.model = load_model(self.file_path)

    def __str__(self):
        return 'Model v' + str(self.version) \
               + ' (' + str(self.min_year) + '-' + str(self.max_year) \
               + ' / g' + str(len(self.genres)) \
               + ' / r' + str(self.ratio) \
               + ' / e' + str(self.epochs) \
               + ')'


def parse_transfer_model(file_name):
    split = file_name.split('_')
    parsed = TransferModel()
    parsed.min_year = int(split[2])
    parsed.max_year = int(split[3])
    parsed.genres = movies.list_genres(int(split[4][1:]))
    parsed.ratio = int(split[5][1:])
    parsed.epochs = int(split[6][1:])
    parsed.version = int(split[7][1:].split('.')[0])
    parsed.file_path = file_name
    return parsed


def list_model_files():
    return sorted(
        [f for f in listdir(saved_models_folder) if isfile(join(saved_models_folder, f)) and f.startswith('genres_')])


def repeat_to_length(string_to_expand, length):
    return (string_to_expand * (int(length / len(string_to_expand)) + 1))[:length]


def format_predictions(movie, genres, predictions):
    predictions_map = dict()
    for i in range(len(genres)):
        predictions_map[genres[i]] = predictions[0][i]

    sorted_predictions = sorted(predictions_map.items(), key=operator.itemgetter(1), reverse=True)

    predictions_str = []
    for genre, probability in sorted_predictions:
        if genre in movie.genres:
            is_present = ''
        else:
            is_present = '[!]'
        predictions_str.append(genre + is_present + ': ' + my_format.format(probability))

    spaces = repeat_to_length(' ', align - len(str(movie)))

    if crop_results is not None:
        return str(movie) + spaces + str(predictions_str[:crop_results])
    else:
        return str(movie) + spaces + str(predictions_str)


def main():
    for model_file_name in list_model_files():
        saved_model = parse_transfer_model(saved_models_folder + model_file_name)
        saved_model.load()
        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------')
        print(saved_model)
        print('------------------------------------------------------------------------')
        if print_summary:
            print(saved_model.model.summary())
        if eval_models:
            saved_model.eval()

        test_movies = dict()

        test_movies['Comedy'] = ["Bienvenue chez les Ch'tis", "Frequently Asked Questions About Time Travel",
                                 "What We Do in the Shadows", "Hollywood Ending", "Whatever Works", "The Mask",
                                 "Liar Liar"]

        test_movies['Drama'] = ["No Country for Old Men", "The Martian", "Vanilla Sky"]

        test_movies['Action'] = ["The Matrix", "Man of Steel", "X-Men: Apocalypse", "Lara Croft: Tomb Raider",
                                 "Edge of Tomorrow", "Batman Forever", "Live Free or Die Hard"]

        test_movies['Horror'] = ["Dracula 2000", "The Blair Witch Project", "The Others", "Aliens",
                                 "Aliens vs. Predator: Requiem", "Alien: Resurrection"]

        test_movies['Animation'] = ["Paprika", "Castle in the Sky", "Spirited Away", "Zootopia", "Trolls"]

        test_movies['Romance'] = ["Notting Hill", "Pretty Woman", "Bridget Jones's Diary"]

        test_movies['?'] = ["Pearl Harbor", "Twelve Monkeys", "The Truman Show", "Blade Runner",
                            "Star Wars: Episode IV - A New Hope", "The Godfather", "A.I. Artificial Intelligence",
                            "Enter the Void", "The Abyss", "Primer", "Coherence", "Pulp Fiction"]

        if print_test_movies:
            print()
            for expected_genre, movies_titles in sorted(test_movies.items()):
                print(' -- ' + expected_genre + ' -- ')
                for movie_title in movies_titles:
                    movie = movies.search_movie(title=movie_title)
                    if movie is not None:
                        predictions = saved_model.predict(movie)
                        print(format_predictions(movie, saved_model.genres, predictions))
                    else:
                        print(movie_title + ' not found')
                print()


if __name__ == '__main__':
    main()
