import movies_dataset as movies
import movies_genre_model

min_year = 1977
max_year = 2017
epochs = 50
genres = movies.list_genres(7)

# select a smaller ratio (e.g. 40) for quicker training
for ratio in [60]:
    # we load the data once for each ratio, so we can use it for multiple versions, epochs, etc.
    x_train, y_train = movies.load_genre_data(min_year, max_year, genres, ratio, 'train')
    x_validation, y_validation = movies.load_genre_data(min_year, max_year, genres, ratio, 'validation')
    for version in [1, 2, 3]:
        movies_genre_model.build(version, min_year, max_year, genres, ratio, epochs,
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_validation=x_validation,
                                 y_validation=y_validation)
