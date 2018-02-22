import movies_dataset as movies
import movies_genre_model

min_year = 1977
max_year = 2017
epochs = 1
genres = movies.list_genres(7)

for ratio in [30]:
    # we load the data once for each ratio, so we can use it for multiple versions, epochs, etc.
    (x_train, y_train), (x_validation, y_validation) = movies.load_genre_data(min_year, max_year, genres, ratio)
    for version in [1, 2]:
        movies_genre_model.build(version, min_year, max_year, genres, ratio, epochs,
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_validation=x_validation,
                                 y_validation=y_validation)
