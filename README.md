## About
Use Convolutional Neural Network (CNN) to classify movies posters by genres. The implementation is based on 
Keras and TensorFlow.

It is a multi-label classification problem (movies can belong to multiple genres). The results look like this:

|__Movies__                  |               |               |               |
|---                         |---            |---            |---            |
|_What we do in the shadows_ |Comedy: 64%    |Horror: 30%    |Adventure: 17% |
|_The Matrix_                |Action: 64%    |Adventure: 30% |Drama: 12%     |
|_Zootopia_                  |Comedy: 71%    |Animation: 68% |Adventure: 34% |
|_Notting Hill_              |Comedy: 88%    |Romance: 85%   |Drama: 62%     |
Each instance (movie) has an independent probability to belong to each class (genre). 

### Data set
The data set was found on [Kaggle](https://www.kaggle.com/neha1703/movie-genre-from-its-poster/version/3) and contains 
about 27,000 posters.

It is split as followed:
* __Training__: 5/7
* __Validation__: 1/7
* __Test__: 1/7

Module `movies_dataset.py` provides functions to access the data set easily (parse MovieGenres.csv, list movies, 
get movie genres, get poster, etc).

### Model parameters
* __min_year__ and __max_year__: Movie release time range (e.g. from 1977 to 2017). 
Posters design is very dependent on release year, therefore using a larger time range might increase noise. 
* __genres__: Classes. In the current configuration, genres are grouped by 3 (Comedy, Drama, Action) 
or 7 (idem + Animation, Romance, Adventure, Horror).
* __ratio__: Original pictures size is 182x268 (ratio 100). You can use a smaller pictures for quicker 
(but probably less accurate) model training (30, 40, 50, etc). You can resize the posters with `get_data.py -resize`. 
* __epochs__: Number of epochs.
* __version__: Version of the model. Different versions can have different parameters (e.g. kernel size, etc), 
so different configurations can be compared easily.

## How to
### Modules prerequisites
* Python 3.5
* [tensorflow](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv)
* [Keras](https://keras.io/#installation)
* pandas
* h5py

### Get posters data
Use flag `-download` to download the posters from Amazon (based on the URLs provided in MovieGenre.csv).
Use flag `-resize` to create smaller posters (30%, 40%, etc). 
```
python3 get_data.py -download -resize
```

### Train the model
This script builds and trains models. Models are saved to 'saved_models'. One or multiple models
(with different parameters) can be produced.
```
python3 __main__.py
```

### Evaluate the model and test predictions
This script iterates through all the saved models in 'saved_models' and evaluates them on the test data.
```
python3 tests.py
```
### Train in the cloud
I use AWS EC2 with this [AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C). No packages install is required.