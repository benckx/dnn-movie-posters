## About
Use Convolutional Neural Network (CNN) to classify movies posters by genres.

It is a multi-label classification exercise. The results look like this:

|__Movies__                  |               |               |               |
|---                         |---            |---            |---            |
|_What we do in the shadows_ |Comedy: 64%    |Horror: 30%    |Adventure: 17% |
|_The Matrix_                |Action: 64%    |Adventure: 30% |Drama: 12%     |
|_Zootopia_                  |Comedy: 71%    |Animation: 68% |Adventure: 34% |
|_Notting Hill_              |Comedy: 88%    |Romance: 85%   |Drama: 62%     |

### Data sets
The data is split as followed:
* __Training__: 5/7
* __Validation__: 1/7
* __Test__: 1/7

### Model parameters
* __min_year__ and __max_year__
* __genres__
* __ratio__
* __epochs__
* __version__

## How to

### Get posters data
```
python3 get_data.py -download -resize
```

### Train the model
```
python3 __main__.py
```

### Evaluate the model
```
python3 tests.py
```
This script iterates through all the saved models in 'saved_models' and evaluates it on the test data. It also prints a few examples. 
