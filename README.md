## About
A simple demo / tutorial / experiment / portfolio project for me to better understand the concepts of Machine Learning.

##Classify Movies by Genre 
Use Convolutional Neural Network (CNN) to classify movies posters by genre. It is a multi-label classification problem 
(movies can belong to multiple genres). Each instance (movie poster) has an independent probability to belong to each label (genre).

The implementation is based on Keras and TensorFlow.

With 14,265 train samples and 2,826 validation samples (movies from 1977 to 2017), 106x161 images and after 50 epochs,
the results look like this ([!] indicates the predicted genre is not found in the original data set):


![](https://images-na.ssl-images-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_UX182_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BMTAxMDE4Mzc3ODNeQTJeQWpwZ15BbWU4MDY2Mjg4MDcx._V1_UY268_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BM2YxYmFjYWMtMzBmMC00MTVmLThhMjUtYWU5Yzg2OGQwZjE3XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_UX182_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BMTc2MTQ3MDA1Nl5BMl5BanBnXkFtZTgwODA3OTI4NjE@._V1_UX182_CR0,0,182,268_AL_.jpg
)
```
The Matrix (1999)                ['Action: 91%', 'Drama[!]: 25%', 'Adventure[!]: 13%']
The Others (2001)                ['Drama[!]: 76%', 'Horror: 65%', 'Action[!]: 41%']
Alien: Resurrection (1997)       ['Horror: 67%', 'Action: 64%', 'Drama[!]: 43%']
The Martian (2015)               ['Drama: 95%', 'Adventure: 81%', 'Comedy[!]: 23%']
```
\
![](https://images-na.ssl-images-amazon.com/images/M/MV5BMDIzODcyY2EtMmY2MC00ZWVlLTgwMzAtMjQwOWUyNmJjNTYyXkEyXkFqcGdeQXVyNDk3NzU2MTQ@._V1_UX182_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BNjk2ODQzNDYxNV5BMl5BanBnXkFtZTgwMTcyNDg4NjE@._V1_UX182_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BMTU2NTA4NzgyNl5BMl5BanBnXkFtZTcwNzEzMjQ1Mg@@._V1_UX182_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BMjEwNzMzMzUxOV5BMl5BanBnXkFtZTcwODcyODA4MQ@@._V1_UY268_CR9,0,182,268_AL_.jpg)
```
The Truman Show (1998)           ['Comedy: 98%', 'Drama: 76%', 'Romance[!]: 7%']
Pretty Woman (1990)              ['Romance: 99%', 'Comedy: 99%', 'Drama[!]: 22%']
Whatever Works (2009)            ['Drama[!]: 86%', 'Comedy: 78%', 'Romance: 76%']
Bienvenue chez les C.. (2008)    ['Comedy: 98%', 'Romance: 98%', 'Drama[!]: 7%']
```
\
![](https://images-na.ssl-images-amazon.com/images/M/MV5BNDI4MGEwZDAtZDg0Yy00MjFhLTg1MjctODdmZTMyNTUyNDI3L2ltYWdlXkEyXkFqcGdeQXVyNTAyODkwOQ@@._V1_UX182_CR0,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BMTk3NTM1NTg1Ml5BMl5BanBnXkFtZTgwOTgzMTMyMDE@._V1_UY268_CR6,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BNTg0NmI1ZGQtZTUxNC00NTgxLThjMDUtZmRlYmEzM2MwOWYwXkEyXkFqcGdeQXVyMzM4MjM0Nzg@._V1_UY268_CR2,0,182,268_AL_.jpg)&nbsp;&nbsp;
![](https://images-na.ssl-images-amazon.com/images/M/MV5BOTMyMjEyNzIzMV5BMl5BanBnXkFtZTgwNzIyNjU0NzE@._V1_UX182_CR0,0,182,268_AL_.jpg)
```
Paprika (2006)                   ['Animation: 66%', 'Comedy[!]: 58%', 'Adventure: 31%']
Spirited Away (2001)             ['Animation: 83%', 'Drama[!]: 57%', 'Adventure: 42%']
Castle in the Sky (1986)         ['Animation: 88%', 'Adventure: 78%', 'Comedy[!]: 30%']
Zootopia (2016)                  ['Animation: 62%', 'Adventure: 59%', 'Comedy: 49%']
```
\
Overall accuracy is 45% (I'm actually not sure it's the most suited metrics for this). 

## Data set
The data set was found on [Kaggle](https://www.kaggle.com/neha1703/movie-genre-from-its-poster/version/3) and contains 
about 27,000 posters.

It is split as followed:
* __Training__: 5/7
* __Validation__: 1/7
* __Test__: 1/7

Module `movies_dataset.py` provides functions to access the data set easily (parse MovieGenres.csv, list movies, 
get movie genres, get poster, etc).

### Data set Parameters
* __min_year__ and __max_year__: Movie release time range (e.g. from 1977 to 2017). 
Posters design is very dependent on release year, therefore using a larger time range might increase noise. 
* __genres__: Classes. In the current configuration, genres are grouped by 3 (Comedy, Drama, Action) 
or 7 (idem + Animation, Romance, Adventure, Horror).
* __ratio__: Original pictures size is 182x268 (ratio 100). You can use a smaller pictures for quicker 
(but probably less accurate) model training (30, 40, 50, etc).

### Model Parameters
* __epochs__: Number of epochs.
* __version__: Version of the model. Different versions can have different parameters (e.g. kernel size, etc), 
so different configurations can be compared easily.

## How to
### Linux prerequisites
* imagemagick (to resize the original poster image files)

### Modules prerequisites
* Python 3.5
* [tensorflow 1.5.0](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv)
* [Keras 2.1.4](https://keras.io/#installation)
* pandas 0.22.0
* h5py 2.7.1

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
I use AWS EC2 with this [AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C). No packages install is required 
(use `source activate tensorflow_p36` to activate the correct Virtualenv environment).

## Generate Posters with DCGAN

Use [Deep Convolutional Generative Adversarial Networks (DCGAN)](https://github.com/Newmu/dcgan_code) to generate movie posters:

<p align="center">
  <a href="https://i.imgur.com/a07ebkC.jpg">
    <img width="70%" src="https://i.imgur.com/a07ebkC.jpg">
  </a>
</p>

[Watch training video](https://youtu.be/oz9dezDHUXs)

### How to

1\. Download [DCGAN-tensorflow](https://github.com/benckx/DCGAN-tensorflow):
```
git clone https://github.com/benckx/DCGAN-tensorflow.git
```
2\. Install the required virtualenv via the script (DCGAN-tensorflow works with an older version of TenserFlow):
```
./install_dcgan_env.sh
```
3\. Activate the environment: 
```
source ~/tensorflow_DCGAN_env/bin/activate
```
4\. Prepare data set with the parameters you want:
```
prepare_dcgan_dataset.py -min_year=1980 -exclude_genres=Animation,Comedy,Family -ratio=60
```
This will create a folder 'movies_posters' with all the posters.

5\. Copy folder 'dcgan_movies_posters' to /DCGAN-tensorflow/data/

6\. In DCGAN-tensorflow, run the command:
```
python3 main.py --dataset dcgan_movies_posters --grid_height=6 --grid_width=10 --train
```

## Going Further
A few things I'm currently working on or thinking about:

* Predict movie release year / rating from poster
* Improve model versioning to compare different settings
