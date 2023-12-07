# Movie-Recommender-System
M111 - Big Data - Fall 2023 - Project


# Commands

## Set-up
```
pip install .
```

## Execute App
```
python .\src\recommender.py -d 'data/ml-latest/' -n 10 -s pearson -a user -i 10 -preprocess 1 -nrows 100000
python .\src\recommender.py -d 'data/ml-100k/' -n 5 -s dice -a hybrid -i 1
```

## Execute Pre-process
```
python .\src\preprocess.py -d 'data/ml-latest/' -nrows 1000000 -u 10
```
