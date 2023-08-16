
## Requisites

kaggle API (for downloading datasets) and Python 3.10

## Preparing datasets

```
$ kaggle competitions download -c cafa-5-protein-function-prediction
$ kaggle datasets download -d sergeifironov/t5embeds
```

## Install libraries

```
$ pip install -f requirements.txt
```

## Running the pipeline

```

$ python pipeline2.py

```