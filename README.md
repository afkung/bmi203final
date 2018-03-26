# bmi203final

[![Build
Status](https://travis-ci.org/afkung/bmi203final.svg?branch=master)](https://travis-ci.org/afkung/bmi203final)

Final Project for BMI203: Predicting TF Binding Sites Using Neural Nets

We create a 3-layer feed-forward artificial neural network, then train it to identify RAP1 transcription factor binding sites using a set of known positive and negative sequences. We then run this trained network on a novel test set, and output the results into `Predictions.txt`

## usage

To use the package, first run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `bmi203final/__main__.py`) can be run as
follows

```
python -m bmi203final
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project. A toy example of an auto-encoder is created and verified.