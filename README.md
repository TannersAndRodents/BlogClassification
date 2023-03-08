# BlogClassification

Simple neural network with keras to determine the age of a blogger using the [Blog Authorship Corpus](https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) by J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). It tries to discriminate between bloggers yonger than 20, between 20 and 30 and older than 30.

The code is heavily inspired by [this](https://keras.io/examples/nlp/text_classification_from_scratch/) article by Mark Omernick and Francois Chollet from 2019.

## Run

Clone repo and run the setup file `downloadAndPrepare.sh`, which will download and unzip the dataset, then run `PrepareDataset.py` to extract the blog posts from the XML files and to put them into folders according to the correct label. Then use `BlogClassification.py` to train and evaluate.

On my laptop it takes about 10 – 20 minutes to train, the dataset is about 800 MB.

## Results

My results are rather bad with a accuracy of about 45%, which is still better than guessing with three categories, but well ...

**Training:**
```13673/13673 [==============================] - 649s 47ms/step - loss: 0.4610 - accuracy: 0.4820 - val_loss: 0.5042 - val_accuracy: 0.3953```

**Evalutation of test split:**
```2066/2066 [==============================] - 21s 10ms/step - loss: 0.4146 - accuracy: 0.4596```

Currently, only one epoch is being trained. To optimize this further (or rather: at all), future stepts could be to remove short blogs with only a link, or find better categories to train on (e.g. gener, industry, ...)
