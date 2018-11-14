# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python (emocontext)
#     language: python
#     name: emocontext
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# +
# %autosave 0
#Please use python 3.5 or above
import numpy as np
import json, argparse, os
import re
import io
import pandas as pd
import sys
import matplotlib as plt
import nltk
from nltk.tokenize import TweetTokenizer

from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from keras.models import load_model
from keras.utils import to_categorical

from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# -

def preprocessData(dataFilePath, mode):
    ids = []
    t1s = []
    t2s = []
    t3s = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                labels.append(line[4])

            t1s.append(line[1])
            t2s.append(line[2])
            t3s.append(line[3])
            ids.append(int(line[0]))
    
    if mode == "train":
        df = pd.DataFrame(data=np.column_stack((ids, t1s, t2s, t3s, labels)), columns=["id", "turn1", "turn2", "turn3", "label"])
        emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}
        df["label_int"] = df["label"].apply(lambda x: emotion2label[x])
        
    else:
        df = pd.DataFrame(data=np.column_stack((ids, t1s, t2s, t3s)), columns=["id", "turn1", "turn2", "turn3"])
        
    df = df.set_index("id")
    return df

def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.twitter.27B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix    

def buildModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
#     model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
#     model.add(Dense(NUM_CLASSES, activation='sigmoid'))

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model

def add_turn_number(turn, turn_int):
    out = [ str(turn_int) + "_" + w for w in turn.split(" ") ]
    return " ".join(out)

def append_t1_t3(df):
    df["feature"] = df.apply(lambda r: r["turn1"] + " " + r["turn3"], axis=1)    
    return df

# ## GLOBALS

label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

# +
config = {
    "train_data_path": "../../data/train.txt",
    "test_data_path": "../../data/devwithoutlabels.txt",
    "solution_path" : "../../data/test.txt",
    "glove_dir" : "../../data/glove.twitter.27B",
    "num_folds" : 5,
    "num_classes" : 4,
    "max_nb_words" : 20000,
    "max_sequence_length" : 100,
    "embedding_dim" : 100,
    "batch_size" : 200,
    "lstm_dim" : 128,
    "learning_rate" : 0.003,
    "dropout" : 0.2,
    "num_epochs" : 75
}

TRAIN_DATA_PATH = config["train_data_path"]
TEST_DATA_PATH = config["test_data_path"]
SOLUTION_PATH = config["solution_path"]
GOLVE_DIR = config["glove_dir"]

NUM_FOLDS = config["num_folds"]
NUM_CLASSES = config["num_classes"]
MAX_NB_WORDS = config["max_nb_words"]
MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
EMBEDDING_DIM = config["embedding_dim"]
BATCH_SIZE = config["batch_size"]
LSTM_DIM = config["lstm_dim"]
DROPOUT = config["dropout"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
# -

# ## ONE HOT

# +
dftrain = preprocessData(TRAIN_DATA_PATH, mode="train")
# encode turn info
dftrain["turn1"] = dftrain["turn1"].apply(lambda x: add_turn_number(x, 1))
dftrain["turn3"] = dftrain["turn3"].apply(lambda x: add_turn_number(x, 3))
dftrain = append_t1_t3(dftrain)

# ONE HOT
onehot = CountVectorizer(binary=True)
onehot.fit(dftrain.iloc[[15]].feature.values)
# -

onehot.transform(dftrain.iloc[[15]].feature.values).toarray()

# +
# TRAINING
# read in data & preprocess
dftrain = preprocessData(trainDataPath, mode="train")

# encode turn info
dftrain["turn1"] = dftrain["turn1"].apply(lambda x: add_turn_number(x, 1))
dftrain["turn3"] = dftrain["turn3"].apply(lambda x: add_turn_number(x, 3))
dftrain = append_t1_t3(dftrain)

# ONE HOT
onehot = CountVectorizer(stop_words='english', binary=True)
onehot.fit(dftrain.feature.values)
train_embed = onehot.transform(dftrain.feature)

# Train model
model = MultinomialNB()
model.fit(train_embed, dftrain.label_int)

# +
# TESTING
# read in data & preprocess
dftest = preprocessData(testDataPath, mode="test")

# encode turn info
dftest["turn1"] = dftest["turn1"].apply(lambda x: add_turn_number(x, 1))
dftest["turn3"] = dftest["turn3"].apply(lambda x: add_turn_number(x, 3))
dftest = append_t1_t3(dftest)

# TFIDF
test_embed = onehot.transform(dftest.feature)

# Output test results
pred = model.predict(test_embed)
dftest = pd.read_csv(testDataPath, sep="\t", index_col=0)
dftest["label"] = [ label2emotion[x] for x in pred]
dftest.to_csv(solutionPath, sep="\t")
# -

# ## ONE HOT NLTK.NAIVE_BAYES

# +
# TRAINING
# read in data & preprocess
dftrain = preprocessData(trainDataPath, mode="train")

# encode turn info
dftrain["turn1"] = dftrain["turn1"].apply(lambda x: add_turn_number(x, 1))
dftrain["turn3"] = dftrain["turn3"].apply(lambda x: add_turn_number(x, 3))
dftrain = append_t1_t3(dftrain)

# ONE HOT
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

vocab = list(set([w for f in dftraining['feature'] for w in f.split(" ")]))

dftrain['feature_dict'] = [ document_features(d) for d in dftrain['feature'] ]


# Train model
model = MultinomialNB()
model.fit(train_embed, dftrain.label_int)
# -

# ## TFIDF

# +
# TRAINING
# read in data & preprocess
dftrain = preprocessData(trainDataPath, mode="train")

# encode turn info
dftrain["turn1"] = dftrain["turn1"].apply(lambda x: add_turn_number(x, 1))
dftrain["turn3"] = dftrain["turn3"].apply(lambda x: add_turn_number(x, 3))
dftrain = append_t1_t3(dftrain)

# TFIDF
tfidf = TfidfVectorizer(max_features=MAX_NB_WORDS, stop_words='english')
tfidf.fit(dftrain.feature.values)
train_embed = tfidf.transform(dftrain.feature)

# Train model
model = GaussianNB()
model.fit(train_embed.toarray(), dftrain.label_int)

# +
# TESTING
# read in data & preprocess
dftest = preprocessData(testDataPath, mode="test")

# encode turn info
dftest["turn1"] = dftest["turn1"].apply(lambda x: add_turn_number(x, 1))
dftest["turn3"] = dftest["turn3"].apply(lambda x: add_turn_number(x, 3))
dftest = append_t1_t3(dftest)

# TFIDF
test_embed = tfidf.transform(dftest.feature)

# Output test results
pred = model.predict(test_embed.toarray())
dftest = pd.read_csv(testDataPath, sep="\t", index_col=0)
dftest["label"] = [ label2emotion[x] for x in pred]
dftest.to_csv(solutionPath, sep="\t")
# -

# ## DEEP LEARNING

# +
config = {
    "train_data_path": "../../data/train.txt",
    "test_data_path": "../../data/devwithoutlabels.txt",
    "solution_path" : "../../data/test.txt",
    "glove_dir" : "../../data/glove.twitter.27B",
    "num_folds" : 5,
    "num_classes" : 4,
    "max_nb_words" : 20000,
    "max_sequence_length" : 100,
    "embedding_dim" : 100,
    "batch_size" : 200,
    "lstm_dim" : 128,
    "learning_rate" : 0.003,
    "dropout" : 0.2,
    "num_epochs" : 75
}

trainDataPath = config["train_data_path"]
testDataPath = config["test_data_path"]
solutionPath = config["solution_path"]
gloveDir = config["glove_dir"]

NUM_FOLDS = config["num_folds"]
NUM_CLASSES = config["num_classes"]
MAX_NB_WORDS = config["max_nb_words"]
MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
EMBEDDING_DIM = config["embedding_dim"]
BATCH_SIZE = config["batch_size"]
LSTM_DIM = config["lstm_dim"]
DROPOUT = config["dropout"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    tweetTkr = TweetTokenizer(preserve_case=False)
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')

            # Preprocess sents turns
            line[1] = " ".join(tweetTkr.tokenize(line[1]))
            line[2] = " ".join(tweetTkr.tokenize(line[2]))
            line[3] = " ".join(tweetTkr.tokenize(line[3]))

            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations

def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.twitter.27B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def buildModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model
# -

print("Processing training data...")
trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
print("Processing test data...")
testIndices, testTexts = preprocessData(testDataPath, mode="test")

t = pd.read_csv(trainDataPath, sep='\t', index_col=False)

t

trainTexts[30144]

# +
print("Processing training data...")
trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
print("Processing test data...")
testIndices, testTexts = preprocessData(testDataPath, mode="test")

print("Extracting tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='')
tokenizer.fit_on_texts(trainTexts)
trainSequences = tokenizer.texts_to_sequences(trainTexts)
testSequences = tokenizer.texts_to_sequences(testTexts)

wordIndex = tokenizer.word_index
print("Found %s unique tokens." % len(wordIndex))

print("Populating embedding matrix...")
embeddingMatrix = getEmbeddingMatrix(wordIndex)

data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print("Shape of training data tensor: ", data.shape)
print("Shape of label tensor: ", labels.shape)

# Randomize data
np.random.shuffle(trainIndices)
data = data[trainIndices]
labels = labels[trainIndices]

# Perform k-fold cross validation
metrics = {"accuracy" : [],
           "microPrecision" : [],
           "microRecall" : [],
           "microF1" : []}

# print("Starting k-fold cross validation...")
# for k in range(NUM_FOLDS):
#     print('-'*40)
#     print("Fold %d/%d" % (k+1, NUM_FOLDS))
#     validationSize = int(len(data)/NUM_FOLDS)
#     index1 = validationSize * k
#     index2 = validationSize * (k+1)

#     xTrain = np.vstack((data[:index1],data[index2:]))
#     yTrain = np.vstack((labels[:index1],labels[index2:]))
#     xVal = data[index1:index2]
#     yVal = labels[index1:index2]
#     print("Building model...")
#     model = buildModel(embeddingMatrix)
#     model.fit(xTrain, yTrain,
#               validation_data=(xVal, yVal),
#               epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

#     predictions = model.predict(xVal, batch_size=BATCH_SIZE)
#     accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
#     metrics["accuracy"].append(accuracy)
#     metrics["microPrecision"].append(microPrecision)
#     metrics["microRecall"].append(microRecall)
#     metrics["microF1"].append(microF1)

# print("\n============= Metrics =================")
# print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
# print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
# print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
# print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

print("\n======================================")

print("Retraining model on entire data to create solution file")
model = buildModel(embeddingMatrix)
model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
# model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

print("Creating solution file...")
testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
predictions = model.predict(testData, batch_size=BATCH_SIZE)
predictions = predictions.argmax(axis=1)

with io.open(solutionPath, "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
    with io.open(testDataPath, encoding="utf8") as fin:
        fin.readline()
        for lineNum, line in enumerate(fin):
            fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
            fout.write(label2emotion[predictions[lineNum]] + '\n')
print("Completed. Model parameters: ")
print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
      % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))


# +
from keras.preprocessing.text import Tokenizer

corp = [
    "red =) :) ;(",
    "yellow =) =DDD",
    "😂😍🎉👍 -_- >_>",
    "I can start working out again!!!!! :‑c o-o",
    "<3 s2",
    "=:o]"
]
tweet_tkr = TweetTokenizer(preserve_case=False)
_corp = [tweet_tkr.tokenize(c) for c in corp]
corp = [ " ".join(tweet_tkr.tokenize(c)) for c in corp]
tkr = Tokenizer(filters='')
tkr.fit_on_texts(corp)
# -

_corp

corp
