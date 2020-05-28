import pandas as pd
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import logging
import numpy as np
import nltk.data

nltk.download()

train = pd.read_csv("trainDataset.csv", encoding="latin1", header=0, error_bad_lines=False,
                    delimiter=",", quotechar='"', skipinitialspace=True, lineterminator='\n', quoting=3)
print(train.shape)

test = pd.read_csv("datasetDisney.csv", encoding="latin1", header=0, delimiter=",", quotechar='"', error_bad_lines=False,
                   skipinitialspace=True, lineterminator='\n', quoting=3)
print(test.shape)
train

def textsToWordlist(text, remove_stopwords=False):
    text = BeautifulSoup(text, features="html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(r'/((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*/ ', "URL", text)
    text = re.sub(r'/#\S+*/', "HASHTAG", text)
    text = re.sub(r'/@\S+*/', "USER", text)
    words = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def textToSentences(text, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(text.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(textsToWordlist(raw_sentence, remove_stopwords))
    return sentences

sentences = []
for text in train["text"]:
    sentences += textToSentences(text, tokenizer)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling, seed=1)

model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(texts, model, num_features):
    counter = 0.
    textFeatureVecs = np.zeros((len(texts), num_features), dtype="float32")
    for text in texts:
        textFeatureVecs[int(counter)] = makeFeatureVec(text, model, num_features)
        counter = counter + 1.
    return textFeatureVecs

clean_train_texts = []
for text in train["text"]:
    clean_train_texts.append(textsToWordlist(text, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_texts, model, num_features)
trainDataVecs = Imputer().fit_transform(trainDataVecs)

clean_test_texts = []
for text in test["twitterText"]:
    clean_test_texts.append(textsToWordlist(text, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_texts, model, num_features)
testDataVecs = Imputer().fit_transform(testDataVecs)
forest = Pipeline([("scale", StandardScaler()),
                   ("forest", RandomForestClassifier(n_estimators=100))])
forest = forest.fit(trainDataVecs, train["sentiment"])

result = forest.predict(testDataVecs)

output = pd.DataFrame(data={"id": test["date"], "sentiment": result})
output.to_csv("C:/myProj/Result.csv", index=False, quoting=3)

