# -*- coding: utf-8 -*-
import csv
import re
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def processTweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # tweet = re.findall(r'[^,;\s]+',tweet)

    # try to split hashtag words
    # tweet = re.findall('#[A-Z][^A-Z]*',tweet)
    # tweet = ' '.join(tweet)

    tweet = re.sub('@[^/]+', ' ', tweet)

    # Remove numbers from string
    # tweet = re.sub(r'-?[0-9]', '', tweet)

    # trim
    tweet = tweet.strip('\'"')

    # drop text begin with '(@'
    tweet = re.sub('[(]', '', tweet)

    return tweet


def replaceTwoOrMore(s):
    # detect 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1+", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def getStopWordList(stopWordListFileName):
    # read the stopwords
    stopWords = ['AT_USER', 'URL']

    fp = io.open(stopWordListFileName, 'r', encoding='utf-8')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords


def getFeatureVector(tweet, stopWords):
    featureVector = []

    # remove , . from the tweets
    tweet = re.findall(r'[^,;\s]+', tweet)
    tweet = ' '.join(tweet)
    tweet = re.findall(r'[^.;\s]+', tweet)
    tweet = ' '.join(tweet)

    # split tweet into words
    words = tweet.split()
    for w in words:

        # replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if the word stats with an turkish alphabet
        # val = re.search(r"^[a-zA-Z0-9ğüşöçİĞÜŞÖÇ]+$", w)

        # ignore if it is a stop word
        if w in stopWords:
            continue
        else:
            # print w
            featureVector.append(w.lower())
    return featureVector


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


# Read the tweets one by one and process it
inpTweets = csv.reader(open('data/newData.csv', 'rb'), delimiter=';')
stopWords = getStopWordList('data/stopWords.txt')

featureList = []
tweets = []
tempList = []
count = 0

for row in inpTweets:
    tweet = row[0].decode('ISO-8859-9')  # tweet from csv file

    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)

    featureList.extend(featureVector)
    testTweet = ' '.join(featureVector)
    tweets.append(testTweet)

vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(tweets)



true_k = 2 # Number of clusters
model = KMeans(n_clusters=true_k,init='k-means++',max_iter=100, n_init=1)
model.fit(X)
indexx=0
print("Top terms per cluster : ")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" %i,
    for ind in order_centroids[i, :10]: # number of words to print
        print ' %s' % terms[ind]