import csv
import io
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
from plotly.figure_factory._annotated_heatmap import np


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

    # remove ' from tweets
    # tweet = re.findall(r"^[^*$<,>?!']*$", tweet)
    # tweet = ' '.join(tweet)

    # split tweet into words
    words = tweet.split()
    for w in words:

        # replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        if w in stopWords:
            continue
        else:
            # print w
            featureVector.append(w.lower())
    return featureVector

inpTweets = csv.reader(open('data/newData.csv', 'rb'), delimiter=';')
stopWords = getStopWordList('data/stopWords.txt')

positive_terms_list = []
negative_terms_list = []
temp =" "
list_display = []

for row in inpTweets:
    tweet = row[0].decode('ISO-8859-9')  # tweet from csv file
    sentiment = row[1].decode('ISO-8859-9')  # sentiment of tweet from csv file
    if(row[1] == "Negatif"):
        processedTweet_negatif = processTweet(tweet)
        featureVector_negatif = getFeatureVector(processedTweet_negatif, stopWords)
        negative_terms_list.append(featureVector_negatif);
        #print(featureVector_negatif)
    if(row[1] == "Pozitif"):
        processedTweet_positive = processTweet(tweet)
        featureVector_positive = getFeatureVector(processedTweet_positive, stopWords)
        positive_terms_list.append(featureVector_positive);
        #print(featureVector_positive)


for term in negative_terms_list:
    temp = temp + " ".join(term)
    temp = temp +" "



list_display = temp.split()
#print (Counter(list_display))
labels, values = zip(*Counter(list_display).most_common(10))

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()