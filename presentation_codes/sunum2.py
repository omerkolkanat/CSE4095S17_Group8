import csv
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
from plotly.figure_factory._annotated_heatmap import np

inpTweets = csv.reader(open('data/newData.csv', 'rb'), delimiter=';')

mylist = []
positiveList = []
positive_count = 0
positive_terms = []
deneme = " "

for row in inpTweets:
    tweet = row[0].decode('ISO-8859-9')  # tweet from csv file
    sentiment = row[1].decode('ISO-8859-9')  # sentiment of tweet from csv file
    # print tweet,sentiment

    mylist.append((tweet,sentiment))

for positive in mylist:
    if(positive[1]=="Negatif"):
        out = "".join(c for c in positive[0] if c not in ( '.',',',';'))
        positiveList.append(out)
        positive_count += 1


for term in positiveList:
    deneme = deneme + " ".join(term.split())
    deneme = deneme +" "

positive_terms = deneme.split()
#print (Counter(positive_terms))
print (Counter(positive_terms).most_common(10))

labels, values = zip(*Counter(positive_terms).most_common(10))

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()