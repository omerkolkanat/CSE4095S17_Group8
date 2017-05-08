# -*- coding: utf-8 -*-
import csv
import string
import re
import matplotlib.pyplot as plt
#import numpy as np
import plotly.plotly as py
from collections import Counter

inpTweets = csv.reader(open('data/newData.csv', 'rb'), delimiter=';')

tweets = []
positive_count = 0
negative_count = 0
mylist = []
positiveList = []
negativeList = []
positive_terms = []

for row in inpTweets:
    tweet = row[0].decode('ISO-8859-9')  # tweet from csv file
    sentiment = row[1].decode('ISO-8859-9')  # sentiment of tweet from csv file
    mylist.append((tweet,sentiment))
    #print (tweet + "------>" +sentiment +"\n")


for positive in mylist:
    if(positive[1]=="Pozitif"):
        #print (positive[0]+"------->"+positive[1]+"\n")
        positiveList.append((positive[0],positive[1]))
        positive_count += 1
        positive_terms.append((positive[0].split()))

#print (positive_count)
for negative in mylist:
    if(negative[1]=="Negatif"):
        #print (negative[0]+"------->"+negative[1]+"\n")
        negativeList.append((negative[0],negative[1]))
        negative_count += 1

#print (negative_count)


def show_pos_neg():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Positive :' + str(positive_count), 'Negative :' + str(negative_count)
    sizes = [positive_count, negative_count]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

#print Counter(list)
show_pos_neg()


