#!/usr/bin/env python
# coding: utf-8

# In[1]:
import requests
import time

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import glob
import random
import seaborn as sns

from IPython.display import clear_output

# http://www.nltk.org/howto/wordnet.html

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import sklearn
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

from scipy import stats
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr

from nltk.wsd import lesk

# ## Load Data

# In[ ]:

#%% Read IMDB data
test_dir = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/IMDB Reviews/IMDB Data/test/'
# test_dir = 'Data/sentiment/Data/IMDB Reviews/IMDB Data/test'
# train_dir = 'C:\\Users\\Phil\\Sync\\entity-recognition-datasets-master\\sentiment\\Data\\IMDB Reviews\\IMDB Data\\train'
train_dir = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/IMDB Reviews/IMDB Data/train/'


# train_dir = 'Data/sentiment/Data/IMDB Reviews/IMDB Data/train'

# Alternatively written as a function for importing from different directory sources
def IMDB_to_csv(directory):
    data = pd.DataFrame()
    linux_appendix = '/neg/*.txt'
    windows_appendix = '\\neg\\*.txt'
    for filename in glob.glob(str(directory) + linux_appendix):
        with open(filename, 'r', encoding="utf8") as f:
            content = f.readlines()
            content_table = pd.DataFrame(
                {'id': filename.split('_')[0].split('\\')[-1], 'rating': filename.split('_')[1].split('.')[0],
                 'pol': 'neg', 'text': content})
        data = data.append(content_table)

    for filename in glob.glob(str(directory) + linux_appendix):
        with open(filename, 'r', encoding="utf8") as f:
            content = f.readlines()
            content_table = pd.DataFrame(
                {'id': filename.split('_')[0].split('\\')[-1], 'rating': filename.split('_')[1].split('.')[0],
                 'pol': 'pos', 'text': content})
        data = data.append(content_table)
    data = data.sort_values(['pol', 'id'])
    data = data.reset_index(drop=True)
    # data['rating_norm'] = (data['rating'] - data['rating'].min())/( data['rating'].max() - data['rating'].min() )
    return (data)


# In[ ]:


IMDB_train = IMDB_to_csv(train_dir)
IMDB_train['pol_id'] = np.where(IMDB_train['pol'] == 'neg', -1,
                                np.where(IMDB_train['pol'] == 'pos', 1, 0))

# In[ ]:


IMDB_train.head()

# In[ ]:


IMDB_test = IMDB_to_csv(test_dir)
IMDB_test['pol_id'] = np.where(IMDB_test['pol'] == 'neg', -1,
                               np.where(IMDB_test['pol'] == 'pos', 1, 0))

# In[ ]:


print(IMDB_test.head())

# In[ ]:

print('RT data - reading')
# rt_train_path = 'C:\\Users\\Phil\\Sync\\entity-recognition-datasets-master\\sentiment\\Data\\RT_Sentiment\\train.tsv'
rt_train_path = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/RT_Sentiment/train.tsv'
rt_train_data = pd.read_csv(rt_train_path, header=0, delimiter="\t", quoting=3)
rt_train_data['pol'] = np.where(rt_train_data['Sentiment'] == 3, "neut", np.where(rt_train_data['Sentiment'] < 3, "neg",
                                                                                  np.where(
                                                                                      rt_train_data['Sentiment'] > 3,
                                                                                      "pos", "")))
# Remove any neutral classified phrases
rt_train_data = rt_train_data[rt_train_data['pol'] != "neut"]
rt_train_data = rt_train_data[rt_train_data['pol'] != ""]

rt_train_data = rt_train_data.reset_index(drop=True)
# rt_test_path = 'E:\\Documents\\Text Data\\rottentomatoes\\test.tsv'
# rt_test_data = pd.read_csv(rt_test_path,header=0,delimiter="\t",quoting=3)


# In[ ]:

count_list = pd.DataFrame()
for i in range(0, len(rt_train_data)):
    count = pd.DataFrame({'count': len(rt_train_data['Phrase'][i].split())}, index=[i])
    count_list = count_list.append(count)
count_list = count_list.reset_index(drop=True)
rt_train_data['word_count'] = count_list['count']

# In[ ]:

print('RT2 data - processing')
rt_train_data_2 = pd.DataFrame()
for i in range(1, max(rt_train_data['SentenceId'])):
    # Some sentence ids are not used and so we skip these
    if (len(rt_train_data[rt_train_data['SentenceId'] == i]) == 0):
        continue
    else:
        rt_train_data_2 = rt_train_data_2.append(
            rt_train_data[rt_train_data['SentenceId'] == i].sort_values('word_count', ascending=False).reset_index(
                drop=True).iloc[0, :])
rt_train_data_2 = rt_train_data_2.reset_index(drop=True)
# rename column from phrase to text for clarity
rt_train_data_2.columns = ['text', 'textId', 'SentenceId', 'Sentiment', 'pol', 'word_count']

# In[ ]:


rt_train_data_2['pol_id'] = np.where(rt_train_data_2['pol'] == 'neg', -1,
                                     np.where(rt_train_data_2['pol'] == 'pos', 1, 0))
rt_train_data_2.head()

# In[ ]:

print('Twitter Train data - reading')
twitter_train_path = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/twitter/train.csv'
twitter_train = pd.read_csv(twitter_train_path, encoding="ISO-8859-1")
twitter_train.columns = ['ItemID', 'pol', 'text']
twitter_train['pol_id'] = np.where(twitter_train['pol'] == 'neg', -1,
                                   np.where(twitter_train['pol'] == 'pos', 1, 0))
twitter_train.head()

# In[ ]:
print('Twitter test data - reading')
twitter_products_path = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/twitter/judge-1377884607_tweet_product_company.csv'
twitter_products = pd.read_csv(twitter_products_path, encoding="ISO-8859-1")
twitter_products.columns = ['text', 'product', 'pol']
twitter_products['pol_id'] = np.where(twitter_products['pol'] == 'Negative emotion', -1,
                                      np.where(twitter_products['pol'] == 'Positive emotion', 1, 0))
twitter_products.head()

# In[2]:


# finance_messages_path = 'C:\\Users\\Phil\\Sync\\entity-recognition-datasets-master\\sentiment\\Data\\finance\\EnglishGS.csv'
finance_messages_path = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/finance/EnglishGS.csv'
finance_messages = pd.read_csv(finance_messages_path)
finance_messages.columns = ['unique_id', 'text', 'pol', 'type', 'id']

finance_messages['pol_id'] = np.where(finance_messages['pol'] < 0, -1,
                                      np.where(finance_messages['pol'] > 0, 1, 0))
finance_messages.head()

# In[ ]:


# finance_headlines_path = 'C:\\Users\\Phil\\Sync\\entity-recognition-datasets-master\\sentiment\\Data\\finance\\SSIX News headlines Gold Standard EN.csv'
finance_headlines_path = '/home/uomadmin_ps/PycharmProjects/SAwBertV00/Data/sentiment/Data/finance/SSIX News headlines Gold Standard EN.csv'
finance_headlines = pd.read_csv(finance_headlines_path)
finance_headlines.columns = ['unique_id', 'company', 'company_fixed', 'text', 'pol', 'num_scores']

finance_headlines['pol_id'] = np.where(finance_headlines['pol'] < 0, -1,
                                       np.where(finance_headlines['pol'] > 0, 1, 0))
finance_headlines.head()

# ## Apply TextBlob pre-trained Sentiment Analysis

# In[ ]:


from textblob import TextBlob

# In[ ]:


text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

# blob = TextBlob(text)

# for sentence in blob.sentences:
    # print(sentence.sentiment.polarity)

# In[ ]:


# blob


# In[ ]:


def text_blob(data_column):
    output_labels = pd.DataFrame()
    for n, phrases in enumerate(data_column):
        blob = TextBlob(phrases)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            label = 1
        elif polarity == 0:
            label = 0
        else:
            label = -1

        output_labels = output_labels.append(pd.DataFrame({'label': label}, index=[n]))

    return (output_labels['label'])


# In[ ]:


IMDB_train['text'][0].split('.')

# In[ ]:


for n, phrases in enumerate(IMDB_train['text'][0:1]):
    blob = TextBlob(phrases)
    phrase_sentence_pol = pd.DataFrame()
    print(blob.sentiment.polarity)

# In[ ]:


blob

# In[ ]:


IMDB_train['text_blob'] = text_blob(IMDB_train['text'])
IMDB_train.head()

# In[ ]:


IMDB_train['pol_id'] = np.where(IMDB_train['pol'] == 'neg', -1,
                                np.where(IMDB_train['pol'] == 'pos', 1, 0))
IMDB_train.head()

# In[ ]:


precision_applied = sklearn.metrics.precision_score(IMDB_train['pol_id'],
                                                    IMDB_train['text_blob'], average='weighted')
recall_applied = sklearn.metrics.recall_score(IMDB_train['pol_id'],
                                              IMDB_train['text_blob'], average='weighted')

F1_applied = 2 * (precision_applied * recall_applied) / (precision_applied + recall_applied)

# In[ ]:


F1_applied

# In[ ]:


from textblob.sentiments import NaiveBayesAnalyzer

blob = TextBlob("neutral", analyzer=NaiveBayesAnalyzer())
blob.sentiment.classification


# In[ ]:


def text_blob_NB(data_column):
    output_labels = pd.DataFrame()
    for n, phrases in enumerate(data_column):
        clear_output(wait=True)
        print("Completed:", np.round(n / len(data_column) * 100), "%")
        blob = TextBlob(phrases, analyzer=NaiveBayesAnalyzer())
        polarity = blob.sentiment.classification

        if polarity == 'pos':
            label = 1
        elif polarity == 'neg':
            label = -1
        else:
            label = 0

        output_labels = output_labels.append(pd.DataFrame({'label': label}, index=[n]))

    return (output_labels['label'])


# In[3]:


t0 = time.time()
IMDB_train['text_blob'] = text_blob_NB(IMDB_train['text'][0:10])
t1 = time.time()

total = t1 - t0

IMDB_train['pol_id'] = np.where(IMDB_train['pol'] == 'neg', -1,
                                np.where(IMDB_train['pol'] == 'pos', 1, 0))
precision_applied = sklearn.metrics.precision_score(IMDB_train['pol_id'],
                                                    IMDB_train['text_blob'], average='weighted')
recall_applied = sklearn.metrics.recall_score(IMDB_train['pol_id'],
                                              IMDB_train['text_blob'], average='weighted')

F1_applied = 2 * (precision_applied * recall_applied) / (precision_applied + recall_applied)
F1_applied

# In[ ]:


total


# ### Apply to all datasets
# 

# In[ ]:


def text_blob_2(data_column):
    output_labels = pd.DataFrame()
    for n, phrases in enumerate(data_column):
        blob = TextBlob(phrases)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            label = 1
        elif polarity == 0:
            label = 0
        else:
            label = -1

        output_labels = output_labels.append(pd.DataFrame({'label': label}, index=[n]))
    return (output_labels['label'])


# In[ ]:


"""
    output_labels_NB = pd.DataFrame()
    for n,phrases in enumerate(data_column):
        blob = TextBlob(phrases,analyzer=NaiveBayesAnalyzer())
        polarity = blob.sentiment.classification

        if polarity == 'pos':
            label = 1
        elif polarity == 'neg':
            label = -1
        else:
            label = 0

        output_labels_NB = output_labels_NB.append(pd.DataFrame({'label':label},index=[n]))

    return(output_labels['label'],output_labels_NB['label'])
"""
"""


    precision_applied_NB = sklearn.metrics.precision_score(IMDB_train['pol_id'],
                                                                IMDB_train['text_blob_NB'], average='weighted')
    recall_applied_NB = sklearn.metrics.recall_score(IMDB_train['pol_id'],
                                                                IMDB_train['text_blob_NB'], average='weighted')

    F1_applied_NB = 2 * (precision_applied_NB * recall_applied_NB) / (precision_applied_NB + recall_applied_NB)
    
    
"""

# In[ ]:


datasets = [IMDB_train, IMDB_test, rt_train_data_2, twitter_products, finance_messages, finance_headlines]
dataset_names = ['IMDB_train', 'IMDB_test', 'rt_train_data_2', 'twitter_products', 'financial_messages',
                 'financial_headlines']
dataset_text_col = ['text', 'text', 'text', 'text', 'text', 'text']

output = pd.DataFrame()
for n, dataset in enumerate(datasets):
    print('Current Dataset:', dataset_names[n])
    datasets[n]['text_blob_def'] = text_blob_2(datasets[n][str(dataset_text_col[n])].astype(str))
    # dataset['text_blob_NB'] = text_blob_2(dataset[str(dataset_text_col[n])])[0]

    precision_applied = sklearn.metrics.precision_score(datasets[n]['pol_id'],
                                                        datasets[n]['text_blob_def'], average='weighted')
    recall_applied = sklearn.metrics.recall_score(datasets[n]['pol_id'],
                                                  datasets[n]['text_blob_def'], average='weighted')

    F1_applied = 2 * (precision_applied * recall_applied) / (precision_applied + recall_applied)

    output = output.append(pd.DataFrame({'dataset': dataset_names[n],
                                         'text_blob_def_prec': precision_applied,
                                         'text_blob_def_recall': recall_applied,
                                         'text_blob_def_F1': F1_applied
                                         }, index=[n]))

# In[ ]:


output

# In[ ]:


plt.bar(output['dataset'], output['text_blob_def_F1'])
plt.title("TextBlob Sentiment Analysis Appied to Datasets (F1)")
plt.xticks(rotation='vertical')
plt.ylim([0, 1])
plt.ylabel("F1 Score")
plt.show()


# In[ ]:


def text_blob_2(data_column):
    output_labels = pd.DataFrame()
    for n, phrases in enumerate(data_column):
        blob = TextBlob(phrases)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            label = 1
        elif polarity < 0:
            label = -1
        else:
            label = 0

        output_labels = output_labels.append(pd.DataFrame({'label': label}, index=[n]))

    output_labels_NB = pd.DataFrame()
    for n, phrases in enumerate(data_column):
        clear_output(wait=True)
        print("Completed:", np.round(n / len(data_column), 4) * 100, "%")
        blob = TextBlob(phrases, analyzer=NaiveBayesAnalyzer())
        polarity = blob.sentiment.classification

        if polarity == 'pos':
            label = 1
        elif polarity == 'neg':
            label = -1
        else:
            label = 0

        output_labels_NB = output_labels_NB.append(pd.DataFrame({'label': label}, index=[n]))

    return (output_labels['label'], output_labels_NB['label'])


# In[ ]:


datasets = [IMDB_train, IMDB_test, rt_train_data_2, twitter_train]
dataset_names = ['IMDB_train', 'IMDB_test', 'rt_train_data_2', 'twitter_train']
dataset_text_col = ['text', 'text', 'text', 'text']

output = pd.DataFrame()
for n, dataset in enumerate(datasets[1:2]):
    print('Current Dataset:', dataset_names[n])
    prediction = text_blob_2(datasets[n][str(dataset_text_col[n])])
    datasets[n]['text_blob_def'] = prediction[0]
    dataset['text_blob_NB'] = prediction[1]

    precision_applied = sklearn.metrics.precision_score(datasets[n]['pol_id'],
                                                        datasets[n]['text_blob_def'], average='weighted')
    recall_applied = sklearn.metrics.recall_score(datasets[n]['pol_id'],
                                                  datasets[n]['text_blob_def'], average='weighted')

    F1_applied = 2 * (precision_applied * recall_applied) / (precision_applied + recall_applied)

    precision_applied_NB = sklearn.metrics.precision_score(datasets[n]['pol_id'],
                                                           datasets[n]['text_blob_NB'], average='weighted')
    recall_applied_NB = sklearn.metrics.recall_score(datasets[n]['pol_id'],
                                                     datasets[n]['text_blob_NB'], average='weighted')

    F1_applied_NB = 2 * (precision_applied_NB * recall_applied_NB) / (precision_applied_NB + recall_applied_NB)

    output = output.append(pd.DataFrame({'dataset': dataset_names[n],
                                         'text_blob_def_prec': precision_applied,
                                         'text_blob_def_recall': recall_applied,
                                         'text_blob_def_F1': F1_applied,
                                         'text_blob_NB_prec': precision_applied_NB,
                                         'text_blob_NB_recall': recall_applied_NB,
                                         'text_blob_NB_F1': F1_applied_NB
                                         }, index=[n]))

# In[ ]:


output

# In[ ]:


n

# In[ ]:


dataset.head()


# In[ ]:

def main():
    pass


if __name__ == '__main__':
    main()
