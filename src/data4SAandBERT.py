#!/usr/bin/env python
# coding: utf-8
# %%
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import os
import glob
from pathlib import Path
import re
import numpy as np
from sklearn.model_selection import train_test_split

#
# def load_directory_data(imdb_dir):
#     """
#     Article
#
#     :param imdb_dir:
#     :return:
#     """
#     abs_imdb_dir = os.path.abspath(imdb_dir)
#     if os.path.exists(abs_imdb_dir):
#         print(f'File with imdb training data  {abs_imdb_dir} exists')
#     else:
#         print(f'File with imdb training data  "{abs_imdb_dir}" does not exists')
#     data = {}
#     data["sentence"] = []
#     data["sentiment"] = []
#     for file_path in os.listdir(abs_imdb_dir):
#         # print(file_path)
#         with tf.gfile.GFile(os.path.join(abs_imdb_dir, file_path), "r") as f:
#             data["sentence"].append(f.read())
#             data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
#     return pd.DataFrame.from_dict(data)
#
#
# # Merge positive and negative examples, add a polarity column and shuffle.
# def load_dataset(directory):
#     """
#     Article
#
#     :param directory:
#     :return:
#     """
#     pos_df = load_directory_data(os.path.join(directory, "pos"))
#     neg_df = load_directory_data(os.path.join(directory, "neg"))
#     pos_df["polarity"] = 1
#     neg_df["polarity"] = 0
#     return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def IMDB_to_csv(directory, read_from_source=False):
    """
    Phillips version

    :param directory:
    :return:
    """
    sentimentMap = {
        'neg': 0,
        'pos': 1
    }
    # typeOfData=os.path.basename(directory)
    mainDir = Path(directory)
    pOut = mainDir / "imdb_complete.csv"
    if pOut.exists() and (not read_from_source):
        print(f'* Reading from File {pOut}')
        dataRes = pd.read_csv(pOut, delimiter=',')
        dataRes.dropna(inplace=True)
    else:
        data = pd.DataFrame()
        negPath = mainDir / "neg" / "*.txt"
        # for filename in glob.glob(str(directory) + '\\neg\\*.txt'):
        print(f'Reading files  {os._fspath(negPath)}')
        dataNeg = extractIMDBdata(data, negPath, sentimentMap)
        # posPath = Path(directory + "/pos/*.txt")
        posPath = mainDir / "pos" / "*.txt"
        print(f'Reading files  {os._fspath(posPath)}')
        dataPos = extractIMDBdata(data, posPath, sentimentMap)
        # data = data.sort_values(['pol', 'id'])
        # data = data.reset_index(drop=True)
        # # data['rating_norm'] = (data['rating'] - data['rating'].min())/( data['rating'].max() - data['rating'].min() )
        dataRes = pd.concat([dataPos, dataNeg]).sample(frac=1).reset_index(drop=True)
        print(f"Saving file to {os._fspath(pOut)}")
        dataRes.to_csv(pOut, index=False)
        dataRes.dropna(inplace=True)
    return (dataRes)


def extractIMDBdata(data, inputPath, sentimentMap):
    """
    extracts data properly from file content, file name and directory name
    :param data:
    :param inputPath:
    :param sentimentMap:
    :return:
    """
    for filename in glob.glob(str(inputPath)):
        basename = os.path.basename(filename)
        pathOnly, fileWhole = os.path.split(filename)
        id_ = basename.split('_')[0].split('\\')[-1]
        rating_ = basename.split('_')[1].split('.')[0]
        sentimentStr = os.path.basename(pathOnly)
        sentiment = sentimentMap[sentimentStr]
        with open(filename, 'r', encoding="utf8") as f:
            content = f.readlines()
            content_table = pd.DataFrame(
                {'id': id_, 'rating': rating_,
                 'polarity': sentiment, 'text': content})
        data = data.append(content_table)
    return data


# %%
def readImdbDataOrig(imdb_dir, readFromSource=False):
    """
    PS:

    :param imdb_dir:
    :return:
    """
    abs_imdb_dir = Path(imdb_dir).absolute().resolve()
    # abs_imdb_dir = os.path.abspath(imdb_dir)
    if os.path.exists(abs_imdb_dir):
        print(f'Directory with imdb training data  {abs_imdb_dir} exists')
    else:
        print(f'Directory with imdb training data  "{abs_imdb_dir}" does not exists')
    # print(f'Training data in {abs_imdb_dir}')
    data = IMDB_to_csv(abs_imdb_dir, readFromSource)
    return (data)


# %% MAIN
def readIMDBData(inputDir, info, dbName, dataVersionAppendix, readFromSource=True):
    mainDBDir = info['dir']
    mainDataPath = str(Path(inputDir) / mainDBDir)
    train_dir_Imdb = Path(mainDataPath) / ('train' + dataVersionAppendix)
    test_dir_Imdb = Path(mainDataPath) / ('test' + dataVersionAppendix)
    trainData = readImdbDataOrig(train_dir_Imdb, readFromSource)
    testData = readImdbDataOrig(test_dir_Imdb, readFromSource)
    return trainData, testData


# %%
def readImdbData(dataDir, DATA_VERSION_APPENDIX, readFromSource=False):
    """
    PS:

    :param imdb_dir:
    :return:
    """

    abs_imdb_parent_dir = Path(dataDir).absolute().resolve()
    # abs_imdb_dir = os.path.abspath(imdb_dir)
    abs_imdb_dir = Path(abs_imdb_parent_dir) / ('train' + DATA_VERSION_APPENDIX)
    if os.path.exists(abs_imdb_dir):
        print(f'Directory with imdb training data  {abs_imdb_dir} exists')
    else:
        print(f'Directory with imdb training data  "{abs_imdb_dir}" does not exists')
    # print(f'Training data in {abs_imdb_dir}')
    data = IMDB_to_csv(abs_imdb_dir, readFromSource)
    return (data)

# %%
def readAndCleanFinanceMessagesData(filePath):
    financeMessagesData = pd.read_csv(filePath, encoding="ISO-8859-1")
    sentimentMap = {
        'neg': 0,
        'pos': 1
    }

    financeMessagesData.dropna(inplace=True)
    print(f'{financeMessagesData.columns.to_list}')
    financeMessagesData.rename(columns={'Unique':'ItemID','Text': 'text'}, inplace=True)
    scoreColName = 'Average Score'
    financeMessagesData['polarity'] = np.where(financeMessagesData[scoreColName] < 0 , 'neg',
                                     np.where(financeMessagesData[scoreColName] == 0, 'neut',
                                              np.where(financeMessagesData[scoreColName] > 0,
                                                       'pos', "")))

    financeMessagesData = financeMessagesData[financeMessagesData['polarity'] != 'neut']
    financeMessagesData['polarity'].replace(sentimentMap, inplace=True)  # ==sentStr] = sentimentMap[sentStr]
    train, test = train_test_split(financeMessagesData, test_size=0.33, random_state=42)
    return train, test



def readFinanceMessagesData(data_dir, info, databaseName2Info, dataVersionAppendix, readFromSource):
    # / home / peter / dev / Work / Transportation / Data / finance / EnglishGS.csv
    abs_imdb_dir = Path(data_dir) / "finance"
    abs_imdb_dir = abs_imdb_dir.absolute().resolve()
    trainTestPath = abs_imdb_dir / ("EnglishGS" + dataVersionAppendix + ".csv")
    train, test = readAndCleanFinanceMessagesData(trainTestPath)
    return train, test


def readAndCleanFinanceHeadlineData(filePath):
    financeHeadlineData = pd.read_csv(filePath, encoding="ISO-8859-1")
    sentimentMap = {
        'neg': 0,
        'pos': 1
    }

    financeHeadlineData.dropna(inplace=True)
    print(f'{financeHeadlineData.columns.to_list}')
    # ['id', 'Company Name (Original)', 'Company Name (Fixed)', 'Text',
    #        'sentiment score', '# Scores']
    financeHeadlineData.rename(columns={'id': 'ItemID', 'Text': 'text'}, inplace=True)
    scoreColName = 'sentiment score'
    financeHeadlineData['polarity'] = np.where(financeHeadlineData[scoreColName] < 0, 'neg',
                                               np.where(financeHeadlineData[scoreColName] == 0, 'neut',
                                                        np.where(financeHeadlineData[scoreColName] > 0,
                                                                 'pos', "")))

    financeHeadlineData = financeHeadlineData[financeHeadlineData['polarity'] != 'neut']
    financeHeadlineData['polarity'].replace(sentimentMap, inplace=True)  # ==sentStr] = sentimentMap[sentStr]
    train, test = train_test_split(financeHeadlineData, test_size=0.33, random_state=42)
    return train, test


def readFinanceHeadlinesData(data_dir, info, databaseName2Info, dataVersionAppendix, readFromSource):
    # / home / peter / dev / Work / Transportation / Data / finance / EnglishGS.csv
    abs_imdb_dir = Path(data_dir) / "finance"
    abs_imdb_dir = abs_imdb_dir.absolute().resolve()
    trainTestPath = abs_imdb_dir / ("SSIX News headlines Gold Standard EN" + dataVersionAppendix + ".csv")
    # finance_headlines
    train, test = readAndCleanFinanceHeadlineData(trainTestPath)
    return train, test


# %%
def readTwitterData(data_dir, info, databaseName2Info, dataVersionAppendix, readFromSource):
    # /home/peter/dev/Work/Transportation/Data/twitter/
    abs_imdb_dir = Path(data_dir) / "twitter"
    abs_imdb_dir = abs_imdb_dir.absolute().resolve()
    # trainPath = abs_imdb_dir / ("train" + dataVersionAppendix + ".tsv")
    # # Test data are without labels
    # testPath = abs_imdb_dir / ("test" + dataVersionAppendix + ".tsv")
    trainTestPath = abs_imdb_dir / ("train" + dataVersionAppendix + ".csv")
    # testPath = abs_imdb_dir / ("test" + dataVersionAppendix + ".csv")
    train, test = readAndCleanTwitterData(trainTestPath)
    return train, test


# %%
def readRTData(data_dir, info, databaseName2Info, dataVersionAppendix, readFromSource):
    """
    PS:
    :param data_dir:
    :return:
    """
    abs_imdb_dir = Path(data_dir) / "RT_Sentiment"
    abs_imdb_dir = abs_imdb_dir.absolute().resolve()
    trainPath = abs_imdb_dir / ("train" + dataVersionAppendix + ".tsv")
    # Test data are without labels
    testPath = abs_imdb_dir / ("test" + dataVersionAppendix + ".tsv")

    # abs_imdb_dir = str(abs_imdb_dir)
    if os.path.exists(abs_imdb_dir):
        print(f'Directory with imdb training data  {abs_imdb_dir} exists')
    else:
        print(f'Directory with imdb training data  "{abs_imdb_dir}" does not exists')
    # print(f'Training data in {abs_imdb_dir}')
    train, test = readAndCleanDataRT(trainPath)
    # testData = readAndCleanRTData(testPath)
    return train, test


def readAndCleanTwitterData(filePath):
    twitter_train = pd.read_csv(filePath, encoding="ISO-8859-1")
    twitter_train.dropna(inplace=True)
    # ['ItemID', 'Sentiment', 'SentimentText']
    twitter_train.columns = ['ItemID', 'polarity', 'text']
    # print(f"{twitter_train['polarity'].min()}")
    # print(f"{twitter_train['polarity'].max()}")
    train, test = train_test_split(twitter_train, test_size = 0.33, random_state = 42)
    return train, test


def readAndCleanDataRT(trainPath):
    sentimentMap = {
        'neg': 0,
        'pos': 1
    }
    trainData = pd.read_csv(trainPath, header=0, delimiter="\t", quoting=3)
    trainData.dropna(inplace=True)
    trainData['polarity'] = np.where(trainData['Sentiment'] < 3, 'neg',
                                     np.where(trainData['Sentiment'] == 3, 'neut',
                                              np.where(trainData['Sentiment'] > 3,
                                                       'pos',"")))
    trainData = trainData[trainData['polarity']!='neut']
    trainData['polarity'].replace(sentimentMap,inplace=True)  #==sentStr] = sentimentMap[sentStr]
    trainData.rename(columns={'Phrase':'text'},inplace=True)
    # trainData['polarity'].astype(int)
    train, test = train_test_split(trainData, test_size = 0.33, random_state = 42)
    return train, test


# %%
def main():
    # %%
    ## read IMDB data
    train_dir_Imdb = '../Data/sentiment/Data/IMDB Reviews/IMDB Data/train/'
    # load_directory_data(train_dir_Imdb)
    print(f'Reading {train_dir_Imdb}')
    imdbTrainData = readImdbData(train_dir_Imdb)
    # %%
    test_dir_Imdb = '../Data/sentiment/Data/IMDB Reviews/IMDB Data/test/'  # 'Data/sentiment/Data/IMDB Reviews/IMDB Data/test/'0
    print(f'Reading {test_dir_Imdb}')
    imdbTestData = readImdbData(test_dir_Imdb)
    # %%
    print(f'Finished')
    print(f'---------------')
    # transfrom Imdb data


if __name__ == "__main__":
    main()
