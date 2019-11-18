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
        dataRes = pd.read_csv(pOut, delimiter = ',')
    else:
        data = pd.DataFrame()
        negPath = mainDir / "neg" /"*.txt"
        # for filename in glob.glob(str(directory) + '\\neg\\*.txt'):
        print(f'Reading files  {os._fspath(negPath)}')
        dataNeg = extractIMDBdata(data, negPath, sentimentMap)
        # posPath = Path(directory + "/pos/*.txt")
        posPath =  mainDir / "pos" /"*.txt"
        print(f'Reading files  {os._fspath(posPath)}')
        dataPos = extractIMDBdata(data, posPath, sentimentMap)
        # data = data.sort_values(['pol', 'id'])
        # data = data.reset_index(drop=True)
        # # data['rating_norm'] = (data['rating'] - data['rating'].min())/( data['rating'].max() - data['rating'].min() )
        dataRes= pd.concat([dataPos, dataNeg]).sample(frac=1).reset_index(drop=True)
        print(f"Saving file to {os._fspath(pOut)}")
        dataRes.to_csv(pOut,  index=False)
    return(dataRes)




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
def readImdbData(imdb_dir, readFromSource=False):
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
    data = IMDB_to_csv(abs_imdb_dir,readFromSource)
    return (data)


# %%
def main():
    # %%
    ## read IMDB data
    train_dir_Imdb = '../Data/sentiment/Data/IMDB Reviews/IMDB Data/train/'
    # load_directory_data(train_dir_Imdb)
    print(f'Reading {train_dir_Imdb}')
    imdbTrainData = readImdbData(train_dir_Imdb)
    # %%
    test_dir_Imdb = '../Data/sentiment/Data/IMDB Reviews/IMDB Data/test/' #'Data/sentiment/Data/IMDB Reviews/IMDB Data/test/'
    print(f'Reading {test_dir_Imdb}')
    imdbTestData = readImdbData(test_dir_Imdb)
    # %%
    print(f'Finished')
    print(f'---------------')
    # transfrom Imdb data

if __name__ == "__main__":
    main()
