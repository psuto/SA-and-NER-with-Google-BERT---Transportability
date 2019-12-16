from unittest import TestCase
from data4SAandBERT import *


class Test(TestCase):

    def test_read_and_clean_rtdata(self):
        trainPath = """/home/peter/dev/Work/Transportation/Data/RT_Sentiment/train_short.tsv"""
        train, test = readAndCleanDataRT(trainPath)
        print(f'Train records {train.__len__()}')
        print(f'Test records {test.__len__()}')
        assert train.__len__() > 0
        assert test.__len__() > 0

    def test_readRTData(self):
        trainPath = """/home/peter/dev/Work/Transportation/Data"""
        train, test = readRTData(trainPath, '', '', "_short", True)
        print(f'Train data {train.__len__()}')
        print(f'Test data {test.__len__()}')
        print('****************************')

    def test_read_twitter_data(self):
        trainPath = """/home/peter/dev/Work/Transportation/Data"""
        train, test = readTwitterData(trainPath, '', '', "_short", True)
        print(f'Train data {train.__len__()}')
        print(f'Test data {test.__len__()}')
        print('****************************')


    def test_read_finance_messages_data(self):
        trainPath = """/home/peter/dev/Work/Transportation/Data"""
        train, test = readFinanceMessagesData(trainPath, '', '', "_short", True)
        print(f'Train data {train.__len__()}')
        print(f'Test data {test.__len__()}')
        print('****************************')

    def test_read_finance_headlines_data(self):
        trainPath = """/home/peter/dev/Work/Transportation/Data"""
        train, test = readFinanceHeadlinesData(trainPath, '', '', "_short", True)
        print(f'Train data {train.__len__()}')
        print(f'Test data {test.__len__()}')
        print('****************************')

readFinanceHeadlinesData