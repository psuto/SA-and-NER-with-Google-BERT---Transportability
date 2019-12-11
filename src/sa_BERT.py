#!/usr/bin/env python
# coding: utf-8

# %% Predicting Movie Review Sentiment with BERT on TF Hub

# In[1]:

# %% Import Libraries
import IPython as ip
from IPython import get_ipython
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import data4SAandBERT

import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path, PurePath
import os
import time, argparse
import cProfile
import timeit
from tensorflow import keras
import argparse
import json
import numpy as np
import pandas as pd
import logging

# import pprint

databaseName2Info = {'imdb': {'dir': "IMDB Reviews",
                              'fnc': data4SAandBERT.readIMDBData,
                              'READ_FROM_SOURCE': True,
                              'dataInfo': {
                                  'self.LABEL_COLUMN': 'pol',
                                  'DATA_COLUMN': 'text',
                                  'CASE_LABEL_LIST': [0, 1]}
                              },
                     'rt': {'dir': "RT_Sentiment",
                            'fnc': data4SAandBERT.readRTData},
                     'READ_FROM_SOURCE': True,
                     'dataInfo': {
                         'self.LABEL_COLUMN': '',
                         'DATA_COLUMN': '',
                         'CASE_LABEL_LIST': ''
                     }

                     }

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


# %%
class ProcessingInfo():
    def __init__(self):
        self.time = time
        self.timeStr = time.strftime("%Y-%m-%d__%H-%M-%S")
        self.dateTimeformatingStr = "%Y-%m-%d__%H-%M-%S"
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 2e-5
        self.NUM_TRAIN_EPOCHS = 3.0
        # Warmup is a period of time where hte learning rate
        # is small and gradually increases--usually helps training.
        self.WARMUP_PROPORTION = 0.1
        # Model configs
        self.SAVE_CHECKPOINTS_STEPS = 500
        self.SAVE_SUMMARY_STEPS = 100
        self.MAX_SEQ_LENGTH = 128


# %%
parser = argparse.ArgumentParser(description="Simple")
parser.add_argument("--inDirIMDB", action="store", dest="inputDirectoryIMDBData", type=str,
                    default="Directory path needs to be specified",
                    help="file path to file with simulation output")

parser.add_argument("--readDS", action="store", dest="readDataSource", type=bool, default=True,
                    help="whether to read original source of data or consolidated table in csv file. Values:True, False.")

params = parser.parse_args()  # ['--fSimul="oooooooooooooooooo"'],'--fWF="xxxxxxxxxxxxxxxxxxxx"'
print(params)

# Create a custom logger
# %% Logger Setting
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

console_log_level = logging.INFO  # default WARNING
file_log_level = logging.INFO  # default ERROR

logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_handler.setLevel(console_log_level)
f_handler.setLevel(file_log_level)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -% (message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


# %%
class DataVersionAppendix:
    def __init__(self):
        self.normalVersion = ""
        self.shortVersion = "_short"


# %% Get examples

def getExamples(df, dataInfo, guid=None, text_b=None):
    cols = df.columns.to_list()
    if any(x for x in cols if 'pol' == x):
        labelColName = 'pol'
    else:
        labelColName = 'polarity'
    print(f'cols: {cols}')

    def processLine(l):
        text_a = l[dataInfo['dataInfo']['DATA_COLUMN']]
        labelColumn = l[labelColName]
        res = bert.run_classifier.InputExample(guid=None,
                                               # Globally unique ID for bookkeeping, unused in this example
                                               text_a=text_a,
                                               # =x[DATA_COLUMN],
                                               text_b=None,
                                               label=labelColumn
                                               )
        return (res)

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    inputExamples = df.apply(lambda x:
                             processLine(x),
                             axis=1)
    return (inputExamples)


# %% MAIN
def readData(inputDir, databaseName2Info, dbName, dataVersionAppendix, readFromSource=True):
    info = databaseName2Info[dbName]
    readDBFnc = info['fnc']
    trainDB, testDB = readDBFnc(inputDir, info, databaseName2Info, dataVersionAppendix, readFromSource)
    return trainDB, testDB


# %%
def create_tokenizer_from_hub_module(dataInfo):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


# %%
def getOutputDir(DATA_LOCATION_RELATIVE_TO_CODE, startDate, infoTrain):
    outDir1 = Path(DATA_LOCATION_RELATIVE_TO_CODE).absolute() / 'Output' / (
            infoTrain['dir'] + '_Output_' + startDate.strftime("%Y-%m-%d__%H-%M-%S"))
    outDir = outDir1.resolve()
    outDir.mkdir(parents=True, exist_ok=True)
    print(f"Output path {str(outDir)}")
    return str(outDir)


# %%
def getTFEstimatorParameters(OUTPUT_DIR, SAVE_SUMMARY_STEPS, SAVE_CHECKPOINTS_STEPS):
    """

    :param OUTPUT_DIR:
    :param SAVE_SUMMARY_STEPS:
    :param SAVE_CHECKPOINTS_STEPS:
    :return:
    """
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    return run_config


# %%
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels, BERT_MODEL_HUB):
    """Creates a classification model."""

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


# %%

def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps, BERT_MODEL_HUB):
    """Returns `model_fn` closure for TPUEstimator.
    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.

    :param num_labels:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :return:
    """

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels, BERT_MODEL_HUB)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            # Calculate evaluation metrics.

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


# %%
class NumberOfStepsEstimator():
    def __init__(self, numTrainingFeatures, batchSize, numTrainingEpochs, warmupProportion, ):
        # Compute # train and warmup steps from batch size
        self.num_train_steps = int(
            numTrainingFeatures / batchSize * numTrainingEpochs)  # int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
        self.num_warmup_steps = int(self.num_train_steps * warmupProportion)
        print(f'num_train_steps = {self.num_train_steps}')
        print(f'num train features = {numTrainingFeatures}')

    def getNumOfTrainSteps(self):
        return (self.num_train_steps)

    def getNumOfWarmUpSteps(self):
        return self.num_warmup_steps


# %%
def createFeatures(inputExamples, tokenizer, labelList, processingInfo):
    features = bert.run_classifier.convert_examples_to_features(inputExamples, labelList,
                                                                processingInfo.MAX_SEQ_LENGTH, tokenizer
                                                                )
    return (features)


# %%
def trainModel(currentTrainDataName, currentTrainData, infoTrain, OUTPUT_DIR):
    PROCESSING_INFO = ProcessingInfo()
    DATA_INFO = infoTrain['dataInfo']
    trainInputExamples = getExamples(currentTrainData, infoTrain)
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)
    train_features = createFeatures(trainInputExamples, tokenizer, DATA_INFO['CASE_LABEL_LIST'], PROCESSING_INFO)
    trEstimatorParameters = getTFEstimatorParameters(OUTPUT_DIR, PROCESSING_INFO.SAVE_SUMMARY_STEPS,
                                                     PROCESSING_INFO.SAVE_CHECKPOINTS_STEPS)

    # %% Estimating number of steps (heuristics?)
    # Todo: 191119 Check huristics in NumberOfStepsEstimator
    numberOfStepsEstimator = NumberOfStepsEstimator(len(train_features), PROCESSING_INFO.BATCH_SIZE,
                                                    PROCESSING_INFO.NUM_TRAIN_EPOCHS, PROCESSING_INFO.WARMUP_PROPORTION)
    # %% Estimator setting
    model_fn = model_fn_builder(
        num_labels=len(DATA_INFO['CASE_LABEL_LIST']),
        learning_rate=PROCESSING_INFO.LEARNING_RATE,
        num_train_steps=numberOfStepsEstimator.getNumOfTrainSteps(),  # num_train_steps,
        num_warmup_steps=numberOfStepsEstimator.getNumOfWarmUpSteps(), BERT_MODEL_HUB=BERT_MODEL_HUB)

    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=PROCESSING_INFO.SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=PROCESSING_INFO.SAVE_CHECKPOINTS_STEPS)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": PROCESSING_INFO.BATCH_SIZE})

    # %% Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=PROCESSING_INFO.MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)
    # %% Train estimator
    print(f'Beginning Training!')
    current_time = time.time()
    # timeit.timeit(estimator.train(input_fn=train_input_fn, max_steps=numberOfStepsEstimator.getNumOfTrainSteps()),
    estimator.train(input_fn=train_input_fn, max_steps=numberOfStepsEstimator.getNumOfTrainSteps())
    print("Training took time ", time.time() - current_time)
    print('')
    return estimator,tokenizer

#%%
def performancEvaluation(estimator,currentTestData, infoTrain, OUTPUT_DIR,tokenizer):
    PROCESSING_INFO = ProcessingInfo()
    DATA_INFO = infoTrain['dataInfo']
    testInputExamples = getExamples(currentTestData, infoTrain)
    test_features = createFeatures(testInputExamples, tokenizer, DATA_INFO['CASE_LABEL_LIST'], PROCESSING_INFO)

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=PROCESSING_INFO.MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)
    print(f'Evaluating trained estimator')
    current_time = time.time()
    evaluationResults = estimator.evaluate(input_fn=test_input_fn, steps=None)
    print("Evaluation took time ", time.time() - current_time)
    return evaluationResults

#%%
def main():
    processingInfo = ProcessingInfo()
    importedTrainData = dict()
    importedTestData = dict()
    trainedModels = dict()
    evaluationResults = dict()
    data2Use2Train = ['imdb']  # 'imdb', 'rt'
    data2Use2Test = ['imdb']
    inputInfo2Save = {}
    dataModelsResults = dict()
    # %% read command line parameters
    readFromDataSource = params.readDataSource
    inputDir = params.inputDirectoryIMDBData

    dataVersionAppendix = DataVersionAppendix()
    DATA_VERSION_APPENDIX = dataVersionAppendix.shortVersion  # dataVersionAppendix.shortVersion
    READ_FROM_SOURCE = True  # True #False
    print('')
    # vals = databaseName2Info.keys()
    pdPerformance = pd.DataFrame(columns=['auc', 'eval_accuracy', 'f1_score', 'false_negatives', 'false_positives', 'loss', 'precision',
                 'recall', 'true_negatives', 'true_positives', 'global_step', 'train', 'test'])
    for currentTrainDataName in data2Use2Train:
        currentTrainDBInfo = databaseName2Info[currentTrainDataName]
        trainData = None
        if (currentTrainDataName not in importedTrainData):
            trainData, testData = readData(inputDir, databaseName2Info, currentTrainDataName, DATA_VERSION_APPENDIX,
                                           readFromDataSource)
            importedTrainData[currentTrainDataName] = trainData
            importedTestData[currentTrainDataName] = testData
        currentTrainData = importedTrainData[currentTrainDataName]
        infoTrain = databaseName2Info[currentTrainDataName]
        OUTPUT_DIR = getOutputDir(inputDir, processingInfo.time, infoTrain)
        # **********************************************************************************
        trainedEstimator,tokenizer = trainModel(currentTrainDataName, currentTrainData, infoTrain, OUTPUT_DIR)
        # **********************************************************************************
        pdPerformance = pd.DataFrame()
        # pdPerformance = pd.DataFrame(
        #     columns=['train','test','auc', 'eval_accuracy', 'f1_score', 'false_negatives', 'false_positives', 'loss', 'precision',
        #              'recall', 'true_negatives', 'true_positives', 'global_step', 'train', 'test'])
        for currentTestDataName in importedTestData:
            if currentTestDataName not in data2Use2Test:
                # readData(inputDir, databaseName2Info, currentTrainDataName, DATA_VERSION_APPENDIX, readFromDataSource)
                trainData, testData = readData(inputDir, databaseName2Info, currentTrainDataName,
                                               DATA_VERSION_APPENDIX, readFromDataSource)
                importedTrainData[currentTestDataName] = trainData
                importedTestData[currentTestDataName] = testData
                # read appropriate test data
            currentTestData = importedTestData[currentTestDataName]
            # **********************************************************************************
            evaluationResults = performancEvaluation(trainedEstimator,currentTestData, infoTrain, OUTPUT_DIR,tokenizer)
            # **********************************************************************************
            mdic= {}
            key = 'auc'
            mdic[key]= evaluationResults[key]
            mdic['train'] = currentTrainDataName
            mdic['test'] = currentTestDataName

            nsd = pd.Series(mdic)
            pdPerformance.append(nsd, ignore_index=True)

            print('')
            # train and test
            # train model of model does not exist
            # getPerformanceMeasure(t)
            print('')
        print('')
    print('')


# %% MAIN
if __name__ == "__main__":
    main()
