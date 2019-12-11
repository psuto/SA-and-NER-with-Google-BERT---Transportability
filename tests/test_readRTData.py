import data4SAandBERT
import pytest


# %%
def test_readRT():
    """

    """
    rtDir = "/home/peter/dev/Work/Transportation/Data/RT_Sentiment/train.tsv"
    print(f'file: {rtDir}')
    readFromSource = False
    data = data4SAandBERT.readRTData(rtDir, readFromSource)
    print("")
    assert len(data.columns.to_list())>0



