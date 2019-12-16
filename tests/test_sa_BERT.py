from unittest import TestCase

from sa_BERT import *

class Test(TestCase):
    def test_write_performance_measures(self):
        data = {'Name': ['Tom', 'nick', 'krish', 'jack'], 'Age': [20, 21, 19, 18]}
        df = pd.DataFrame(data)
        inputDir = "/home/peter/dev/Work/Transportation/Data/"
        writePerformanceMeasures(df,inputDir,'2019-12-13__15-06-42','short')
        assert(1==1)
