from baseline_koelectra import *

dh = DataHandler()
dh.jsonlload('data/train.jsonl')

tr = Trainer()
tr.train_sentiment_analysis()

em = EvaluateModel()
em.test_sentiment_analysis()
