import pandas as pd
from torchtext.data import LabelField, Field, Dataset, Example


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, fields, filter_pred=None):

        examples = []
        for idx, row in dataframe.iterrows():
            text = row['text']
            label = row["label"]
            examples.append(Example.fromlist([text, label], fields))
        super().__init__(examples, fields)

  

