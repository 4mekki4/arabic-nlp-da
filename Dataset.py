import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from farasa.segmenter import FarasaSegmenter
from sklearn.model_selection import train_test_split
from preprocess_arabert import never_split_tokens, preprocess
import numpy as np
import pandas as pd
farasa_segmenter = FarasaSegmenter(interactive=True)
RANDOM_SEED = 42

class ReviewDataset(Dataset):
    def __init__(self, df, pretraine_path='aubmindlab/bert-base-arabert', max_length=128, farasa=False):
        self.df = df
        self.max_length = max_length
        if farasa:
            self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path,
                                                       do_lower_case=False,
                                                       do_basic_tokenize=True,
                                                       never_split=never_split_tokens)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)


    def __getitem__(self, index):
        review = self.df.iloc[index]["Feed"]
        sentiment = self.df.iloc[index]["Sentiment"]
        sentiment_dict = {
            "Positive": 1,
            "Negative": 0,
        }
        label = sentiment_dict[sentiment]

        encoded_input = self.tokenizer(
                review,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            pass

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return data_input["input_ids"], data_input["attention_mask"], data_input["label"]

    def __len__(self):
        return self.df.shape[0]


def loadData(task, batchsize=16, num_worker=2, dosegmentation=True, pretraine_path='aubmindlab/bert-base-arabert', split=1000,ttype='nothing'):
    train = f'data/{task}_train.csv'
    test = f'data/{task}_test.csv'
    REG_train = pd.read_csv(train)
    REG_test = pd.read_csv(test)

    if dosegmentation:
        REG_train['Feed'] = REG_train['Feed'].apply(lambda x: preprocess(x, do_farasa_tokenization=True,
                                                                       farasa=farasa_segmenter,
                                                                       use_farasapy=True))
        REG_test['Feed'] = REG_test['Feed'].apply(lambda x: preprocess(x, do_farasa_tokenization=True,
                                                                     farasa=farasa_segmenter,
                                                                     use_farasapy=True))
    REG_train = ReviewDataset(REG_train, pretraine_path)
    REG_test = ReviewDataset(REG_test, pretraine_path)

    farasa_segmenter.terminate()

    REG_train_loader = DataLoader(dataset=REG_train, batch_size=batchsize, shuffle=True,
                                   num_workers=num_worker)
    REG_test_loader = DataLoader(dataset=REG_test, batch_size=batchsize, shuffle=False,
                                  num_workers=num_worker)
    return REG_train_loader, REG_test_loader
