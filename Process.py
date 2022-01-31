import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, BertTokenizerFast,RobertaTokenizer, RobertaModel,DistilBertTokenizer, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
le = LabelEncoder()

def data_load(dataset):
# We have prepared a chitchat dataset with 5 labels
    with open(dataset) as csv_file:
        df = pd.read_csv(csv_file)
        df.head()
        df["label"].value_counts()
        print(df)
        # Converting the labels into encodings
        df['label'] = le.fit_transform(df['label'])
        # check class distribution
        df['label'].value_counts(normalize = True)
        print(df)
    return df


def data_process(model,dataset):

    df = data_load(dataset)

    if(model=="bert"):
        # Load the BERT tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # Import BERT-base pretrained model
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
    if(model=="roberta"):
    # Load the Roberta tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # Import Roberta pretrained model
        bert_model = RobertaModel.from_pretrained("roberta-base")
    if(model=="distilbert"):
    # Load the DistilBert tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # Import the DistilBert pretrained model
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    # In this example we have used all the utterances for training purpose
    train_text, train_labels = df["Text"], df["label"]

    # Based on the histogram we are selecting the max len as 8
    max_seq_len = 8
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer(
        train_text.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    #define a batch size
    batch_size = 16
    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # DataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return(bert_model,train_labels,train_dataloader,tokenizer)


