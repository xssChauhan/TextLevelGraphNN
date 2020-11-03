import contractions
import re
import torch
import numpy as np
import pandas as pd

from typing import List
from typing import Dict
from typing import Iterable
from typing import Tuple
from typing import Union

from torch import device


def clean_and_tokenize(text: str) -> List[str]:

    text = contractions.fix(text)

    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)     
    text = re.sub(r",", " , ", text) 
    text = re.sub(r"!", " ! ", text) 
    text = re.sub(r"\(", " \( ", text) 
    text = re.sub(r"\)", " \) ", text) 
    text = re.sub(r"\?", " \? ", text) 
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip().lower().split()


def get_vocab(docs: List[List[str]]) -> Dict[str, int]:
    '''
    Get a dictionary with mapping word -> id
    '''
    vocab = dict()
    
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)+1

    print("Vocabulary size: ", len(vocab))
        
    return vocab


def preprocess(docs: List[str]) -> List[List[str]]:

    preprocessed = []

    for doc in docs:
        preprocessed.append(clean_and_tokenize(
            doc
        ))
    return preprocessed


def load_glove(filename) -> Dict[str, np.array]:
    """
    Load glove embeddings
    """
    embeddings_index = {}
    with open(filename, "r") as f:
        for line in f:
            values = line.split(' ')
            word = values[0] ## The first entry is the word
            coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
            embeddings_index[word] = coefs

    return embeddings_index

def read_dataset(filename: str, preprocessing: bool=False) -> pd.DataFrame:
    """
    Read the R8 dataset
    """

    labels = []
    texts = []

    with open(filename, "r") as f:
        data = map(
            lambda x: x.strip().split("\t"),
            f
        )
        for label, text in data:
            labels.append(label)
            texts.append(text)
        
        if preprocessing:
            texts = preprocess(texts)

    # Wrap the data nicely into a dataframe
    df = pd.DataFrame()
    df["labels"] = labels
    df["texts"] = texts

    return df

def load_embeddings(filename: str, vocab: Dict[str, int]):
    """
    Convert Embeddings into vector that can be loaded in PyTorch
    """
    base_embedding = load_glove(filename)

    dimension = base_embedding["is"].shape[0]

    vectors = np.zeros(
        (len(vocab) +1, dimension)
    )

    unknown_words = set()

    for word in vocab:
        if word in base_embedding:
            vectors[vocab[word],:] = base_embedding[word]
        else:
            unknown_words.add(word)
            vectors[vocab[word],:] = np.random.uniform(
                -0.25, 0.25, dimension
            )

    
    return vectors, unknown_words


def collate_fn(data: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = [e[0] for e in data]
    texts = [e[1] for e in data]

    labels = torch.LongTensor(labels)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return labels, texts


def get_neighbours(X: torch.Tensor, n_neighbours: int=2) -> torch.Tensor:
    
    neighbours = []
    pad = [0] * n_neighbours
    x_ids = pad + list(X) + pad

    for i in range(n_neighbours, len(x_ids) - n_neighbours):
        x = x_ids[i - n_neighbours:i] + x_ids[i+1: i+n_neighbours + 1]
        neighbours.append(x)
    
    return torch.Tensor(neighbours)

def make_graph(batch:Tuple[torch.Tensor, torch.Tensor], vocab: Dict[str, int], n_neighbours: int, device='cpu'):

        batch_size, max_len = batch[1].shape

        nb_nodes = len(vocab)

        neighbours = torch.zeros((
            batch_size, max_len, 2*n_neighbours
        )).to(device)
        edges = torch.zeros((
            batch_size, max_len, 2*n_neighbours
        )).to(device)

        for i, data in enumerate(batch[1]):
            data = data[data > 0]
            length = len(data)
            nx = get_neighbours(data, n_neighbours).to(device)
            ed = ((data[data > 0]-1)*nb_nodes).reshape(-1,1) + nx
            ed[nx == 0] = 0

            neighbours[i, :length] = nx
            edges[i, :length] = nx

        # Return the batch data, neighbours matrix, edge matrix, batch labels
        return batch[1], neighbours, edges, batch[0] 
