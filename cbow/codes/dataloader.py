# dataloader.py
# Load the training/validation data

import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data import to_map_style_dataset

# from utils.constants import (
#     CBOW_N_WORDS,
#     SKIPGRAM_N_WORDS,
#     MIN_WORD_FREQUENCY,
#     MAX_SEQUENCE_LENGTH,
# )
CBOW_N_WORDS = 4

MIN_WORD_FREQUENCY = 50
MAX_SEQUENCE_LENGTH = 256

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1

def get_eng_tokenizer():
    '''
    Get the tokenizer from torchtext.data.utils
    We use the English tokenizer to seperate the words
    '''

    tokenizer = get_tokenizer("basic_english", language="en")

    return tokenizer


def get_data_iterator(dataset_name, dataset_type, data_dir):
    '''
    Download the corpus or load the corpus from local files

    parameters:
    - dataset_name: the name of dataset, can be chosen from "WikiText2"
                    and "WikiText103"
    - dataset_type: the type of dataset, can be chosen from "train" and "val"
    - data_dir: the directory to save or load corpus

    return:
    - a map-style dataset
    '''

    if dataset_name == "WikiText103":
        data_iterator = WikiText103(root=data_dir, split=(dataset_type))
    elif dataset_name == "WikiText2":
        data_iterator = WikiText2(root=data_dir, split=(dataset_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")

    data_iterator = to_map_style_dataset(data_iterator)

    return data_iterator


def build_vocabulary(data_iterator, tokenizer):
    '''
    Build the vocabulary of all words in the corpus from an iterator.
    Only the words occur over MIN_WORD_FREQUENCY (a constant defined globally)
    will be considered.

    parameters:
    - data_iterator: the iterator of corpus
    - tokenizer: the tokenizer to seperate the words, obtained through the
    get_eng_tokenizer()

    return:
    - a vocabulary of all words in thr corpus
    '''

    vocabulary = build_vocab_from_iterator(
        map(tokenizer, data_iterator),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocabulary.set_default_index(vocabulary["<unk>"])

    return vocabulary


def collate_cbow(batch, text_pipline):
    '''
    Collation function used in the Dataloader.
    It truncates long paragraphs to make them less than MAX_SEQUENCE_LENGTH (a
    constant defined globally) words.
    Context is represented as N=CBOW_N_WORDS (a constant defined globally) past
    words and N=CBOW_N_WORDS future words.
    Paragraphs that are shorter than 2*CBOW_N_WORDS+1 words will be dropped.

    parameters:
    - batch: expected to be a list of text paragraphs.
    - text_pipline: the pipline to process the corpus, including tokenization
    and changing words into indices.

    return:
    - batch_input: prepared Tensors that can be used directly by the model.
    - batch_labels: prepared labels(Tensors) that can be used directly.

    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_labels` is a middle word.
    '''

    batch_input, batch_labels = [], []
    for text in batch:
        text_tokens_ids = text_pipline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_labels.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_input, batch_labels


def get_dataloader_and_vocab(model_name, dataset_name, dataset_type, data_dir,
                             batch_size, shuffle=True, vocab=None):
    '''
    Get DataLoader for model training/validation

    parameters:
    - model_name: can only be "cbow"
    - dataset_name: the name of dataset, can be chosen from "WikiText2"
                    and "WikiText103"
    - dataset_type: the type of dataset, can be chosen from "train" and "val"
    - data_dir: the directory to save or load corpus
    - batch_size: the batch size in DataLoader
    - shuffle: whether to shuffle the dataset, default to be True
    - vocab: vocabulary of the words in corpus. It can't be None when
    dataset_type='val'
    '''

    data_iterator = get_data_iterator(dataset_name, dataset_type, data_dir)
    tokenizer = get_eng_tokenizer()

    if not vocab:
        vocab = build_vocabulary(data_iterator, tokenizer)

    text_pipline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    else:
        raise ValueError("Choose model from: cbow")

    dataloader = DataLoader(
        data_iterator,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipline=text_pipline),
    )

    return dataloader, vocab