# Word2Vec-from-scratch

![](https://visitor-badge.glitch.me/badge?page_id=Doslim.Word2Vec-from-scratch)

PyTorch implementations of the Continuous Bags of Words (CBOW) model  - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) and an improved version. We borrowed some codes from this well-organized post about Word2Vec model - [Word2vec with PyTorch: Reproducing Original Paper](https://notrocketscience.blog/word2vec-with-pytorch-implementing-original-paper/).

We also provide some evaluation tools and a brief report.

## Project Structure and Environments
We use PyTorch to implement the original version of Word2Vec .We only consider CBOW here. The required environments are as follows.
- torch-1.11.0
- torchtext-0.12.0
- web-0.0.1
- nltk-3.6.5
  
The structure of our projects is as follows.
- cbow: implementations of CBOW.
    - /codes: contain all the codes.
        - main.py: the entrance of our project.
        - train.py: define the class to train the model.
        - model.py: define the model.
        - dataloader.py: load the corpus for training.
        - help\_function.py: define some tools to parse the configurations.
        - evaluation.py: define functions to evaluate word embeddings.
        - config.yaml: store the configurations for model trianing.
    - /weight: the directory to save models and training logs.
    - /data: the directory of corpus used in training and evaluation.
- cbow\_improve: improved version of CBOW.
    - /codes: includes the following new codes.
      - model\_finetune.py: define the model to fine-tune the embeddings.
      - train\_finetune.py: define the function to train the fine-tune model.
      - data\_finetune.py: prepare data for training the fine-tune model.
      - main\_finetune.py: the entrance to fine-tune the word embeddings.
    - /weight: the directory to save models and training logs.
    - /data: the directory of corpus used in training and evaluation.
- report.pdf: a brief introduction of our implementation details, the improved model and evaluation results.

## Usage
You can use the following command to run our codes in the codes directory.
```
python main.py --config=config.yaml
```
The meaning of each configuration can be found in our report.
