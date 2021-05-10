'''Preprocessing functions and pipeline'''
import nltk
nltk.download('punkt')
import torch
import logging
import numpy as np
from collections import defaultdict

from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp_mods.numeric_field import NumericField

from tasks import STSBTask

PATH_PREFIX = './glue_data/'

ALL_TASKS = ['sts-b']
NAME2INFO = {'sts-b': (STSBTask, 'STS-B/')}

for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], PATH_PREFIX + v[1])

def build_tasks(args):
    '''Prepare tasks'''
    
    task_names = [args.task]
    tasks = get_tasks(args, task_names, args.max_seq_len)

    max_v_sizes = {'word': args.max_word_v_size}
    token_indexer = {}
    token_indexer["words"] = SingleIdTokenIndexer()

    logging.info("\tProcessing tasks from scratch")
    word2freq = get_words(tasks)
    vocab = get_vocab(word2freq, max_v_sizes)
    word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)
    for task in tasks:
        train, val, test = process_task(task, token_indexer, vocab)
        task.train_data = train
        task.val_data = val
        task.test_data = test
        del_field_tokens(task)
    logging.info("\tFinished indexing tasks")

    train_eval_tasks = [task for task in tasks if task.name in task_names]
    logging.info('\t  Training and evaluating on %s', ', '.join([task.name for task in train_eval_tasks]))

    return train_eval_tasks, vocab, word_embs

def del_field_tokens(task):
    ''' Save memory by deleting the tokens that will no longer be used '''

    all_instances = task.train_data + task.val_data + task.test_data
    for instance in all_instances:
        if 'input1' in instance.fields:
            field = instance.fields['input1']
            del field.tokens
        if 'input2' in instance.fields:
            field = instance.fields['input2']
            del field.tokens

def get_tasks(args, task_names, max_seq_len):
    '''Load tasks'''
    tasks = []
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        task = NAME2INFO[name][0](args, NAME2INFO[name][1], max_seq_len, name)
        tasks.append(task)
    logging.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))

    return tasks

def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq = defaultdict(int)

    def count_sentence(sentence):
        '''Update counts for words in the sentence'''
        for word in sentence:
            word2freq[word] += 1
        return

    for task in tasks:
        splits = [task.train_data_text, task.val_data_text, task.test_data_text]
        for split in [split for split in splits if split is not None]:
            for sentence in split[0]:
                count_sentence(sentence)
            for sentence in split[1]:
                count_sentence(sentence)

    logging.info("\tFinished counting words")

    return word2freq

def get_vocab(word2freq, max_v_sizes):
    '''Build vocabulary'''
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes['word'])
    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    for word, _ in words_by_freq[:max_v_sizes['word']]:
        vocab.add_token_to_namespace(word, 'tokens')
    logging.info("\tFinished building vocab. Using %d words", vocab.get_vocab_size('tokens'))

    return vocab

def get_embeddings(vocab, vec_file, d_word):
    '''Get embeddings for the words in vocab'''
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, d_word)
    with open(vec_file) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                idx = vocab.get_token_index(word)
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index('@@PADDING@@')] = 0.
    embeddings = torch.FloatTensor(embeddings)
    logging.info("\tFinished loading embeddings")

    return embeddings

def process_task(task, token_indexer, vocab):
    '''
    Convert a task's splits into AllenNLP fields then
    Index the splits using the given vocab (experiment dependent)
    '''
    if hasattr(task, 'train_data_text') and task.train_data_text is not None:
        train = process_split(task.train_data_text, token_indexer)
    else:
        train = None
    if hasattr(task, 'val_data_text') and task.val_data_text is not None:
        val = process_split(task.val_data_text, token_indexer)
    else:
        val = None
    if hasattr(task, 'test_data_text') and task.test_data_text is not None:
        test = process_split(task.test_data_text, token_indexer)
    else:
        test = None

    for instance in train + val + test:
        instance.index_fields(vocab)

    return train, val, test

def process_split(split, indexers):
    '''
    Convert a dataset of sentences into padded sequences of indices.
    '''
    inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
    inputs2 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[1]]
    labels = [NumericField(l) for l in split[-1]]

    if len(split) == 4:   # weight
        weights = [NumericField(w) for w in split[2]]
        instances = [Instance({"input1": input1, "input2": input2, "label": label, "weight": weight}) for \
                     (input1, input2, label, weight) in zip(inputs1, inputs2, labels, weights)]
    else:
        instances = [Instance({"input1": input1, "input2": input2, "label": label}) for \
                        (input1, input2, label) in zip(inputs1, inputs2, labels)]

    return instances
