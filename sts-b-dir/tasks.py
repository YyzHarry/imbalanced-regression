import os
import nltk
import codecs
import logging
import numpy as np
from scipy.ndimage import convolve1d
from util import get_lds_kernel_window, STSShotAverage

def process_sentence(sent, max_seq_len):
    '''process a sentence using NLTK toolkit'''
    return nltk.word_tokenize(sent)[:max_seq_len]

def load_tsv(data_file, max_seq_len, s1_idx=0, s2_idx=1, targ_idx=2, targ_fn=None, skip_rows=0, delimiter='\t', args=None):
    '''Load a tsv '''
    sent1s, sent2s, targs = [], [], []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.strip().split(delimiter)
                sent1 = process_sentence(row[s1_idx], max_seq_len)
                if (targ_idx is not None and not row[targ_idx]) or not len(sent1):
                    continue

                if targ_idx is not None:
                    targ = targ_fn(row[targ_idx])
                else:
                    targ = 0

                if s2_idx is not None:
                    sent2 = process_sentence(row[s2_idx], max_seq_len)
                    if not len(sent2):
                        continue
                    sent2s.append(sent2)

                sent1s.append(sent1)
                targs.append(targ)

            except Exception as e:
                logging.info(e, " file: %s, row: %d" % (data_file, row_idx))
                continue

    if args is not None and args.reweight != 'none':
        assert args.reweight in {'inverse', 'sqrt_inv'}
        assert args.reweight != 'none' if args.lds else True, "Set reweight to \'inverse\' (default) or \'sqrt_inv\' when using LDS"

        bins = args.bucket_num
        value_lst, bins_edges = np.histogram(targs, bins=bins, range=(0., 5.))

        def get_bin_idx(label):
            if label == 5.:
                return bins - 1
            else:
                return np.where(bins_edges > label)[0][0] - 1

        if args.reweight == 'sqrt_inv':
            value_lst = [np.sqrt(x) for x in value_lst]
        num_per_label = [value_lst[get_bin_idx(label)] for label in targs]

        logging.info(f"Using re-weighting: [{args.reweight.upper()}]")

        if args.lds:
            lds_kernel_window = get_lds_kernel_window(args.lds_kernel, args.lds_ks, args.lds_sigma)
            logging.info(f'Using LDS: [{args.lds_kernel.upper()}] ({args.lds_ks}/{args.lds_sigma})')
            smoothed_value = convolve1d(value_lst, weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[get_bin_idx(label)] for label in targs]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]

        return sent1s, sent2s, weights, targs

    return sent1s, sent2s, targs

class STSBTask:
    ''' Task class for Sentence Textual Similarity Benchmark.  '''
    def __init__(self, args, path, max_seq_len, name="sts-b"):
        ''' '''
        super(STSBTask, self).__init__()
        self.args = args
        self.name = name
        self.train_data_text, self.val_data_text, self.test_data_text = None, None, None
        self.val_metric = 'mse'
        self.scorer = STSShotAverage(metric=['mse', 'l1', 'gmean', 'pearsonr', 'spearmanr'])
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        tr_data = load_tsv(os.path.join(path, 'train_new.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x), args=self.args)
        val_data = load_tsv(os.path.join(path, 'dev_new.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x))
        te_data = load_tsv(os.path.join(path, 'test_new.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: np.float32(x))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        logging.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False, type=None):
        metric = self.scorer.get_metric(reset, type)
        
        return metric
