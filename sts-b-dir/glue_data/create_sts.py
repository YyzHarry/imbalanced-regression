import os
import sys
import codecs
import numpy as np
import urllib
if sys.version_info >= (3, 0):
    import urllib.request
import zipfile

URLLIB=urllib
if sys.version_info >= (3, 0):
    URLLIB=urllib.request

##### Downloading raw STS-B dataset
print("Downloading and extracting STS-B...")
data_file = "./glue_data/STS-B.zip"
URLLIB.urlretrieve("https://dl.fbaipublicfiles.com/glue/data/STS-B.zip", data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall('./glue_data')
os.remove(data_file)
print("Completed!")

##### Creating STS-B-DIR dataset
print("Creating STS-B-DIR dataset...")
contents = {'train': [], 'dev': [], 'test': []}
target = {'train': [], 'dev': [], 'test': []}

for set_name in ['train', 'dev']:
    with codecs.open(f'./glue_data/STS-B/{set_name}.tsv', 'r', 'utf-8') as data_fh:
        for _ in range(1):
            titles = data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            contents[set_name].append(row)
            row = row.strip().split('\t')
            targ = row[9]
            target[set_name].append(np.float32(targ))

bins = 20
target_all = target['train'] + target['dev']
contents_all = contents['train'] + contents['dev']

bins_numbers, bins_edges = np.histogram(a=target_all, bins=bins, range=(0., 5.))

def _get_bin_idx(label):
    if label == 5.:
        return bins - 1
    else:
        return np.where(bins_edges > label)[0][0] - 1

bins_contents = [None] * bins
bins_targets = [None] * bins
bins_numbers = list(bins_numbers)

for i, score in enumerate(target_all):
    bin_idx = _get_bin_idx(score)
    if bins_contents[bin_idx] is None:
        bins_contents[bin_idx] = []
    if bins_targets[bin_idx] is None:
        bins_targets[bin_idx] = []
    bins_contents[bin_idx].append(contents_all[i])
    bins_targets[bin_idx].append(score)
contents_bins_numbers = []
targets_bins_numbers = []
for i in range(bins):
    contents_bins_numbers.append(len(bins_contents[i]))
    targets_bins_numbers.append(len(bins_targets[i]))

new_contents = {'train': [None] * bins, 'dev': [None] * bins, 'test': [None] * bins}
new_targets = {'train': [None] * bins, 'dev': [None] * bins, 'test': [None] * bins}
select_num = 100
for i in range(bins):
    new_index = {}
    new_dev_test_index = np.random.choice(a=list(range(bins_numbers[i])), size=select_num, replace=False)
    new_index['train'] = np.setdiff1d(np.array(range(bins_numbers[i])), new_dev_test_index)
    new_index['dev'] = np.random.choice(a=new_dev_test_index, size=int(select_num / 2), replace=False)
    new_index['test'] = np.setdiff1d(new_dev_test_index, new_index['dev'])
    for set_name in ['train', 'dev', 'test']:
        new_contents[set_name][i] = np.array(bins_contents[i])[new_index[set_name]]
        new_targets[set_name][i] = np.array(bins_targets[i])[new_index[set_name]]

new_contents_merged = {'train': [], 'dev': [], 'test': []}
for i in range(bins):
    for set_name in ['train', 'dev', 'test']:
        new_contents_merged[set_name] += new_contents[set_name][i].tolist()
print('Number of samples for train/dev/test set in STS-B-DIR:',
      len(new_contents_merged['train']), len(new_contents_merged['dev']), len(new_contents_merged['test']))

for set_name in ['train', 'dev', 'test']:
    for i in range(len(new_contents_merged[set_name])):
        content_split = new_contents_merged[set_name][i].split('\t')
        content_split[0] = str(i)
        content_split = '\t'.join(content_split)
        new_contents_merged[set_name][i] = content_split
for set_name in ['train', 'dev', 'test']:
    with open(f'./glue_data/STS-B/{set_name}_new.tsv', 'w') as f:
        f.write(titles)
        for i in range(len(new_contents_merged[set_name])):
            f.write(new_contents_merged[set_name][i])
print("STS-B-DIR dataset created! ('./glue_data/STS-B/(train_new/dev_new/test_new).tsv')")
