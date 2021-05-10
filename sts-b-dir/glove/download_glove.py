import os
import wget
import zipfile

print("Downloading and extracting GloVe word embeddings...")
data_file = "./glove/glove.840B.300d.zip"
wget.download("http://nlp.stanford.edu/data/glove.840B.300d.zip", out=data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall('./glove')
os.remove(data_file)
print("\nCompleted!")