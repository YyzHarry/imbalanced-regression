import os
import wget

print("Downloading IMDB faces...")
imdb_file = "./data/imdb_crop.tar"
wget.download("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar", out=imdb_file)
print("Downloading WIKI faces...")
wiki_file = "./data/wiki_crop.tar"
wget.download("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar", out=wiki_file)
print("Extracting IMDB faces...")
os.system(f"tar -xvf {imdb_file} -C ./data")
print("Extracting WIKI faces...")
os.system(f"tar -xvf {wiki_file} -C ./data")
os.remove(imdb_file)
os.remove(wiki_file)
print("\nCompleted!")