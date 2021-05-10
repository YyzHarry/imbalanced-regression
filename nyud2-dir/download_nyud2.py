import os
import gdown
import zipfile

print("Downloading and extracting NYU v2 dataset to folder './data'...")
data_file = "./data.zip"
gdown.download("https://drive.google.com/uc?id=1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw")
print('Extracting...')
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall('.')
os.remove(data_file)
print("Completed!")