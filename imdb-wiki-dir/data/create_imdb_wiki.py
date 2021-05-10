import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from datetime import datetime


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # date
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)
    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def combine_dataset(path='meta'):
    args = get_args()
    data_imdb = pd.read_csv(os.path.join(args.data_path, path, "imdb.csv"))
    data_wiki = pd.read_csv(os.path.join(args.data_path, path, "wiki.csv"))
    data_imdb['path'] = data_imdb['path'].apply(lambda x: f"imdb_crop/{x}")
    data_wiki['path'] = data_wiki['path'].apply(lambda x: f"wiki_crop/{x}")
    df = pd.concat((data_imdb, data_wiki))
    output_path = os.path.join(args.data_path, path, "imdb_wiki.csv")
    df.to_csv(str(output_path), index=False)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--min_score", type=float, default=1., help="minimum face score")
    args = parser.parse_args()
    return args


def create(db):
    args = get_args()
    mat_path = os.path.join(args.data_path, f"{db}_crop", f"{db}.mat")
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    ages, img_paths = [], []

    for i in tqdm(range(len(face_score))):
        if face_score[i] < args.min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 200):
            continue

        ages.append(age[i])
        img_paths.append(full_path[i][0])

    outputs = dict(age=ages, path=img_paths)
    output_dir = os.path.join(args.data_path, "meta")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{db}.csv")
    df = pd.DataFrame(data=outputs)
    df.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    create("imdb")
    create("wiki")
    combine_dataset()
