import os
import argparse
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./data")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ages, img_paths = [], []

    for filename in tqdm(os.listdir(os.path.join(args.data_path, 'AgeDB'))):
        _, _, age, gender = filename.split('.')[0].split('_')

        ages.append(age)
        img_paths.append(f"AgeDB/{filename}")

    outputs = dict(age=ages, path=img_paths)
    output_dir = os.path.join(args.data_path, "meta")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "agedb.csv")
    df = pd.DataFrame(data=outputs)
    df.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    main()
