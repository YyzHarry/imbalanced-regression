from os.path import join
import pandas as pd
import matplotlib.pyplot as plt


BASE_PATH = './data'


def visualize_dataset(db="agedb"):
    file_path = join(BASE_PATH, "meta", "agedb.csv")
    data = pd.read_csv(file_path)
    _, ax = plt.subplots(figsize=(6, 3), sharex='all', sharey='all')
    ax.hist(data['age'], range(max(data['age']) + 2))
    # ax.set_xlim([0, 102])
    plt.title(f"{db.upper()} (total: {data.shape[0]})")
    plt.tight_layout()
    plt.show()


def make_balanced_testset(db="agedb", max_size=30, seed=666, verbose=True, vis=True, save=False):
    file_path = join(BASE_PATH, "meta", f"{db}.csv")
    df = pd.read_csv(file_path)
    df['age'] = df.age.astype(int)
    val_set, test_set = [], []
    import random
    random.seed(seed)
    for value in range(121):
        curr_df = df[df['age'] == value]
        curr_data = curr_df['path'].values
        random.shuffle(curr_data)
        curr_size = min(len(curr_data) // 3, max_size)
        val_set += list(curr_data[:curr_size])
        test_set += list(curr_data[curr_size:curr_size * 2])
    if verbose:
        print(f"Val: {len(val_set)}\nTest: {len(test_set)}")
    assert len(set(val_set).intersection(set(test_set))) == 0
    combined_set = dict(zip(val_set, ['val' for _ in range(len(val_set))]))
    combined_set.update(dict(zip(test_set, ['test' for _ in range(len(test_set))])))
    df['split'] = df['path'].map(combined_set)
    df['split'].fillna('train', inplace=True)
    if verbose:
        print(df)
    if save:
        df.to_csv(str(join(BASE_PATH, f"{db}.csv")), index=False)
    if vis:
        _, ax = plt.subplots(3, figsize=(6, 9), sharex='all')
        df_train = df[df['split'] == 'train']
        ax[0].hist(df_train['age'], range(max(df['age'])))
        ax[0].set_title(f"[{db.upper()}] train: {df_train.shape[0]}")
        ax[1].hist(df[df['split'] == 'val']['age'], range(max(df['age'])))
        ax[1].set_title(f"[{db.upper()}] val: {df[df['split'] == 'val'].shape[0]}")
        ax[2].hist(df[df['split'] == 'test']['age'], range(max(df['age'])))
        ax[2].set_title(f"[{db.upper()}] test: {df[df['split'] == 'test'].shape[0]}")
        ax[0].set_xlim([0, 120])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    make_balanced_testset()
    visualize_dataset()
