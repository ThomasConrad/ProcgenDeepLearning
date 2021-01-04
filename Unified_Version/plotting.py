import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def plot_training_data(filename, figurename, title, size=4):
    
    plt.figure(figsize=(size,size))
    plt.tight_layout()
    plt.title(title)
    plt.xlabel("Steps (M)")
    plt.ylabel("Reward")

    header_list = ["steps", "train", "test"]
    df = pd.read_csv(filename, names=header_list)
    df["steps"] = pd.to_numeric(df["steps"], downcast="integer")/1e6
    df["train"] = pd.to_numeric(df["train"], downcast="float")
    df["test"] = pd.to_numeric(df["test"], downcast="float")

    err = df.groupby(np.arange(len(df))//10).std()
    df = df.groupby(np.arange(len(df))//10).mean()
    plt.plot(df['steps'],df['train'],color="blue")
    plt.fill_between(df['steps'], df['train'] - err['train'], df['train'] + err['train'],
                     color='blue', alpha=0.1)

    plt.plot(df['steps'],df['test'],color="orange")
    plt.fill_between(df['steps'], df['test'] - err['test'], df['test'] + err['test'],
                     color='orange', alpha=0.3)
    
    plt.xticks([0,4,8])

    plt.savefig(figurename, dpi=300)