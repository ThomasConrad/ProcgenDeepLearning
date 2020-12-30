# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:46:00 2020

@author: thoma
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Dirs = (next(os.walk('.'))[1])


text = [0]*len(Dirs)
plt.figure(1)
for i,dir in enumerate(Dirs):
    header_list = ["steps", "train", "test"]
    df = pd.read_csv(f'{dir}/training_stats_basic.csv',names=header_list)
    df["steps"] = pd.to_numeric(df["steps"], downcast="integer")
    df["train"] = pd.to_numeric(df["train"], downcast="float")

    df['test'] = df['test'].apply(lambda x: float(x[7:-1]))
    err = df.groupby(np.arange(len(df))//10).std()
    df = df.groupby(np.arange(len(df))//10).mean()
    plt.figure(figsize=(4,4))
    plt.plot(df['steps'],df['train'],color="blue")
    plt.fill_between(df['steps'], df['train'] - err['train'], df['train'] + err['train'],
                    color='blue', alpha=0.1)

    plt.plot(df['steps'],df['test'],color="orange")
    plt.fill_between(df['steps'], df['test'] - err['test'], df['test'] + err['test'],
                    color='orange', alpha=0.3)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title(f'{dir} baseline')

    plt.savefig(f'{dir}plot_baseline.png')
    text[i] = df.iloc[-1]

df = pd.concat(text,axis=1)
df.columns = Dirs
df.to_excel('results.xlsx')
print(df)