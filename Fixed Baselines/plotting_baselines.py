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
fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=False, figsize=(8, 8))
i = 0
for row in ax:
    for col in row:
        dir = Dirs[i]
        header_list = ["steps", "train", "test"]
        df = pd.read_csv(f'{dir}/training_stats.csv',names=header_list)
        df["steps"] = pd.to_numeric(df["steps"], downcast="integer")/1e6
        df["train"] = pd.to_numeric(df["train"], downcast="float")
        df["test"] = pd.to_numeric(df["test"], downcast="float")
        #df['test'] = df['test'].apply(lambda x: float(x[7:-1]))
        err = df.groupby(np.arange(len(df))//10).std()
        df = df.groupby(np.arange(len(df))//10).mean()
        col.plot(df['steps'],df['train'],color="blue")
        col.fill_between(df['steps'], df['train'] - err['train'], df['train'] + err['train'],
                        color='blue', alpha=0.1)

        col.plot(df['steps'],df['test'],color="orange")
        col.fill_between(df['steps'], df['test'] - err['test'], df['test'] + err['test'],
                        color='orange', alpha=0.3)
        
        col.set_title(f'{dir}')

        #col.savefig(f'{dir}plot_baseline.png')
        text[i] = df.iloc[-1]
        i += 1
    fig.text(0.52, 0, 'Steps (M)', ha='center')
    fig.text(0, 0.5, 'Reward', va='center', rotation='vertical')
    plt.tight_layout()
    plt.xticks([0,4,8])
#plt.locator_params(axis='x', nbins=3)
plt.savefig('alltogether.png',dpi=300)
plt.legend(['Train','Test'])
plt.show()
# df = pd.concat(text,axis=1)
# df.columns = Dirs
# df.to_excel('results.xlsx')
# print(df)