# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:46:00 2020

@author: thoma
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
header_list = ["steps", "train", "test"]
df = pd.read_csv('training_stats_crop.csv',names=header_list)
df["steps"] = pd.to_numeric(df["steps"], downcast="integer")
df["train"] = pd.to_numeric(df["train"], downcast="float")
df["test"] = pd.to_numeric(df["test"], downcast="float")

plt.figure(figsize=(4,4))

err = df.groupby(np.arange(len(df))//10).std()
#print(err)
df = df.groupby(np.arange(len(df))//10).mean()
plt.plot(df['steps'],df['train'],color="blue")
plt.fill_between(df['steps'], df['train'] - err['train'], df['train'] + err['train'],
                 color='blue', alpha=0.1)

plt.plot(df['steps'],df['test'],color="orange")
plt.fill_between(df['steps'], df['test'] - err['test'], df['test'] + err['test'],
                 color='orange', alpha=0.3)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("StarPilot - Crop")

plt.show()
