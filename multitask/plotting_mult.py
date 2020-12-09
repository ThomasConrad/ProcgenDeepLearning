# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:46:00 2020

@author: thoma
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

header_list = ["steps", "train", "test_star", "test_coin", "test_fish"]


df1 = pd.read_csv('training_stats_multi_basic.csv',names=header_list)
df1["steps"] = pd.to_numeric(df1["steps"], downcast="integer")
df1["train"] = pd.to_numeric(df1["train"], downcast="float")
df1["test_star"] = pd.to_numeric(df1["test_star"], downcast="float")
df1["test_coin"] = pd.to_numeric(df1["test_coin"], downcast="float")
df1["test_fish"] = pd.to_numeric(df1["test_fish"], downcast="float")

plt.figure(figsize=(4,4))

err1 = df1.groupby(np.arange(len(df1))//10).std()
#print(err1)
df1 = df1.groupby(np.arange(len(df1))//10).mean()
plt.plot(df1['steps'],df1['train'],color="blue")
plt.fill_between(df1['steps'], df1['train'] - err1['train'], df1['train'] + err1['train'],
                 color='blue', alpha=0.1)

plt.plot(df1['steps'],df1['test_star'],color="red")
plt.fill_between(df1['steps'], df1['test_star'] - err1['test_star'], df1['test_star'] + err1['test_star'],
                 color='red', alpha=0.3)
plt.plot(df1['steps'],df1['test_coin'],color="orange")
plt.fill_between(df1['steps'], df1['test_coin'] - err1['test_coin'], df1['test_coin'] + err1['test_coin'],
                 color='orange', alpha=0.3)
plt.plot(df1['steps'],df1['test_fish'],color="green")
plt.fill_between(df1['steps'], df1['test_fish'] - err1['test_fish'], df1['test_fish'] + err1['test_fish'],
                 color='green', alpha=0.3)

plt.legend(["Train", "Test/StarPilot", "Test/Coinrun", "Test/Bigfish"])
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Multi - Basic")



df2 = pd.read_csv('training_stats_multi_mix_2_basic.csv',names=header_list)
df2["steps"] = pd.to_numeric(df2["steps"], downcast="integer")
df2["train"] = pd.to_numeric(df2["train"], downcast="float")
df2["test_star"] = pd.to_numeric(df2["test_star"], downcast="float")
df2["test_coin"] = pd.to_numeric(df2["test_coin"], downcast="float")
df2["test_fish"] = pd.to_numeric(df2["test_fish"], downcast="float")

plt.figure(figsize=(4,4))

err2 = df2.groupby(np.arange(len(df2))//10).std()
#print(err2)
df2 = df2.groupby(np.arange(len(df2))//10).mean()
plt.plot(df2['steps'],df2['train'],color="blue")
plt.fill_between(df2['steps'], df2['train'] - err2['train'], df2['train'] + err2['train'],
                 color='blue', alpha=0.1)

plt.plot(df2['steps'],df2['test_star'],color="red")
plt.fill_between(df2['steps'], df2['test_star'] - err2['test_star'], df2['test_star'] + err2['test_star'],
                 color='red', alpha=0.3)
plt.plot(df2['steps'],df2['test_coin'],color="orange")
plt.fill_between(df2['steps'], df2['test_coin'] - err2['test_coin'], df2['test_coin'] + err2['test_coin'],
                 color='orange', alpha=0.3)
plt.plot(df2['steps'],df2['test_fish'],color="green")
plt.fill_between(df2['steps'], df2['test_fish'] - err2['test_fish'], df2['test_fish'] + err2['test_fish'],
                 color='green', alpha=0.3)

plt.legend(["Train", "Test/StarPilot", "Test/Coinrun", "Test/Bigfish"])
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Multi - MixReg")

plt.show()