import pandas as pd
reader = open('output.txt')
out = reader.read().splitlines()
steps = []
train = []
test = []
for line in out:
    steps.append(line.split()[1])
    train.append(line.split()[4])
    test.append(line.split()[8])
df = pd.DataFrame(list(zip(steps,train,test)),columns=['steps','train','test'])
df.to_csv('training_stats.csv',index=False,header=False)
