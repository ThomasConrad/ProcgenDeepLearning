reader = open('output.txt')
out = reader.read().splitlines()

for line in out:
    print(line.split(" "))