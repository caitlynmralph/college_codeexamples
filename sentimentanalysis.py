import csv
import string
import numpy as np
import matplotlib.pyplot as plt

posts = []
exclude = set(string.punctuation)

with open('data.csv',"rt") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        row = ''.join(ch for ch in row if ch not in exclude)
        row = ''.join(ch for ch in row if ch not in exclude)
        row = ''.join(ch for ch in row if ch not in exclude)
        posts.append(row)

with open('negativewords.txt', 'r') as f:
    negatives = f.readlines()

for index, item in enumerate(negatives):
    negatives[index] = item.rstrip()

scores = []

for post in posts:
    scores.append(0)
    for word in post.split(" "):
        if word in negatives:
            i = posts.index(post)
            scores[i] = scores[i] + 1

LNF = []

with open('LNF.csv',"rt") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        for number in row:
            LNF.append(int(number))

LNF_data = np.asarray(LNF)
scores_data = np.asarray(scores)

N = 272

colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

plt.scatter(scores_data, LNF_data, s=area, c=colors, alpha=0.5)
plt.title("Scatterplot Negative Scores vs. LNF")
plt.xlabel("Negative Scores")
plt.ylabel("LNF")
plt.show()
