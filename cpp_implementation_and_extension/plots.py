import matplotlib.pyplot as plt
import numpy as np
import os

def mutualInfo(pws, pwe):
    ps = sum(pws)
    pe = sum(pwe)
    pw = [pwE + pwS for pwE, pwS in zip(pws, pwe)]
    MI = 0
    for e in range(2):
        for w in range(len(pws)):
            pxy = pws[w] if e == 0 else pwe[w]
            px = ps if e == 0 else pe
            py = pw[w]
            if pxy != 0:
                MI += pxy * np.log2(pxy / (px * py))
    return MI

MIs = []
for f in os.listdir():
    if f.startswith("hist"):
        dic = {}
        print(f)
        for line in f.split("_"):
            if line.__contains__("="):
                dic[line.split("=")[0]] = int(line.split("=")[1])
        print(dic)
        nbins = dic["nbins"]
        min = dic["min"]
        max = dic["max"]
        if dic["dim2"] == 2:
            continue
        if dic["nbins"] != 100:
            continue
        with open(f) as file:
            for wit in range(dic["amountW"]):
                pwsep = [int(i) / dic["amountS"] for i in file.readline().split(",")]
                pwent = [int(i) / dic["amountS"] for i in file.readline().split(",")]
#                 print(sum(pwsep), sum(pwent), sum(pwsep) + sum(pwent))
                MIs.append(mutualInfo(pwsep, pwent))
            plt.hist(MIs, bins=dic["nbins"], density=True, log=True)
            plt.show()
            plt.cla()


#             print(file.read())

# h, bins = np.histogram([], bins=10, range=(-1, 1))
# for i in range(h.shape[0]):
#     h[i] = i**2
# 
# plt.hist(h, bins=bins)
# plt.show()
