import os

index = [0,1,2,3,4]
target = [0,1,2,3,4,5,6,7,8,9]
network = ["mnist2x256_cw","mnist4x256_cw","mnist6x256_cw"]

results = []

for i in index:
    for t in target:
        for n in network:
            # get CW results
            eps_cw = float(open("summaries/{}_tar{}_ind{}.txt".format(n, t, i), 'r').read().split()[1])
            # get SOI results
            eps_soi = eps_cw
            for filename in os.listdir("{}_tar{}_ind{}".format(n, t, i)):
                res = open("{}_tar{}_ind{}/".format(n, t, i) + filename).read().split()[0]
                if res == "sat":
                    curEps = float(filename[:-8])
                    if curEps < eps_soi:
                        eps_soi = curEps
            if eps_cw < 0.00001:
                continue
            results.append("{},{},{},{}".format("{}_tar{}_ind{}".format(n, t, i), n, eps_cw, eps_soi))

for line in results:
    print(line)
