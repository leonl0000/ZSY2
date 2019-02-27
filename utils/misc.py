import time
import numpy as np

def timer(f,args,iters):
    tic = time.time()
    for i in range(iters):
        f(*args)
    toc = time.time()
    print("%d ms"%(1000*(toc-tic)))
    print("%f us/iter"%(1e6*(toc-tic)/iters))

def RoundRobin(Ms, matches = 10):
    numModels = len(Ms)
    wins = np.zeros(numModels)
    for i in range(numModels):
        if i == 0:
            tic = time.time()
        if i == 1:
            toc = time.time()
            t = toc-tic
            t_ = t * (numModels-1.)/2
            secs = t_%60
            mins = (t_/60)%60
            hous = t_/3600
            print("Round 1 took %.2f seconds"%(t))
            print("Projected Runtime: %d:%d:%d"%(hous, mins, secs))
        print(i)
        for j in range(i+1, numModels):
                w = np.sum([zsy.game(Ms[i], Ms[j])[1] for k in range(matches)])
                wins[i] += w
                wins[j] += matches-w
    return wins