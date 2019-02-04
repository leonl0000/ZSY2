import time

def timer(f,args,iters):
    tic = time.time()
    for i in range(iters):
        f(*args)
    toc = time.time()
    print("%d ms"%(1000*(toc-tic)))
    print("%f us/iter"%(1e6*(toc-tic)/iters))

