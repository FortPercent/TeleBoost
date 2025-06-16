import multiprocessing as mp

def spawn(nprocs, func, *args):
    # size = 4
    processes = []
    q = mp.Queue()
    for i in range(nprocs):
        p = mp.Process(target=func,args=(q, ) + args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join(1)
    
    return q
