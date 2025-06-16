import pytest 
from unittest import TestCase
# import torch.multiprocessing as mp 
from multiprocessing import Process
import multiprocessing as mp 
import argparse
from unit_test.test_utils import spawn



# if __name__ == "__main__":
#     import os
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ["MASTER_PORT"] = "12355"

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ngpu", default=2)
#     parser.add_argument("--world_size", default=2)
#     args = parser.parse_args()
#     mp.spawn(fail,nprocs=2,args=(args,))


def success(q):
    q.put("True")


def fail(q):
    q.put("False")


class testSample(TestCase):

    def test(self):
        # size = 4
        # processes = []
        # q = mp.Queue()
        # for i in range(4):
        #     p = Process(target=fail,args=(q,))
        #     p.start()
        #     processes.append(p)

        # for p in processes:
        #     p.join(1)
        size = 4
        q = spawn(size, fail)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "True")
            cnt += 1 
        self.assertEqual(cnt, size)
