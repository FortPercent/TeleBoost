import pytest 
from unittest import TestCase
# import torch.multiprocessing as mp 
from multiprocessing import Process
import multiprocessing as mp 
import argparse
from unit_test.test_utils import spawn


def success(q):
    q.put("True")


def fail(q):
    q.put("False")


class testMPTesting(TestCase):

    def testSuccess(self):
        size = 4
        q = spawn(size, success)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "True")
            cnt += 1 
        self.assertEqual(cnt, size)
    
    def testFail(self):
        size = 4
        q = spawn(size, fail)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "False")
            cnt += 1 
        self.assertEqual(cnt, size)
