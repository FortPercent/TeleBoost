
import os

TEST_LIST=[
    "1.1.test_case_verify_accuracy.py",
]

idx=0
for test in TEST_LIST:
    print(f"Running {idx+1} test: {test}")
    os.system(f"python3 {test}")
    idx+=1

