import re
import os 
import torch
import numpy as np 
test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_data/")
baseline_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_data/baseline")
baseline_logs = os.listdir(baseline_log_path)
logs_to_verify = os.listdir(test_data_path)

def extract_loss(log_file, drop_first=True):
    # 正则表达式模式
    pattern = r'loss:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)'

    with open(log_file, "r") as f:
        log = f.read()
        # 搜索所有匹配
        matches = re.findall(pattern, log)

        # 提取匹配的数字
        loss_values = [eval(match[0]) for match in matches]
    if drop_first:
        loss_values = loss_values[1:]
    return loss_values

def extract_loss_with_iteration(log_file):
    losses = {}
    iteration_pattern = r"iteration\s*(\d+)"
    loss_pattern = r'loss:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)'
    with open(log_file, "r") as f:
        for line in f:
            iteration_match = re.search(iteration_pattern, line)
            loss_match = re.search(loss_pattern, line)
            if iteration_match and loss_match:
                iteration = eval(iteration_match.group(1))
                loss = eval(loss_match.group(1))
                losses[iteration] = loss
                
    return losses 


def isClose(base, target, rtol, atol):
    if abs(base-target) < abs(base) * rtol + atol:
        return True 
    return False 

baseline_loss_dict = {}
for log in baseline_logs:
    if not log.endswith(".log"): continue 
    baseline_loss_dict[log] = extract_loss_with_iteration(os.path.join(baseline_log_path, log))
    
test_loss_dict = {}
for log in logs_to_verify:
    if not log.endswith(".log"): continue 
    test_loss_dict[log] = extract_loss_with_iteration(os.path.join(test_data_path, log))
    
for key, baseline_loss in baseline_loss_dict.items():
    print(f"baseline {key} {baseline_loss}")
    test_loss = test_loss_dict.get(key)
    print(f"test {key} {test_loss}")
    if not test_loss:
        print(f"{key} loss is empty. Training has probably failed. Not OK.")
        continue 
    no_loss_iters = []
    wrong_loss_iters = []
    ok_iters = []
    for iter, loss_iter in baseline_loss.items():
        if not iter in test_loss:
            no_loss_iters.append(iter)
            continue 
        test_loss_iter = test_loss[iter]
        isclose = isClose(loss_iter, test_loss_iter, rtol=0.02, atol=0.001)
        if isclose:
            ok_iters.append(iter)
        else:
            wrong_loss_iters.append(iter)
    if not no_loss_iters or no_loss_iters == [1]: # allow first loss missing
        if not wrong_loss_iters:
            print(f"{key} test OK!")
            continue 
    print(f"{key} test failed, losses of these iterations are missing: {no_loss_iters}, \
            losses of these iterations are not close to baseline: {wrong_loss_iters}.")


