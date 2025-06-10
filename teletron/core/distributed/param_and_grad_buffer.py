import torch


def finish_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the grad buffer.

    When overlap_grad_reduce is set to True, dispatches asynchronous communication
    calls. When overlap_grad_reduce is set to False, calls synchronous
    communication ops.
    """


    from megatron.training import get_args 
    args = get_args()

    if args.debug:
        from tensorwatch import report_grad
        report_grad(self.buckets)
    

    # print("finish grad sync")
    for bucket in self.buckets:
        #bucket.grad_data = bucket.grad_data.to(torch.float32)
        bucket.finish_grad_sync()
        #bucket.grad_data = bucket.grad_data.to(torch.bfloat16)
