import torch
import math

class BucketSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, seed=42, drop_last=True, shuffle=True, infinite=True):
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.pre_consumed_samples = consumed_samples
        self.consumed_samples = 0
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        self.epoch = 0
        self.bucket_indices = self.dataset.get_bucket_index_list()

        self.compute_num_samples()

        self.total_size = self.num_samples * self.data_parallel_size * self.micro_batch_size

        self.which_bucket_iter = self.fixed_infinite_iter([len(bucket)/self.total_size for bucket in self.bucket_indices])

        self.bucket_indices_iters = [self.fixed_infinite_bucket_iter(bucket) for bucket in self.bucket_indices]

    def fixed_infinite_iter(self, ratio):
        g = torch.Generator().manual_seed(self.seed)
        w = torch.tensor(ratio, dtype=torch.float)
        while True:
            yield from torch.multinomial(w, 10**6, replacement=True, generator=g).tolist()

    def fixed_infinite_bucket_iter(self, bucket):
        while True:
            g = torch.Generator().manual_seed(self.seed + self.consumed_samples)
            index = torch.randperm(len(bucket), generator=g).tolist()
            bucket_indices = [bucket[i] for i in index]
            for i in range(0, len(bucket_indices), self.micro_batch_size * self.data_parallel_size):
                start_idx = i + self.data_parallel_rank * self.micro_batch_size
                yield bucket_indices[start_idx: start_idx + self.micro_batch_size]

    def compute_num_samples(self):
        self.num_samples = 0
        for i in range(len(self.bucket_indices)):
            if self.drop_last:
                self.num_samples += len(self.bucket_indices[i]) // self.micro_batch_size // self.data_parallel_size
                self.bucket_indices[i] = self.bucket_indices[i][:len(self.bucket_indices[i]) // self.micro_batch_size // self.data_parallel_size * self.micro_batch_size * self.data_parallel_size]
            else:
                self.num_samples += math.ceil(math.ceil(len(self.bucket_indices[i]) / self.micro_batch_size) / self.data_parallel_size)
                self.bucket_indices[i] += self.bucket_indices[i][:math.ceil(math.ceil(len(self.bucket_indices[i]) / self.micro_batch_size) / self.data_parallel_size) * self.micro_batch_size * self.data_parallel_size - len(self.bucket_indices[i])]

    def skip_consumed_samples(self):
        for _ in range(self.pre_consumed_samples // self.micro_batch_size // self.data_parallel_size):
            which_bucket_idx = next(self.which_bucket_iter)
            self.consumed_samples += 1
            next(self.bucket_indices_iters[which_bucket_idx])

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.pre_consumed_samples > 0:
            self.skip_consumed_samples()

        while True:
            which_bucket_idx = next(self.which_bucket_iter)
            self.consumed_samples += 1
            yield next(self.bucket_indices_iters[which_bucket_idx])
