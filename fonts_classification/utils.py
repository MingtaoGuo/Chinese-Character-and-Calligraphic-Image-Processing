import numpy as np


def random_read_batch(data, labels, batch_size):
    nums = data.shape[0]
    rand_select = np.random.randint(0, nums, [batch_size])
    batch = data[rand_select, :]
    label = labels[0, rand_select]
    return batch, label