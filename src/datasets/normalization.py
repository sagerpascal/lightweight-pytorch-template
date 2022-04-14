def zero_norm(x, mean, std):
    return (x - mean) / std


def undo_zero_norm(x, mean, std):
    return x * std + mean
