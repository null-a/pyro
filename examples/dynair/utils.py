def batch_expand(t, b):
    return t.expand((b,) + t.size())

def assert_size(t, expected_size):
    actual_size = t.size()
    assert actual_size == expected_size, 'Expected size {} but got {}.'.format(expected_size, tuple(actual_size))

def delta_mean(prev, mean_or_delta, use_delta):
    if use_delta and not prev is None:
        return prev + mean_or_delta
    else:
        return mean_or_delta
