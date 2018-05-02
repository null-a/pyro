def batch_expand(t, b):
    return t.expand((b,) + t.size())

def assert_size(t, expected_size):
    actual_size = t.size()
    assert actual_size == expected_size, 'Expected size {} but got {}.'.format(expected_size, tuple(actual_size))
