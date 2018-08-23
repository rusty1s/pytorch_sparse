from torch_sparse import eye


def test_eye():
    index, value = eye(3)
    assert index.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert value.tolist() == [1, 1, 1]
