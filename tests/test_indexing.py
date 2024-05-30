from WildfireLogan.convLSTM import Input_target_Split
from WildfireLogan.convLSTM.utils.indexing import find_group


def test_InputTargetSplit():
    input_split, target_split = Input_target_Split(12500, 4, target_length=4)
    assert len(input_split[0]) == len(target_split[0])
    assert len(Input_target_Split(12500, 4)[0]) == 7500
    assert len(Input_target_Split(5000, 4)[0]) == 3000


def test_InputIndices():
    input_indices, _ = Input_target_Split(12500, 4)
    for i in range(len(input_indices)):
        assert input_indices[i][-1] % 100 <= 90


def test_TargetIndices():
    _, target_indices = Input_target_Split(12500, 4, target_length=1)
    for i in range(len(target_indices)):
        assert target_indices[i] % 100 >= 40


def test_find_group_typical():
    # Typical case
    seq = 5
    step_size = 2
    start = 1
    expected_result = [1, 3, 5, 7, 9]
    assert find_group(seq, step_size, start) == expected_result


def test_find_group_zero_step():
    # Zero step size
    seq = 4
    step_size = 0
    start = 1
    expected_result = [1, 1, 1, 1]
    assert find_group(seq, step_size, start) == expected_result
