def find_group(seq, step_size, start):
    """Generates a sequence of numbers based on the given sequence length,
       step size, and starting point.

    Args:
        seq (int): The number of elements in the sequence.
        step_size (int): The step size between consecutive
                         elements in the sequence.
        start (int): The starting value of the sequence.

    Returns:
        list: A list of numbers forming the sequence.
    """

    group = []
    for i in range(seq):
        group.append(start + step_size*i)
    return group


def Input_target_Split(size, seq, step_size=10, period=100, target_length=1):
    """Splits the data into input and target sequences
       based on the provided parameters.

    Args:
        size (int): The total size of the dataset; should be a multiple of 100.
        seq (int): The length of each input sequence.
        step_size (int, optional): The step size (time step) between
                                   consecutive elements in the sequence.
                                   Defaults to 10.
        period (int, optional): The period of the spread of wildfire in train
                                and test dataset. Defaults to 100.
        target_length (int, optional): The length of each target sequence
                                    element. Defaults to 1.

    Raises:
        ValueError: If `seq` times `step_size` is greater
                    than or equal to `period`.

    Returns:
        tuple: A tuple containing two lists - the input sequences
               and the target sequences.
    """

    if seq * step_size >= period:
        raise ValueError("Ensure seq times step_size less than period")
    num_chunk = size // period
    input = []
    target = []
    for i in range(num_chunk):
        chunk_input = []
        for j in range(i * period, (i + 1) * period):
            group = find_group(seq, step_size, j)
            if group[-1] >= (i + 1) * period:
                break
            chunk_input.append(group)
        input.extend(chunk_input[:-step_size])
        target.extend(chunk_input[step_size:])
    if target_length == 1:
        return input, [sublist[-1] for sublist in target]
    else:
        return input, target
