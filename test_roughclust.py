import pandas as pd
import numpy as np
from roughclust import roughclust


def test_roughclust_basic():
    # Make a DataFrame with 1000 rows and columns, filled with random numbers
    data = pd.DataFrame(
        np.random.rand(1000, 1000),
        columns=[f"col_{i}" for i in range(1000)],
        index=[f"row_{i}" for i in range(1000)]
    )

    # Perform rough clustering to sort the rows
    sorted_data = roughclust(data, init_n=10, iter_n=10)

    # Make sure that the output is the same shape as the input
    assert sorted_data.shape == data.shape


def test_roughclust_blocks():
    # Make a DataFrame with blocks of rows that all are very similar
    # This should be very easy to sort
    n_blocks = 10
    block_size = 100

    data = pd.DataFrame([
        [
            int(col_ix == i_block) + np.random.normal(0, 1. / n_blocks)
            for col_ix in range(n_blocks)
        ]
        for i_block in range(n_blocks)
        for _ in range(block_size)
    ])

    sorted = roughclust(data, init_n=50, iter_n=10)

    # Check to see that most rows from the same block are close together
    # We can do this by checking the distance between the indices of the sorted DataFrame
    # The distance should be small for most rows, and large for rows that are from different blocks
    # Calculate the absolute difference between the indices of the sorted DataFrame
    offset = np.abs(sorted.index[:-1] - sorted.index[1:])
    assert (offset <= block_size).mean() > 0.9, "Most rows from the same block should be close together"
