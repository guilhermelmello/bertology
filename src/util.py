"""Module with general purpose utilities.
"""
import numpy as np


def datasplit(data, train_size, test_size, val_size):
    """Train, test, validation data split.

    Creates 3 subsets from `data`, based on each size argument. The rows of
    `data` are sampled without repositions. It means that each subset have
    no sobrepositions, avoiding data leakage.

    Parameters
    ----------
    data : pandas.DataFrame
        the dataset to split.
    train_size : float
        float number between 0 and 1.
        Percentage of `data` used as train dataset.
    test_size : float
        float number between 0 and 1.
        Percentage of `data` used as test dataset.
    val_size : float
        float number between 0 and 1.
        Percentage of `data` used as validation dataset.

    Returns
    -------
    train : pandas.DataFrame
        train dataset with size train_size.
    test : pandas.DataFrame
        test dataset with size test_size.
    val : pandas.DataFrame
        validation dataset with size val_size.

    Raises
    ------
    ValueError
        the sum of size splits must be less or equal 1
    """
    if np.sum((train_size, test_size, val_size)) > 1:
        raise ValueError('Split size sum must be less or equal 1.')

    indexes = data.index.to_numpy()
    np.random.shuffle(indexes)

    _len = len(indexes)
    _train = int(_len * train_size)
    _test = int(_len * (train_size + test_size))

    train_index, test_index, val_index = np.split(indexes, [_train, _test])

    train = data.loc[train_index]
    test = data.loc[test_index]
    val = data.loc[val_index]

    return train, val, test
