"""Module with general purpose utilities.
"""
import gcp_util
import numpy as np
import tensorflow as tf


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

    return train, test, val


def get_tf_strategy(accelerator='tpu'):
    """Creates a TensorFlow Strategy for distributed training.

    Creates and configure a strategy for TPU/GPU accelerators. This
    strategy can be used for TensorFlow model training.

    Parameters
    ----------
    accelerator : str, optional
        accelerator name to use for training.
        Available options are None, 'tpu' and 'gpu'.
    """
    # default strategy
    strategy = tf.distribute.get_strategy()

    if accelerator == 'tpu':
        print('Using TPU Strategy')
        ip = gcp_util.get_tpu_ip()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(ip)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.TPUStrategy(tpu)

    elif accelerator == 'gpu':
        raise NotImplementedError

    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy
