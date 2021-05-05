"""Fine Tuning a BERT model for classification.

BERT model: neuralmind/bert-base-portuguese-cased
Dataset: B2W text review.
"""
import os
import sys

try:
    import src
except Exception:
    module_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)


import dotenv
import re
import yaml

from src import gcp_util
from src import util
from src.data import b2w


# global variables
CONFIG = None
STORAGE = None


def setup(cfg_file):
    """Prepare the environment for experimentation.

    Parameters
    ----------
    cfg_file : str
        path to a YAML file, containing all experient configurations.
    """
    # load environment variables
    dotenv.load_dotenv()

    # storage connection
    global STORAGE
    bucket = gcp_util.get_bucket()
    STORAGE = f'gs://{bucket.name}'
    print('Using storage:', STORAGE)

    # load experiment configurations
    global CONFIG
    CONFIG = open(cfg_file, 'r')
    CONFIG = yaml.load(CONFIG, Loader=yaml.FullLoader)


def text_tranformation(data):
    """Default data preprocessing.

    This is a default procedure used to preprocess the dataset for
    classification tasks. The `text` column of the dataset receives the
    following transformations:
        Lower case.
        Numbers changed to `NUM` token.
        Insert spaces between punctuations.

    Parameters
    ----------
    data : pandas.DataFrame
        dataset with `text` and `target` columns.

    Returns
    -------
    pandas.DataFrame
    """
    # remove rows with empty values.
    data.dropna(inplace=True)

    # lower case
    data.text = data.text.apply(lambda s: s.lower().strip())

    # creates a number token
    data.text = data.text.apply(lambda s: re.sub(r"\d+", "NUM", s))
    data.text = data.text.apply(lambda s: re.sub(r"NUM[.,]NUM", "NUM", s))

    # insert spaces between words and punctuations
    # exemple: "he is a boy." => "he is a boy ."
    data.text = data.text.apply(lambda s: re.sub(r"([?.!,¿])", r" \1 ", s))
    data.text = data.text.apply(lambda s: re.sub(r'[" "]+', " ", s))

    return data


def dataprep(
        csv_dataset,
        csv_train, csv_test, csv_val,
        size_train, size_test, size_val,
        nrows=0, max_sentence_size=0):
    """Transform raw B2W data into ready to use data.

    This method execute the data preparation step. Transform a B2W csv
    file in a raw format into a ready-to-use data.

    Parameters
    ----------
    csv_dataset : str
        raw csv dataset file name (without bucket name).
    csv_train : str
        path to save train dataset (without bucket name).
    csv_test : str
        path to save test dataset (without bucket name).
    csv_validation : str
        path to save validation dataset (without bucket name).
    size_train : float
        float number between 0 and 1, split size to train the model.
    size_test : float
        float number between 0 and 1, split size to test the model.
    size_validation : float
        float number between 0 and 1, split size to validate the model.
    nrows : int, optional
        read the first `nrows` from raw csv dataset. Defaults to 0,
        means to use full dataset.
    max_sentence_size : int, optional
        remove sentences with more tokens than `max_sentence_size`.
    """
    # storage path
    storage_dataset = STORAGE + '/' + csv_dataset
    storage_train = STORAGE + '/' + csv_train
    storage_test = STORAGE + '/' + csv_test
    storage_val = STORAGE + '/' + csv_val

    # download to storage
    if gcp_util.exists_on_storage(csv_dataset):
        print('Found Source Dataset at', storage_dataset)
    else:
        print(f'Saving B2W dataset to {storage_dataset}')
        b2w.download_csv(storage_dataset, nrows=nrows)
        print(f'B2W download complete.')

    # read and prepare
    print('Preprocessing classification dataset...')
    dataset = b2w.get_recommendation_data(storage_dataset)
    dataset = text_tranformation(dataset)

    # remove rows with more tokens than max_sentence_size
    if max_sentence_size:
        print(f'Removing sentences bigger than {max_sentence_size} tokens...')
        ntokens = dataset.text.apply(lambda s: len(s.split()))
        fl_tokens = ntokens <= max_sentence_size
        dataset = dataset[fl_tokens]

    # data split
    print('Creating Train, Test and Validation dataset...')
    print('Dataset:')
    print('\tShape:', dataset.shape)
    vc = dataset.target.value_counts(normalize=True)
    print(f'\tTarget: 0 ({vc[0]:.2f}) 1 ({vc[1]:.2f})')

    train_df, test_df, val_df = util.datasplit(
        dataset, train_size=size_train, test_size=size_test, val_size=size_val)

    # train dataset
    print('Train Dataset:')
    vc = train_df.target.value_counts(normalize=True)
    train_df.to_csv(storage_train, sep=';', encoding='utf-8', index=False)
    print('\tShape:', train_df.shape)
    print(f'\tTarget: 0 ({vc[0]:.2f}) 1 ({vc[1]:.2f})')
    print('\tPath:', storage_train)

    # test dataset
    print('Test Dataset:')
    vc = test_df.target.value_counts(normalize=True)
    test_df.to_csv(storage_test, sep=';', encoding='utf-8', index=False)
    print('\tShape:', test_df.shape)
    print(f'\tTarget: 0 ({vc[0]:.2f}) 1 ({vc[1]:.2f})')
    print('\tPath:', storage_test)

    # validation dataset
    print('Validation Dataset:')
    vc = val_df.target.value_counts(normalize=True)
    val_df.to_csv(storage_val, sep=';', encoding='utf-8', index=False)
    print('\tShape:', val_df.shape)
    print(f'\tTarget: 0 ({vc[0]:.2f}) 1 ({vc[1]:.2f})')
    print('\tPath:', storage_val)


def run(cfg_file):
    # environment setup
    setup(cfg_file=cfg_file)

    # dataprep
    if '--dataprep' in sys.argv:
        print('Running Data Preparation.')
        dataprep(**CONFIG['data'])


if __name__ == '__main__':
    run(cfg_file='experiments/cls_b2w_bert.yaml')
