"""B2W Corpus Inteface.

This module contains a high-level Python interface
to B2W-Reviews01 corpus. B2W-Reviews01 is an open
corpus of product reviews available at:
    https://github.com/b2wdigital/b2w-reviews01
"""

import pandas as pd
import urllib.request


# B2W repository
CORPUS_CSV = 'https://raw.githubusercontent.com/' \
    'b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv'


def download_csv(path, url=CORPUS_CSV, **kwargs):
    """Creates a local copy of the csv corpus.


    Parameters
    ----------
    path : str
        local path to save the data.
    url : str, optional
        url of B2W dataset.
    **kwargs
        arguments passed to pandas.read_csv function.
        Only have effects for partial download.

    Notes
    -----
    The corpus can be downloaded on partial or full size. To enable partial
    download use the `nrows` argument. If `nrows` is set to zero, then the
    complete file will be downloaded.
    """
    nrows = kwargs.get('nrows', 0)
    if nrows and (nrows > 0):
        data = pd.read_csv(url, sep=';', encoding='utf-8', **kwargs)
        data.to_csv(path, sep=';', encoding='utf-8', index=False)
    else:
        urllib.request.urlretrieve(url=url, filename=path)


def get_recommendation_data(path, **kwargs):
    """Reads recomendation columns from B2W corpus.

    Creates a new dataset from the B2W corpus. This new dataset contains the
    columns `review_text` and `recommend_to_a_friend` renamed as `text` and
    `target`, respectively. This dataset is usefull for classification task.

    Parameters
    ----------
    path : str
        path to B2W corpus in csv format. Can be the URL to original csv corpus
        or any URL.
    **kwargs
        extra arguments passed to ``pandas.read_csv`` function.

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    This dataset is usefull for classification tasks.
    """
    usecols = dict(review_text='text', recommend_to_a_friend='target')
    df = pd.read_csv(path, usecols=usecols, **kwargs)
    df.columns = [usecols[c] for c in df.columns]

    return df
