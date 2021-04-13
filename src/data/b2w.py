"""B2W Corpus Inteface.

This module contains a high-level Python interface
to B2W-Reviews01 corpus. B2W-Reviews01 is an open
corpus of product reviews available at:
    https://github.com/b2wdigital/b2w-reviews01
"""

import pandas as pd
import re
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

    Creates a new dataset from B2W corpus. This new dataset contains
    `review_text` and `recommend_to_a_friend` columns renamed as `text` and
    `target`, respectively.

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
    This dataset is usefull for classification tasks, and contains the values:
    - text      is the main content of the product review.
    - target    is a binary value where 1 represents the user would recommend
                the product to a friend, and 0 represents that he would not.
                This column may contain null and other values.
    """
    usecols = dict(review_text='text', recommend_to_a_friend='target')
    df = pd.read_csv(path, usecols=usecols, **kwargs)
    df.columns = [usecols[c] for c in df.columns]

    # change target to numeric
    df.target = df.target.apply(lambda t: 1 if t == 'Yes' else t)
    df.target = df.target.apply(lambda t: 0 if t == 'No' else t)

    return df


def cls_dataprep(data, drop_na=False):
    """Data preparation for classification task.

    This is a default procedure used to preprocess the dataset for
    classification tasks. The `text` column of the dataset receives the
    following transformations:
        Lower case.
        Numbers changed to `NUM` token.
        Insert spaces between ponctuatoins.

    Parameters
    ----------
    data : pandas.DataFrame
        dataset with `text` and `target` columns.
    drop_na : bool, optional
        remove rows with empty values.

    Returns
    -------
    pandas.DataFrame
    """
    if drop_na:
        data.dropna(inplace=True)

    # lower case
    data.text = data.text.apply(lambda s: s.lower().strip())

    # creates a number token
    data.text = data.text.apply(lambda s: re.sub(r"\d+", "NUM", s))
    data.text = data.text.apply(lambda s: re.sub(r"NUM[.,]NUM", "NUM", s))

    # insert spaces between words and ponctuations
    # exemple: "he is a boy." => "he is a boy ."
    data.text = data.text.apply(lambda s: re.sub(r"([?.!,¿])", r" \1 ", s))
    data.text = data.text.apply(lambda s: re.sub(r'[" "]+', " ", s))

    return data
