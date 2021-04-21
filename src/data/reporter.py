import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

# type hints
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

from IPython.display import display
from IPython.display import Markdown


def plot_size_distribution(
        sentences: List[str],
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs: Optional[Any]) -> None:
    """
    Plot the size distribution of sentences in a corpus.

    Parameters
    ----------
    sentences : list
        corpus with sentences (strings).
    title : str, optional
        plot description.
    figsize : (float, float), optional.
        tuple with plot width and height in inches.
    **kwargs :
        extra paramenters passed to `pyplot.hist`.
    """
    if title:
        display(Markdown(f'## {title}'))
    if figsize:
        plt.figure(figsize=figsize)

    sizes = [len(s.split()) for s in sentences]

    plt.subplot(1, 2, 1)
    plt.title('Sentence Size Distribution')
    plt.hist(sizes, **kwargs)
    plt.xlabel('# Words')
    plt.ylabel('# Sentences')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title('Sentence Size Distribution (log scale)')
    plt.hist(sizes, log=True, **kwargs)
    plt.xlabel('# Words')
    plt.ylabel('# Sentences')
    plt.grid()

    plt.show()


def plot_cumulative_size_distribution(
        sentences: List[str],
        title: Optional[str] = None,
        x_values: Optional[List[int]] = None,
        figsize: Optional[Tuple[int, int]] = None) -> None:
    """
    Plot the cumulative size distribution of sentences in a corpus.

    Parameters
    ----------
    sentences : list
        corpus with sentences (strings).
    title : str, optional
        plot description.
    x_values: list, optional
        values of x axis.
    figsize: (float, float), optional
        tuple with plot width and height, in inches.
    """
    if title:
        display(Markdown(f'## {title}'))
    if figsize:
        plt.figure(figsize=figsize)

    sizes = np.array([len(s.split()) for s in sentences])

    if x_values is None:
        x_values = np.linspace(0, sizes.max(), 100, dtype=np.int)

    y_values = list()
    for x in x_values:
        fl = sizes <= x
        counter = Counter(fl)
        y_values.append(counter[True])

    def _relative(a):
        return 100 * a / len(sentences)

    def _absolute(r):
        return r * len(sentences) / 100

    rel_yticks = [0, 20, 40, 60, 80, 100]

    abs_ax = plt.axes()
    abs_ax.set_ylabel('Sentences (frequency)')
    abs_ax.set_xlabel('Maximum number of words')
    abs_ax.set_yticks([int(_absolute(rel)) for rel in rel_yticks])

    rel_ax = abs_ax.secondary_yaxis('right', functions=(_relative, _absolute))
    rel_ax.set_ylabel('Sentences (percentage)')

    plt.plot(x_values, y_values)
    plt.grid()
    plt.show()


def corpus_analysis(df, dist_kw=None, cdist_kw=None):
    """Reporter for corpus distribution analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns to analyse.
    dist_kw : dict, optional
        extra arguments passed to `plot_size_distribution`.
    cdist_kw : dict, opitonal
        extra arguments passed to `plot_cumulative_size_distribution`.
    """
    # sentence size distribution
    display(Markdown('# Sentence Size'))
    if dist_kw is None:
        dist_kw = dict(
            figsize=(15, 4),
            bins=100
        )

    for col in df.columns:
        plot_size_distribution(
            df[col].to_list(),
            title=col,
            **dist_kw
        )

    # cumulative size distribution
    display(Markdown('# Cumulative Sentence Size'))
    if cdist_kw is None:
        cdist_kw = dict(
            figsize=(15, 4)
        )
    x_values = cdist_kw.pop('x_values', dict())

    for col in df.columns:
        plot_cumulative_size_distribution(
            df[col],
            title=col,
            x_values=x_values.pop(col, None),
            **cdist_kw
        )
