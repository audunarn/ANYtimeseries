"""
Readers CSV formatted time series files
"""
import pandas as pd


def read_names(path):
    """
    Read time series names from a comma-separated file.

    Parameters
    ----------
    path : str
        Name of .csv file

    Returns
    -------
    list
        Time series names

    Notes
    -----
    The series names are expected found on the header row. Time is expected to be in the first column.

    """
    # pandas will infer the format e.g. delimiter.
    df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8')
    names = list(df)
    _ = names.pop(0)    # remove time which is assumed to be in the first column

    numeric_names = []
    for name in names:
        # Skip non-numeric columns (e.g. source file strings).
        if pd.to_numeric(df[name], errors='coerce').notna().any():
            numeric_names.append(name)

    return numeric_names


def read_data(path, ind=None):
    """
    Read time series data arranged column wise on a comma-separated file.

    Parameters
    ----------
    path : str
        CSV file path (relative or absolute)
    ind : list|tuple, optional
        Read only a subset of the data series specified by name. The default, None, results in all data columns
        being read.

    Returns
    -------
    array
        Time and data

    """
    df = pd.read_csv(path, sep=None, engine='python')  # pandas will infer the format e.g. delimiter.

    if ind is None:
        cols = list(df.columns[1:])
    else:
        cols = list(ind)

    # Keep first column (time) and selected numeric data columns.
    filtered_cols = [df.columns[0]]
    for col in cols:
        if col in df and pd.to_numeric(df[col], errors='coerce').notna().any():
            filtered_cols.append(col)

    return df[filtered_cols].T.to_numpy()
