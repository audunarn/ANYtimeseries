"""
Readers CSV formatted time series files
"""
import pandas as pd


def _find_time_column(df):
    """Return preferred time column name."""
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    for key in ("time", "t", "referencetime", "reference_time", "datetime", "date"):
        if key in lower_to_col:
            return lower_to_col[key]

    for col in df.columns:
        col_lower = str(col).strip().lower()
        if "time" in col_lower:
            return col

    return df.columns[0]


def _prepare_table(df):
    """
    Normalize CSV data to a wide table where:
      * first column is time
      * remaining columns are numeric series
    """
    time_col = _find_time_column(df)
    lowered = {str(col).strip().lower(): col for col in df.columns}

    value_col = lowered.get("value")
    variable_col = None
    for candidate in ("elementid", "element_id", "variable", "name"):
        if candidate in lowered:
            variable_col = lowered[candidate]
            break

    if value_col is not None and variable_col is not None:
        long_df = df[[time_col, variable_col, value_col]].copy()
        long_df[value_col] = pd.to_numeric(long_df[value_col], errors="coerce")
        long_df = long_df.dropna(subset=[value_col])
        if not long_df.empty:
            wide = long_df.pivot_table(
                index=time_col,
                columns=variable_col,
                values=value_col,
                aggfunc="first",
            ).reset_index()
            wide.columns.name = None
            return wide, time_col

    return df, time_col


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
    df, time_col = _prepare_table(df)
    names = [name for name in df.columns if name != time_col]

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
    df, time_col = _prepare_table(df)

    if ind is None:
        cols = [col for col in df.columns if col != time_col]
    else:
        cols = list(ind)

    # Keep selected time column and selected numeric data columns.
    filtered_cols = [time_col]
    for col in cols:
        if col in df and pd.to_numeric(df[col], errors='coerce').notna().any():
            filtered_cols.append(col)

    return df[filtered_cols].T.to_numpy()
