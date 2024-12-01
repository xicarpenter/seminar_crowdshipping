import pickle
import pandas as pd


def pretty_print(df1: pd.DataFrame, 
                 df2: pd.DataFrame, 
                 column: str, 
                 mode: str = "of"):
    """
    Prints a comparison of two dataframes with the mean and standard deviation for a given column.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe.
    df2 : pd.DataFrame
        The second dataframe.
    column : str
        The column to compare.
    mode : str, optional
        The mode of printing. Can be "of" or "10". The default is "of".
    """
    if mode == "of":
        col1 = "MAX_PROFIT"
        col2 = "MAX_PARCELS"
        empty = " "
      
    elif mode == "10":
        col1 = "10"
        col2 = "no_10"
        empty = ""

    print(column)
    print(f" {col1:<10}: {empty}{df1[column].mean(): .2f} +/- {round(df1[column].std(), 2)}", "\n",
          f"{col2:<10}: {df2[column].mean(): .2f} +/- {round(df2[column].std(),2)}", "\n",)



def print_stats(df1: pd.DataFrame, 
                df2: pd.DataFrame,
                mode: str = "of"):
    """
    Prints statistical comparisons between two dataframes for specified columns.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first dataframe containing the data to be compared.
    df2 : pd.DataFrame
        The second dataframe containing the data to be compared.
    mode : str, optional
        The mode of comparison. Can be "of" or "10". The default is "of".

    Prints
    ------
    The mean and standard deviation for the columns "profit", "parcels", and "runtime"
    for both dataframes. Also prints the mean differences between the two dataframes 
    for each of these columns.
    """
    if mode == "of":
        col1 = "MAX_PROFIT"
        col2 = "MAX_PARCELS"
      
    elif mode == "10":
        col1 = "10"
        col2 = "no_10"

    for col in ["profit", "parcels", "runtime"]:
      pretty_print(df1, df2, col, mode)

    print(f"--- DIFF ({col1} - {col2}) ---")
    print("Mean diff in profit", 
          round(df1["profit"].mean() - df2["profit"].mean(),2))
    print("Mean diff in parcels", 
          round(df1["parcels"].mean() - df2["parcels"].mean(),2))
    print("Mean diff in runtime", 
          round(df1["runtime"].mean() - df2["runtime"].mean(), 2), "\n")
    
    print(f"Median runtime {col1}", round(df1["runtime"].median(), 2))
    print(f"Median runtime {col2}", round(df2["runtime"].median(), 2), "\n")


def analyze_results(path: str, 
                    mode: str = "of"):
    """
    Analyzes the results of an experiment and prints a comparison of two dataframes.
    
    Parameters
    ----------
    path : str
        The path to the file containing the results.
    mode : str, optional
        The mode of comparison. Can be "of" or "10". The default is "of".
    
    Returns
    -------
    df1 : pd.DataFrame
        The first dataframe.
    df2 : pd.DataFrame
        The second dataframe.
    seeds : list
        The list of seeds used for the experiment.
    """
    with open(path, "rb") as f:
        results, seeds = pickle.load(f)

    if mode == "of":
        df1 = pd.DataFrame.from_dict(results["MAX_PROFIT"]).transpose()
        df2 = pd.DataFrame.from_dict(results["MAX_PARCELS"]).transpose()

    elif mode == "10":
        df1 = pd.DataFrame.from_dict(results["10"]).transpose()
        df2 = pd.DataFrame.from_dict(results["no_10"]).transpose()

    print_stats(df1, df2, mode)

    return df1, df2, seeds


if __name__ == "__main__":
    paths = [
        "num_analysis/results_of.pkl",
        "num_analysis/results_10_bin.pkl",
        "num_analysis/results_10_bin_2.pkl",
        "num_analysis/results_10_uni_4.pkl"
    ]

    for idx, path in enumerate(paths):
        if idx == 0:
            analyze_results(path, mode="of")

        else:
            analyze_results(path, mode="10")
