from typing import Tuple
import pandas as pd
import random
from scipy.spatial import distance
from scipy.cluster import hierarchy
import sys
import logging

# Log to stdout
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Create a logger for this module
logger = logging.getLogger(__name__)


def roughclust(
    df: pd.DataFrame,
    metric="euclidean",
    init_method="ward",
    init_n=1000,
    iter_n=1000,
):
    """
    Approximate linkage-clustering based sorting of a dense DataFrame.
    This function performs a rough clustering of a DataFrame using an iterative approach.
    The process is as follows:

        1. Select a random subset of the dataset (param: `init_n`)
        2. Split the dataset into the initialization subset and the remainder.
        3. Perform exhaustive linkage clustering on the initialization subset to sort it
        4. Compute the distance of the items in the remainder to the items in the sorted table
        5. Select a subset of those items with the lowest computed distances (param: `iter_n`),
           and move those items to the sorted table
        6. Repeat steps 4-5 until the dataset is complete. Note that the distances need only be
           calculated for those items which were moved to the sorted table in the last iteration.


    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be clustered and sorted.
    metric : str, optional
        The distance metric to use for clustering. Default is "euclidean".
    init_method : str, optional
        The method to use for hierarchical clustering on the initial subset. Default is "ward".
    init_n : int, optional
        The number of items to sample for the initial clustering. Default is 1000.
    iter_n : int, optional
        The number of items to select in each iteration after the initial clustering. Default is 1000.

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted based on the rough clustering.
    """

    rc = RoughClust(
        df=df,
        metric=metric,
        init_method=init_method,
        init_n=init_n,
        iter_n=iter_n,
    )

    return rc.run()


def _add_best_matches(
    best_matches: pd.DataFrame,
    init_df: pd.DataFrame,
    remaining_df: pd.DataFrame,
    iter_n: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add the best matches from the remaining items to the initial sorted items.

    Parameters
    ----------
    best_matches : pd.DataFrame
        A DataFrame containing the best matches and their distances.
    init_df : pd.DataFrame
        The DataFrame containing the already sorted initial items.
    remaining_df : pd.DataFrame
        The DataFrame containing the items that are not yet sorted.

    Returns
    -------
    None
    """

    # Update the initial DataFrame with the best matches
    for idx, best_match in best_matches["best_match"].head(iter_n).items():
        logger.info(idx, best_match)
        # Get the index position of the best match
        match_idx = init_df.index.get_loc(best_match)

        # If this is the first match, we can just append it
        if match_idx == 0:
            init_df = pd.concat([remaining_df.loc[[idx]], init_df])
        elif match_idx == len(remaining_df) - 1:
            init_df = pd.concat([init_df, remaining_df.loc[[idx]]])
        else:
            # Insert the best match at the correct position in the initial DataFrame
            init_df = pd.concat([
                init_df.iloc[:match_idx],
                remaining_df.loc[[idx]],
                init_df.iloc[match_idx:]
            ])

    best_matches.drop(
        index=best_matches.head(iter_n).index,
        inplace=True
    )

    return init_df, remaining_df.drop(best_matches.head(iter_n).index)


def _empty_best_matches(index: list) -> pd.DataFrame:
    """
    Create an empty DataFrame to hold the best matches for each item in the index.

    Parameters
    ----------
    index : list
        The index of the DataFrame for which to create the empty best matches DataFrame.

    Returns
    -------
    pd.DataFrame
        An empty DataFrame with the same index as the input index.
    """
    return pd.DataFrame(index=index, columns=["best_match", "best_distance"])


class RoughClust:
    df: pd.DataFrame
    sorted_index: list
    unsorted_index: list
    best_matches: pd.DataFrame
    metric: str
    init_method: str
    init_n: int
    iter_n: int

    def __init__(
        self,
        df: pd.DataFrame,
        metric="euclidean",
        init_method="ward",
        init_n=1000,
        iter_n=1000
    ):
        """
        Initialize the RoughClust object with the DataFrame and parameters.
        Parameters
        ----------
        init_df : pd.DataFrame
            The DataFrame to be clustered and sorted.
        metric : str, optional
            The distance metric to use for clustering. Default is "euclidean".
        init_method : str, optional
            The method to use for hierarchical clustering on the initial subset. Default is "ward".
        init_n : int, optional
            The number of items to sample for the initial clustering. Default is 1000.
        iter_n : int, optional
            The number of items to select in each iteration after the initial clustering. Default is 1000.
        """

        # Keep track of the sorted index position for each item
        self.sorted_index = []
        # Keep track of which items still need to be sorted
        self.unsorted_index = df.index.tolist()
        # Create an empty DataFrame to hold the best matches
        self.best_matches = _empty_best_matches(self.unsorted_index)

        self.df = df
        self.metric = metric
        self.init_method = init_method
        self.init_n = init_n
        self.iter_n = iter_n

    def run(self) -> pd.DataFrame:
        # If there are no items in the DataFrame, return an empty DataFrame
        if self.df.empty:
            logger.info("Input DataFrame is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        # If there are fewer than three items in the DataFrame, return the DataFrame as is
        if len(self.df) < 3:
            logger.info("Input DataFrame has fewer than three items. Returning the DataFrame as is.")
            return self.df

        # Sort the initial items in the DataFrame using hierarchical clustering
        logger.info("Starting rough clustering...")

        # Call the method to sort the initial items
        self.sort_initial_items()

        # Compute the best matches for the initial items
        if len(self.unsorted_index) > 0:
            self.add_best_matches(
                self.unsorted_index,
                self.sorted_index,
            )

        while len(self.unsorted_index) > 0:
            self.sort_additional_items()

        return self.df.loc[self.sorted_index].copy()

    def sort_initial_items(self):
        """
        Sort the initial items in the DataFrame using hierarchical clustering.
        This method samples a subset of the DataFrame, performs clustering, and iteratively finds the best matches.
        """

        # Randomly sample the initial items
        init_idx = random.sample(
            self.unsorted_index,
            k=min(self.init_n, len(self.unsorted_index))
        )

        logger.info(f"Initial items sampled: {len(init_idx):,}")

        # Perform hierarchical clustering on the initial items
        logger.info("Running hierarchical clustering on initial items")
        logger.info(f"method={self.init_method}, metric={self.metric}")
        Z = hierarchy.linkage(
            self.df.loc[init_idx],
            method=self.init_method,
            metric=self.metric
        )

        # Get the order of items from the linkage matrix
        order = hierarchy.leaves_list(Z)

        # Add to the sorted index the order of the initial items
        self.sorted_index.extend([
            init_idx[i] for i in order
        ])
        for idx in init_idx:
            self.unsorted_index.remove(idx)

    def add_best_matches(self, query_idx, ref_idx):
        """
        Find the best matches between the remaining items and the initial sorted items.
        This method computes distances and updates the best matches DataFrame.

        Parameters
        ----------
        query_idx : list
            The index of the items to find matches for.
        ref_idx : list
            The index of the reference items to match against.
        """

        # Create a DataFrame for the items to search for matches
        query_df = self.df.loc[query_idx]

        # Create a DataFrame of items to search against
        ref_df = self.df.loc[ref_idx]

        # Calculate distances between remaining items and initial sorted items
        all_distances = distance.cdist(query_df, ref_df, metric=self.metric)
        logger.info(f"Calculated distances between {len(query_idx):,} query items and {len(ref_df):,} reference items.")

        _new_best_matches = 0
        for idx, distances in zip(query_idx, all_distances):

            best_distance = distances.min()
            best_match = ref_df.index[distances.argmin()]

            # If the best match is better than anything previously, update the best_matches DataFrame
            if (
                pd.isnull(self.best_matches.loc[idx, "best_distance"]) or 
                best_distance < self.best_matches.loc[idx, "best_distance"]
            ):
                self.best_matches.loc[idx, "best_match"] = best_match
                self.best_matches.loc[idx, "best_distance"] = best_distance
                _new_best_matches += 1

        logger.info(f"Searched {len(query_idx):,} items against {len(ref_idx):,} reference items.")
        logger.info(f"Found {_new_best_matches:,} new best matches.")

    def sort_additional_items(self):

        # Pick a random subset of unsorted items to add to the sorted items
        if len(self.unsorted_index) < self.iter_n:
            to_add = self.unsorted_index
        else:
            to_add = random.sample(
                self.unsorted_index,
                k=self.iter_n
            )
        logger.info(f"Adding {len(to_add):,} items to the sorted items.")
        # Add the best matches for these items to the sorted items
        for idx in to_add:
            best_match = self.best_matches.loc[idx, "best_match"]
            self.sorted_index.insert(
                self.sorted_index.index(best_match),
                idx
            )
            self.unsorted_index.remove(idx)

        # Find the best matches for the newly added items
        self.add_best_matches(
            self.unsorted_index,
            to_add
        )
