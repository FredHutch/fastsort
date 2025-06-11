# roughclust
Rapidly sort a Pandas DataFrame of numeric values to minimize the
spatial distance between adjacent elements.

When working with large dense datasets, with large numbers of
both observations and features, it can be useful to perform
linkage clustering to visualy inspect what groupings of correlated
observations / features exist in the data.

The computational method most commonly used for this purpose,
linkage clustering, scales exponentially with dataset size
and quickly becomes impractical with 10,000's of items.
While there is an accelerated version of linkage clustering
provided by the `fastcluster` package, that method still
performs the full linkage clustering process.

Another useful approach for sorting large dense tables is to
use t-SNE or UMAP to project each item onto a single dimension.
While this process can better approximate the desired outcome,
it similarly incurs the all-by-all cost that scales exponentially
with dataset size.

This method performs a rough approximation which is intended to
provide a useful workaround for very large datasets.
It will:

1. Select a random subset of the dataset (param: `init_n`)
2. Split the dataset into the initialization subset and the remainder.
3. Perform exhaustive linkage clustering on the initialization subset to sort it
4. Compute the distance of the items in the remainder to the items in the sorted table
5. Select a subset of those items with the lowest computed distances (param: `iter_n`), and move those items to the sorted table
6. Repeat steps 4-5 until the dataset is complete. Note that the distances need only be calculated for those items which were moved to the sorted table in the last iteration.
