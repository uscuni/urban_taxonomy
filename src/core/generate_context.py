import numba
import pandas as pd
import numpy as np

@numba.njit(parallel=True)
def numba_limit_range(rows, cols, partial_vals, output_vals):
    # print(partial_vals)
    ngroups = int(rows[-1]) + 1
    nrows = rows.shape[0]
    result = np.empty((ngroups, partial_vals.shape[1] * output_vals))

    istart = 0
    for g in range(ngroups):
        # # find focal start
        # istart = 0
        # while istart < nrows and rows[istart] != g:
        #     istart += 1

        # find neighbors
        iend = istart + 1
        while iend < nrows and rows[iend - 1] == rows[iend]:
            iend += 1

        ## for every column apply iqr and percentiles
        for c in numba.prange(partial_vals.shape[1]):
            col_vals = partial_vals[cols[istart:iend], c]
            res_index = output_vals * c

            if np.isnan(col_vals).all():
                result[g, res_index] = np.nan
                result[g, res_index + 1] = np.nan
                result[g, res_index + 2] = np.nan
                continue

            lower, med, higher = np.nanpercentile(col_vals, (5, 50, 95))
            result[g, res_index] = lower
            result[g, res_index + 1] = med
            result[g, res_index + 2] = higher

        # # go to next group
        istart = iend
    return result


def parallel_higher_order_context(df, graph, k, n_splits, output_vals):
    """Calculate higher_order neighbours in chunks and calculate percentiles for values"""
    A = graph.transform("B").sparse
    ids = graph.unique_ids.values
    rows = np.arange(A.shape[0])
    values = df.values

    final_result = pd.DataFrame(
        np.empty((values.shape[0], values.shape[1] * output_vals)), index=ids
    )

    for source in np.array_split(rows, n_splits):
        Q = A[source, :].copy()
        for _ in range(1, k):
            next_step = Q @ A
            Q += next_step

        sparray = Q.tocoo(copy=False)
        sorter = sparray.row.argsort()
        unique_tail = np.unique(sparray.col)
        partial_vals = values[unique_tail, :]

        cols_dict = pd.Series(np.arange(len(unique_tail)), index=unique_tail)
        columns_to_pass = cols_dict.loc[sparray.col].values
        rows_to_pass = sparray.row[sorter]

        partial_res = numba_limit_range(
            rows_to_pass, columns_to_pass, partial_vals, output_vals
        )

        final_result.iloc[source, :] = partial_res

    return final_result




@numba.njit
def _interpolate(weights, group):
    q = (25, 50, 75)
    nan_tracker = np.isnan(group)
    if nan_tracker.all():
        return np.array([float(np.nan) for _ in q])
    group = group[~nan_tracker]
    sorter = np.argsort(group)
    group = group[sorter]
    weights = weights[~nan_tracker][sorter]

    xs = np.cumsum(weights) - 0.5 * weights
    xs = xs / weights.sum()
    ys = group
    interpolate = np.interp(
        [x / 100 for x in q],
        xs,
        ys,
    )
    return interpolate

@numba.njit(parallel=True)
def partial_weighted_percentile(rows, cols, partial_vals, centroids, kernel, bandwidth):
    """rows are the re-mapped focals, cols are re-mapped neighbours"""
    output_vals = 2
    ngroups = len(np.unique(rows))
    nrows = rows.shape[0]
    result = np.empty((ngroups, partial_vals.shape[1] * output_vals))

    istart = 0
    for g in range(ngroups):
        # # find focal start
        # istart = 0
        # while istart < nrows and rows[istart] != g:
        #     istart += 1

        # find neighbors
        iend = istart + 1
        while iend < nrows and rows[iend - 1] == rows[iend]:
            iend += 1

        neighbours = centroids[cols[istart:iend], :]
        focals = centroids[rows[istart:iend], :]
        weights = np.sqrt(((neighbours - focals)**2).sum(axis=1))
        
        not_zero = weights != 0

        if bandwidth == -1:
            bandwidth = np.max(weights)
        elif bandwidth == 0:
            bandwidth = np.mean(weights)
        else:
            bandwidth = bandwidth
            

        if kernel == 'gaussian':
            u = weights / bandwidth
            weights = np.exp(-((u / 2) ** 2)) / (np.sqrt(2) * np.pi)
        elif kernel == 'inverse':
            weights = 1 / weights
        else:
            # default - reverse weights
            weights = 0 - weights
        
        
        ## for every column apply iqr and percentiles
        for c in numba.prange(partial_vals.shape[1]):
            
            col_vals = partial_vals[cols[istart:iend], c]
            res_index = output_vals * c

            if np.isnan(col_vals).all():
                result[g, res_index] = np.nan
                result[g, res_index+1] = np.nan
                continue

            else:
                res = _interpolate(weights[not_zero], col_vals[not_zero])
                result[g, res_index] = res[1]
                result[g, res_index+1] = res[2] - res[0]

        # # go to next group
        istart = iend
    return result


def spatially_weighted_partial_lag(df, graph, centroids, kernel, k, n_splits, bandwidth=-1):
    """Calculate higher_order neighbours in chunks and calculate spatially-weighted percentiles for values
        bandwidth = -1, calculate maximum locally
        bandiwdth == 0, - calculate mean locally
        bandwidth > 0: use as constant value
    """

    A = graph.transform("B").sparse
    ids = graph.unique_ids.values
    rows = np.arange(A.shape[0])
    values = df.values
    
    final_result = pd.DataFrame(
        np.empty((values.shape[0], values.shape[1] * 2)), index=ids
    )
    
    for source in np.array_split(rows, n_splits):
        Q = A[source, :].copy()
        for _ in range(1, k):
            next_step = Q @ A
            Q += next_step
    
        sparray = Q.tocoo(copy=False)
    
        unique_tail = np.unique(sparray.col)
        cols_dict = pd.Series(np.arange(len(unique_tail)), index=unique_tail)
        columns_to_pass = cols_dict.loc[sparray.col].values
        rows_to_pass = cols_dict.loc[source[sparray.row]].values
    
        partial_vals = values[unique_tail, :]
        partial_centroids = centroids[unique_tail, :]
    
        partial_res = partial_weighted_percentile(
                rows_to_pass, columns_to_pass, partial_vals, partial_centroids, kernel, bandwidth
            )
    
        final_result.iloc[source, :] = partial_res

    final_result.columns = np.concatenate([(c + "_median", c + "_iqr") for c in df.columns])
    return final_result