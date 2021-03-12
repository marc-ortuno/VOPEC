import numpy as np

def recurrence_histogram(time_series, radius, max_distance):
    """Compute the recurrence histogram from a waveform, as per
    http://www.biomedical-engineering-online.com/content/6/1/23

    Args:
        time_series (1D array-like): time series over which the
            recurrence histogram is calculated.
        radius (float): The radius of the recurrence ball
        max_distance (int): the maximum distance between two samples 
            in the calculation of the recurrence histogram

    Returns:
        np.array, []
    """
    distances = np.zeros(len(time_series))
    # For every sample in the time_series, we start moving "to the right" and
    # we find the index of the first sample which is more than max_distance away 
    # from the given sample. This cannot exceed "max_distance".
    # Then, we find the first sample which is less than max_distance away from
    # the original sample
    for i, sample in enumerate(time_series):
        # for safety, if no sample is found.
        out_idx = len(time_series)
        for j, other_sample in enumerate(time_series[i + 1:]):
            # other_sample == time_series[other_sample_index]
            other_sample_index = i + 1 + j
            # stop at max distance
            if j + 1 > max_distance:
                break
            # compute distance and see if we have exceded the maximum distance.
            d = np.abs(sample - other_sample)
            if d > radius:
                out_idx = other_sample_index
                break

        for k, third_sample in enumerate(time_series[out_idx + 1:]):
            third_sample_idx = out_idx + 1 + k
            # stop at max distance
            if k + 1 > max_distance:
                break
            # compute distance and see if we have stopped exceeding the maximum distance.
            d = np.abs(sample - third_sample)
            # increment the (k + 1)'st entry, since that's the sample distance between 
            # sample and third_sample.
            if d < radius:
                distances[k + 1] += 1
                break
    return distances

def their_recurrence_histogram(ts: np.ndarray, epsilon: float, t_max: int):
    """Numba implementation of the recurrence histogram described in
    http://www.biomedical-engineering-online.com/content/6/1/23
    Parameters
    ----------
    ts: np.ndarray
        Time series of dimension (T,N)
    epsilon: float:
        Recurrence ball radius
    t_max: int
        Maximum distance for return.
        Larger distances are not recorded. Set to -1 for infinite distance.
    Returns
    -------
    recurrence_histogram: np.ndarray
        Histogram of return distances
    """
    return_distances = np.zeros(len(ts), dtype=np.int32)
    for i in np.arange(len(ts)):
        # finding the first "out of ball" index
        first_out = len(ts)  # security
        for j in np.arange(i + 1, len(ts)):
            if 0 < t_max < j - i:
                break
            d = np.linalg.norm(ts[i] - ts[j])
            if d > epsilon:
                first_out = j
                break

        # finding the first "back to the ball" index
        for j in np.arange(first_out + 1, len(ts)):
            if 0 < t_max < j - i:
                break
            d = np.linalg.norm(ts[i] - ts[j])
            if d < epsilon:
                return_distances[j - i] += 1
                break
    return return_distances


def embed_time_series(data: np.ndarray, dim: int, tau: int):
    """Embeds the time series, tested against the pyunicorn implementation
    of that transform"""
    embed_points_nb = data.shape[0] - (dim - 1) * tau
    # embed_mask is of shape (embed_pts, dim)
    embed_mask = np.arange(embed_points_nb)[:, np.newaxis] + np.arange(dim)
    tau_offsets = np.arange(dim) * (tau - 1)  # shape (dim,1)
    embed_mask = embed_mask + tau_offsets
    return data[embed_mask]


# def rpde(time_series: np.ndarray,
#          dim: int = 4,
#          tau: int = 35,
#          epsilon: float = 0.12,
#          tmax: Optional[int] = None,
#          parallel: bool = True) -> Tuple[float, np.ndarray]:
#     """
#     Parameters
#     ----------
#     time_series: np.ndarray
#         The input time series. Has to be float32, normalized to [-1,1]
#     dim: int
#         The dimension of the time series embeddings.
#         Defaults to 4
#     tau: int
#         The "stride" between each of the embedding points in a time series'
#         embedding vector. Should be adjusted depending on the
#         sampling rate of your input data.
#         Defaults to 35.
#     epsilon: float
#         The size of the unit ball described in the RPDE algorithm.
#         Defaults to 0.12.
#     tmax: int, optional
#         Maximum return distance (n1-n0), return distances higher than this
#         are ignored. If set, can greatly improve the speed of the distance
#         histogram computation (especially if your input time series has a lot of points).
#         Defaults to None.
#     parallel: boolean, optional
#         Use the parallelized Numba implementation. The parallelization overhead
#         might make this slower in cases where the time series is very short.
#         Defaults to True.
#     Returns
#     -------
#     rpde: float
#         Value of the RPDE
#     histogram: np.ndarray
#         1-dimensional array corresponding to the histogram of the return
#         distances
#     """

#     # (RPDE expects the array to be floats in [-1,1]
#     if not time_series.dtype == np.float32:
#         raise ValueError("Time series should be float32")

#     if not (np.abs(time_series) <= 1.0).all():
#         raise ValueError("The time series' values have to be normalized "
#                          "to [-1, 1]")

#     #  building the time series, and computing the recurrence histogram
#     embedded_ts = embed_time_series(time_series, dim, tau)

#     # "flattening" all dimensions but the first (the number of embedding vectors)
#     # this doesn't change the distances computed in the recurrence histogram,
#     # as it just changes the way the embedding vector are presented, not their values
#     if len(embedded_ts.shape) > 2:
#         embedded_ts = embedded_ts.reshape(embedded_ts.shape[0], -1)

#     if parallel:
#         histogram_fn = parallel_recurrence_histogram
#     else:
#         histogram_fn = recurrence_histogram

#     if tmax is None:
#         rec_histogram = histogram_fn(embedded_ts, epsilon, -1)
#         # tmax is the highest non-zero value in the histogram
#         if rec_histogram.max() == 0.0:
#             tmax_idx = 0
#         else:
#             tmax_idx = np.argwhere(rec_histogram != 0).flatten()[-1]
#     else:
#         rec_histogram = histogram_fn(embedded_ts, epsilon, tmax)
#         tmax_idx = tmax
#     # culling the histogram at the tmax index
#     culled_histogram = rec_histogram[:tmax_idx]

#     # normalizing the histogram (to make it a probability)
#     # and computing its entropy
#     if culled_histogram.sum():
#         normalized_histogram = culled_histogram / culled_histogram.sum()
#         histogram_entropy = entropy(normalized_histogram)
#         return histogram_entropy / np.log(culled_histogram.size), culled_histogram
#     else:
#         return 0.0, culled_histogram