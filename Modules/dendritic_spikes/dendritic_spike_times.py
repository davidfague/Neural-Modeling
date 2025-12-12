import numpy as np
import pandas as pd

def get_dendritic_spike_times(df: pd.DataFrame, seg_id: int, spike_type: str, 
                               start_step: int = None, end_step: int = None):
    """
    Gather dendritic spike start (lower bound) and end (upper bound) times 
    for a given segment ID if they exist.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike data with columns like '{spike_type}_lower_bound'.
    seg_id : int
        Segment ID to filter for.
    spike_type : str
        Spike type ('ca', 'nmda', or 'na').
    start_step : int, optional
        Lower time bound (inclusive). If None, no lower limit.
    end_step : int, optional
        Upper time bound (inclusive). If None, no upper limit.
    
    Returns
    -------
    dict
        Dictionary with keys 'lower_bound' and 'upper_bound', each an array of ints.
    """
    lower_col = f"{spike_type}_lower_bound"
    upper_col = f"{spike_type}_upper_bound"

    seg_df = df[df['segmentID'] == seg_id]

    if lower_col in seg_df.columns:
        lower_times = np.array(seg_df[lower_col].dropna()).astype(int)
    else:
        lower_times = np.array([])
    if upper_col in seg_df.columns:
        upper_times = np.array(seg_df[upper_col].dropna()).astype(int)
    else:
        upper_times = np.array([])

    if start_step is not None and end_step is not None:
        lower_times = lower_times[(lower_times >= start_step) & (lower_times <= end_step)]
        upper_times = upper_times[(upper_times >= start_step) & (upper_times <= end_step)]

    return {"lower_bound": lower_times, "upper_bound": upper_times}
