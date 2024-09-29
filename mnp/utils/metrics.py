import torch
import numpy as np
from typing import Tuple, Union
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             matthews_corrcoef, f1_score, precision_score,
                             balanced_accuracy_score, recall_score, 
                             average_precision_score, roc_auc_score,
                             accuracy_score)
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr
from torch.nn.functional import gaussian_nll_loss



# Metrics
# NOTE The CNP arrays have dimensions (num_functions, num_samples) but these metrics
# functions require dimensions (num_samples, num_fuctions). Therefore, always transpose
# the x and the y before passing them as input to these functions.

def to_numpy_if_tensor(arraylike: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(arraylike, torch.Tensor):
        return arraylike.detach().cpu().numpy()
    elif isinstance(arraylike, np.ndarray):
        return arraylike
    else:
        print('Input should be either torch tensor or numpy array.')


def compute_mae(y, y_hat):
    y = np.squeeze(to_numpy_if_tensor(y))
    y_hat = np.squeeze(to_numpy_if_tensor(y_hat))
    mae = mean_absolute_error(y, y_hat, multioutput='raw_values')
    # If a single function, we want to return a float
    if mae.squeeze().ndim == 0:
        mae = float(mae)
    return mae


def compute_mse(y, y_hat):
    y = np.squeeze(to_numpy_if_tensor(y))
    y_hat = np.squeeze(to_numpy_if_tensor(y_hat))
    # If y and y_hat are a single element with no shape, then the function
    # mean_squared_error will throw an error
    if y.shape == () and y_hat.shape == ():
        y = [y]
        y_hat = [y_hat]
    mse = mean_squared_error(y, y_hat, multioutput='raw_values')
    # If a single function, we want to return a float
    if mse.squeeze().ndim == 0:
        mse = float(mse)
    return mse


def compute_r2(y, y_hat):
    y = np.squeeze(to_numpy_if_tensor(y))
    y_hat = np.squeeze(to_numpy_if_tensor(y_hat))
    r2 = r2_score(y, y_hat, multioutput='raw_values')
    # If a single function and a single task, we want to return a float
    if r2.squeeze().ndim == 0:
        r2 = float(r2)
    return r2


def compute_nlpd(y: torch.Tensor, y_hat_mean: torch.Tensor,
                 y_hat_var: torch.Tensor) -> torch.Tensor:
    y = torch.squeeze(y)
    y_hat_mean = torch.squeeze(y_hat_mean)
    y_hat_var = torch.squeeze(y_hat_var)
    nlpd = gaussian_nll_loss(y_hat_mean.float(), y.float(),
                             y_hat_var.float()).item()
    return nlpd


def compute_logprob(y: torch.Tensor, y_hat_mean: torch.Tensor,
                 y_hat_var: torch.Tensor) -> torch.Tensor:
    return -1 * compute_nlpd(y=y, y_hat_mean=y_hat_mean, y_hat_var=y_hat_var)


def compute_phi(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor],
                threshold: float=0.5) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    y_hat = (p_hat > threshold).astype(int)
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = matthews_corrcoef(y[i], y_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return matthews_corrcoef(y, y_hat)


def compute_f1(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor],
                threshold: float=0.5) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    y_hat = (p_hat > threshold).astype(int)
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = f1_score(y[i], y_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return f1_score(y, y_hat)


def compute_precision(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor],
                threshold: float=0.5) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    y_hat = (p_hat > threshold).astype(int)
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = precision_score(y[i], y_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return precision_score(y, y_hat)


def compute_recall(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor],
                threshold: float=0.5) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    y_hat = (p_hat > threshold).astype(int)
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = recall_score(y[i], y_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return recall_score(y, y_hat)


def compute_balanced_accuracy(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor],
                threshold: float=0.5) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    y_hat = (p_hat > threshold).astype(int)
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = balanced_accuracy_score(y[i], y_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return balanced_accuracy_score(y, y_hat)


def compute_average_precision(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor]) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = average_precision_score(y[i], p_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return average_precision_score(y, p_hat)


def compute_roc_auc(y: Union[np.ndarray, torch.Tensor],
                p_hat: Union[np.ndarray, torch.Tensor]) -> float:
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    # if the label array is 2d, we assume each row corresponds to a function
    if len(y.shape) == 2:
        num_rows = y.shape[0]
        result = np.zeros(num_rows)
        for i in range(num_rows):
            result[i] = roc_auc_score(y[i], p_hat[i])
        return result
    # if the label array is 1d, it corresponds to a single function
    else:
        return roc_auc_score(y, p_hat)


def compute_confident_accuracy(y: Union[np.ndarray,torch.Tensor],
                               p_hat: Union[np.ndarray, torch.Tensor],
                               threshold: float=0.5,
                               num_chunks: int=10, verbose: bool=False
                              ) -> Tuple[torch.Tensor]:
    """
    Originally the number of chunks was always 100, hence the name percentiles.
    """
    y = torch.squeeze(y)
    p_hat = torch.squeeze(p_hat)
    # separate 0s and 1s
    mask_pred_0 = p_hat < threshold
    mask_pred_1 = p_hat >= threshold
    y_0 = y[mask_pred_0]
    p_0 = p_hat[mask_pred_0]
    y_1 = y[mask_pred_1]
    p_1 = p_hat[mask_pred_1]
    num_true_1 = sum(y).int()
    num_true_0 = len(y) - num_true_1
    if num_true_0 < num_chunks or num_true_1 < num_chunks:
        raise NotImplementedError
    num_pred_0 = sum(mask_pred_0).int()
    num_pred_1 = sum(mask_pred_1).int()
    # rank predictions within 0s and within 1s
    confidence_rank_0 = torch.argsort(p_0, descending=False)
    y_ranked_0 = y_0[confidence_rank_0] # should be all 0s so no need to rank them
    p_ranked_0 = p_0[confidence_rank_0]
    confidence_rank_1 = torch.argsort(p_1, descending=True)
    y_ranked_1 = y_1[confidence_rank_1] # should be all 1s so no need to rank them
    p_ranked_1 = p_1[confidence_rank_1]
    y_hat_ranked_0 = (p_ranked_0 > threshold).int()
    y_hat_ranked_1 = (p_ranked_1 > threshold).int()
    # compute accuracy within each percentile
    percentile_idx_0 = torch.tensor_split(torch.arange(num_pred_0), num_chunks)
    percentile_idx_1 = torch.tensor_split(torch.arange(num_pred_1), num_chunks)
    percentile_accuracy_0 = torch.zeros(num_chunks)
    percentile_accuracy_1 = torch.zeros(num_chunks)

    for i, each_percentile_idx in enumerate(percentile_idx_0):
        this_y = y_ranked_0[each_percentile_idx].detach().cpu().numpy()
        this_y_hat = y_hat_ranked_0[each_percentile_idx].detach().cpu().numpy()
        percentile_accuracy_0[i] = balanced_accuracy_score(this_y, this_y_hat)

    for i, each_percentile_idx in enumerate(percentile_idx_1):
        this_y = y_ranked_1[each_percentile_idx].detach().cpu().numpy()
        this_y_hat = y_hat_ranked_1[each_percentile_idx].detach().cpu().numpy()
        percentile_accuracy_1[i] = balanced_accuracy_score(this_y, this_y_hat)

    return percentile_accuracy_0, percentile_accuracy_1     


def compute_calibration_curve(y: Union[np.ndarray,torch.Tensor],
                              p_hat: Union[np.ndarray, torch.Tensor]):
    y = np.squeeze(to_numpy_if_tensor(y))
    p_hat = np.squeeze(to_numpy_if_tensor(p_hat))
    true_prob, pred_prob = calibration_curve(y, p_hat)
    return true_prob, pred_prob


def compute_calibration_percentiles(y: torch.Tensor, y_hat_mean: torch.Tensor,
                            y_hat_var: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Split predictions into variance (i.e. confidence) percentile buckets and
    calculate the average variance and the average MSE in each bucket.

    Only calculate if num of datapoints is >= 200 (otherwise too small to
    split in 100 buckets, bec).
    """
    y = torch.squeeze(y)
    num_datapoints = len(y)
    if num_datapoints >= 100:
        y_hat_mean = torch.squeeze(y_hat_mean)
        y_hat_var = torch.squeeze(y_hat_var)
        _, ranked_variance_index = y_hat_var.topk(num_datapoints, largest=False)
        confidence_percentiles = torch.tensor_split(ranked_variance_index, 100)
        percentile_var = torch.zeros(100)
        percentile_mse = torch.zeros(100)
        for i, each_percentile in enumerate(confidence_percentiles):
            percentile_var[i] = y_hat_var[each_percentile].mean()
            percentile_mse[i] = compute_mse(y[each_percentile],
                                            y_hat_mean[each_percentile])
    else:
        percentile_var, percentile_mse = None, None
    return percentile_var, percentile_mse


def compute_pearson_var_mse(percentile_var: torch.Tensor, percentile_mse: torch.Tensor) -> float:
    """
    Estimate the NP's uncertainty calibration as the Pearson's correlation between
    the predicted variance and the MSE.
    """
    percentile_var = percentile_var.detach().cpu().numpy()
    percentile_mse = percentile_mse.detach().cpu().numpy()
    pearson, _ = pearsonr(percentile_var, percentile_mse)
    return pearson


def rank_by_prediction(y: np.ndarray,
                       y_hat_mean: np.ndarray) -> Tuple[np.ndarray]:
    y = y.squeeze()
    y_hat_mean = y_hat_mean.squeeze()
    ranking = np.argsort(y_hat_mean)[::-1]
    y_ranked = y[ranking]
    y_hat_mean_ranked = y_hat_mean[ranking]
    return y_ranked, y_hat_mean_ranked


def count_num_actives_selected(y: torch.Tensor, y_hat_mean: torch.Tensor,
                         num_selected: int=100, threshold: float=0.8) -> int:
    """
    Metric that counts the number of actives in a top-ranking selected subset.

    Args:
        y: non-binary antibiotic activity labels (to be binarized).
        hat_y: activity predictions.
        num_selected: size of top-ranking selected subset.
        threshold: threshold to binarize labels y.

    Returns:
        Integer that represents the number of actives in the selected subset.
    """
    y = np.squeeze(to_numpy_if_tensor(y))
    y_hat_mean = np.squeeze(to_numpy_if_tensor(y_hat_mean))
    y_ranked, y_hat_mean_ranked = rank_by_prediction(y, y_hat_mean)
    y_ranked_binary = (y_ranked >= threshold).astype(int)
    num_actives = sum(y_ranked_binary[:num_selected])
    return num_actives


def compute_enrichment_factor(y: torch.Tensor, y_hat_mean: torch.Tensor,
                              num_selected: int=100,
                              threshold: float=0.8) -> float:
    """
    Metric that computes the enrichment factor.

    Args:
        y (numpy array): non-binary antibiotic activity labels (to be binarized).
        hat_y (numpy array): activity predictions.
        num_selected (int, optional): size of top-ranking selected subset.
        threshold (float, optional): threshold to binarize labels y.

    Returns:
        Integer that represents the number of actives in the selected subset.
    """
    y = np.squeeze(to_numpy_if_tensor(y))
    y_hat_mean = np.squeeze(to_numpy_if_tensor(y_hat_mean))
    # Get number of actives selected
    n_actives_selected = count_num_actives_selected(y, y_hat_mean,
                                              num_selected=num_selected,
                                              threshold=threshold)
    # Compute EF
    rate_before = (y >= threshold).astype(int).sum() / len(y)
    rate_after = n_actives_selected / num_selected
    return rate_after / rate_before