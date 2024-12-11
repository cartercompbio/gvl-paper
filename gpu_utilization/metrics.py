import torch
from bpnetlite.losses import MNLLLoss, log1pMSELoss
from bpnetlite.performance import calculate_performance_measures


def bpnetlite_loss(outputs_dict, targets_dict, alpha=1):
    y_profile = outputs_dict["profile"]
    y_counts = outputs_dict["counts"].reshape(-1, 1)
    y = targets_dict["track"]
    y_profile = y_profile.reshape(y_profile.shape[0], -1)
    y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
    y = y.reshape(y.shape[0], -1)
    profile_loss = MNLLLoss(y_profile, y)
    count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 1))
    loss = profile_loss + alpha * count_loss
    return {
        "loss": loss,
        "profile_loss": profile_loss,
        "count_loss": count_loss,
    }


def bpnetlite_metrics(outputs_dict, targets_dict, alpha=1):
    # (b l) -> (b 1 l)
    y_profile = outputs_dict["profile"]
    # (b) -> (b 1)
    y_counts = outputs_dict["counts"]
    # (b l) -> (b 1 l)
    y = targets_dict["track"]
    z = y_profile.shape
    y_profile = y_profile.reshape(y_profile.shape[0], -1)
    y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
    y_profile = y_profile.reshape(*z)
    measures = calculate_performance_measures(
        y_profile,
        y,
        y_counts,
        kernel_sigma=7,
        kernel_width=81,
        measures=["profile_mnll", "profile_pearson", "count_mse", "count_pearson"],
    )
    profile_mnll = measures["profile_mnll"]
    count_mse = measures["count_mse"]
    profile_corr = measures["profile_pearson"]
    count_corr = measures["count_pearson"]
    loss = measures["profile_mnll"] + alpha * measures["count_mse"]
    metrics_dict = {
        "profile_mnll": profile_mnll,
        "count_mse": count_mse,
        "profile_corr": profile_corr,
        "count_corr": count_corr,
        "loss": loss,
    }
    return metrics_dict
