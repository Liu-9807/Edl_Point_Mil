import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .utils import weight_reduce_loss

EPS = 1e-16



def aggregate_instance_evidence(instance_evidence: torch.Tensor, method: str, weight: torch.Tensor | None = None):
    """
    instance_evidence: [1, N, C] (evidence, not alpha)
    return: bag-level alpha [1, C]
    """
    if method == 'mean':
        mean_evi = instance_evidence.sum(1) / instance_evidence.shape[1]
        return mean_evi + 1
    elif method == 'max':
        temp_alpha = instance_evidence.detach() + 1
        temp_prob = temp_alpha[:, :, [1]] / temp_alpha.sum(-1, keepdim=True)  # 假设正类为索引1
        sel_idx = torch.argmax(temp_prob.detach())
        bag_evi = instance_evidence[:, sel_idx, :]
        return bag_evi + 1
    elif method == 'diweight':
        assert weight is not None and weight.shape[1] == instance_evidence.shape[1]
        if len(weight.shape) > 2 and weight.shape[-1] > 1:
            assert weight.shape[-1] == instance_evidence.shape[-1]
            w = weight / weight.sum(-1, keepdim=True)       # [1, N, C]
            w = w / w.sum(1, keepdim=True)                  # column-normalized
            cur_w = w[:, :, [1]]                            # 取正类权重
        else:
            cur_w = weight
        bag_evi = (cur_w * instance_evidence).sum(1)
        return bag_evi + 1
    else:
        raise NotImplementedError(f"{method} not supported.")

def compute_auxiliary_output(ins_alpha, target, separate='II', aggregate='diweight', weight=None):
    """
    ins_alpha: list of Tensor, each [1, N, C] (alpha)
    target: [B, C] one-hot
    return depends on separate:
      - 'II': (pos_alpha_bag, pos_target_bag, neg_alpha_list, neg_target_list)
      - 'I' : (alpha_bag_all, target_bag_all)
    """
    if separate == 'II':
        num_classes = target.shape[-1]
        neg_idx = torch.nonzero(target[:, 0] == 1).squeeze(-1).tolist()
        pos_idx = torch.nonzero(target[:, 1] == 1).squeeze(-1).tolist()

        # negative: keep instance-level
        if len(neg_idx) == 0:
            neg_alpha, neg_target = None, None
        else:
            neg_alpha = [ins_alpha[i].squeeze(0) for i in neg_idx]  # [N, C]
            neg_target = [torch.zeros(x.shape[0], device=target.device, dtype=target.dtype) for x in neg_alpha]
            neg_target = [F.one_hot(t.long(), num_classes).to(target.dtype) for t in neg_target]

        # positive: aggregate to bag-level
        if len(pos_idx) == 0:
            pos_alpha, pos_target = None, None
        else:
            pos_target = target[pos_idx]  # [B_pos, C]
            pos_alpha = []
            for i in pos_idx:
                ins_evi = ins_alpha[i] - 1
                w = weight[i] if weight is not None else None
                bag_alpha = aggregate_instance_evidence(ins_evi, aggregate, w)  # [1, C]
                pos_alpha.append(bag_alpha)
            pos_alpha = torch.cat(pos_alpha, dim=0)  # [B_pos, C]
        return pos_alpha, pos_target, neg_alpha, neg_target

    elif separate == 'I':
        n_batch, bag_alpha = len(ins_alpha), []
        for i in range(n_batch):
            ins_evi = ins_alpha[i] - 1
            w = weight[i] if weight is not None else None
            bag_alpha.append(aggregate_instance_evidence(ins_evi, aggregate, w))
        bag_alpha = torch.cat(bag_alpha, dim=0)  # [B, C]
        return bag_alpha, target
    else:
        raise NotImplementedError(f"separate={separate} not supported.")

def kl_divergence(alpha, num_classes, device=None):
    """
    Calculates the KL divergence of a Dirichlet distribution for the EDL loss.
    KL(Dir(α)||Dir(1)) where Dir(1) is uniform Dirichlet prior.

    Args:
        alpha (torch.Tensor): The parameters of the Dirichlet distribution, shape [B, num_classes].
        num_classes (int): The number of classes.
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The KL divergence value, shape [B, 1].
    """
    if not device:
        device = alpha.device
    
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    
    kl = first_term + second_term  # [B, 1]
    return kl


def loglikelihood_loss(y, alpha, device=None):
    """
    Calculates the log-likelihood loss component for the EDL MSE loss.
    This includes both the error term and the variance term.

    Args:
        y (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        alpha (torch.Tensor): The parameters of the Dirichlet distribution, shape [B, num_classes].
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The log-likelihood loss, shape [B, 1].
    """
    if not device:
        device = alpha.device
    
    assert y.shape[-1] != 1, "y must be one-hot encoded."
    y = y.to(device)
    alpha = alpha.to(device)
    
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Error term: squared difference between target and predicted probability
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    
    # Variance term: captures prediction uncertainty
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    
    loglikelihood = loglikelihood_err + loglikelihood_var  # [B, 1]
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, use_kl_div=True, device=None):
    """
    Calculates the total MSE loss for EDL, which includes the log-likelihood loss 
    and a KL divergence regularizer.

    Args:
        y (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        alpha (torch.Tensor): The parameters of the Dirichlet distribution, shape [B, num_classes].
        epoch_num (int): The current epoch number, for annealing.
        num_classes (int): The number of classes.
        annealing_step (int): The number of epochs for the annealing coefficient to reach 1.
        use_kl_div (bool): Whether to use KL divergence regularization.
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The total EDL MSE loss, shape [B, 1].
    """
    if not device:
        device = alpha.device
    
    y = y.to(device)
    alpha = alpha.to(device)
    
    # Compute log-likelihood loss
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    # Compute annealing coefficient (gradually increases from 0 to 1)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device)
    )
    
    # KL divergence regularization
    if use_kl_div:
        # For non-target classes, we want alpha -> 1 (no evidence)
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    else:
        kl_div = 0
    
    return loglikelihood + kl_div  # [B, 1]


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, use_kl_div=True, device=None):
    """
    A generic function for calculating the EDL loss (log or digamma version).

    Args:
        func (function): The function to apply to the alpha parameters (torch.log or torch.digamma).
        y (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        alpha (torch.Tensor): The parameters of the Dirichlet distribution, shape [B, num_classes].
        epoch_num (int): The current epoch number, for annealing.
        num_classes (int): The number of classes.
        annealing_step (int): The number of epochs for the annealing coefficient to reach 1.
        use_kl_div (bool): Whether to use KL divergence regularization.
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The EDL loss, shape [B, 1].
    """
    if not device:
        device = alpha.device
    
    y = y.to(device)
    alpha = alpha.to(device)
    
    S = torch.sum(alpha, dim=1, keepdim=True)  # [B, 1]
    
    # Bayes risk term: negative log-likelihood
    A = torch.sum(y * (torch.digamma(S) - func(alpha)), dim=1, keepdim=True)  # [B, 1]

    # Compute annealing coefficient
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device)
    )

    # KL divergence regularization
    if use_kl_div:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    else:
        kl_div = 0

    return A + kl_div  # [B, 1]


def edl_red_loss(target, output_alpha, num_classes, loss_type='log-alpha', device=None):
    """
    RED loss: Regularization for avoiding zero-evidence regions (Pandey et al., ICML 2023).
    This encourages the model to produce evidence for the target class.

    Args:
        target (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        output_alpha (torch.Tensor): The parameters of the Dirichlet distribution, shape [B, num_classes].
        num_classes (int): The number of classes.
        loss_type (str): Type of RED loss ('log-alpha', 'expec-alpha', 'bayes-alpha-digamma', etc.).
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The RED loss, shape [B, 1].
    """
    if not device:
        device = output_alpha.device
    
    target = target.to(device)  # [B, num_classes]
    output_alpha = output_alpha.to(device)  # [B, num_classes]
    
    S = torch.sum(output_alpha, dim=1, keepdim=True)  # [B, 1]
    cor = num_classes / S  # [B, 1], correction factor
    
    # Extract alpha for the target class
    target_alpha = (target * output_alpha).sum(-1, keepdim=True)  # [B, 1]
    
    if loss_type == 'log-alpha':
        # Pandey et al., ICML 2023
        edv_target = target_alpha - 1 + EPS
        loss = -1 * cor * torch.log(edv_target)
    elif loss_type == 'expec-alpha':
        # Expectation-based NLL loss
        loss = -1 * cor * torch.log(target_alpha / S + EPS)
    elif loss_type == 'bayes-alpha-digamma':
        # Bayes risk with digamma
        loss = -1 * cor * (torch.digamma(target_alpha) - torch.digamma(S))
    elif loss_type == 'bayes-alpha-log':
        # Bayes risk with log
        loss = -1 * cor * (torch.log(target_alpha + EPS) - torch.log(S + EPS))
    else:
        raise NotImplementedError(f"RED loss type '{loss_type}' is not implemented.")
    
    return loss  # [B, 1]


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, 
                 use_kl_div=True, red_type=None, device=None):
    """
    A wrapper function that calculates the EDL loss using the MSE formulation.

    Args:
        output (torch.Tensor): The model's raw output (logits), shape [B, num_classes].
        target (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The annealing step.
        use_kl_div (bool): Whether to use KL divergence regularization.
        red_type (str, optional): Type of RED loss to add. If None, no RED loss is used.
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The mean EDL MSE loss (scalar).
    """
    if not device:
        device = alpha.device
   
    # Compute MSE loss
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, use_kl_div, device)
    
    # Add RED loss if specified
    if red_type is not None and len(red_type) > 0:
        loss = loss + edl_red_loss(target, alpha, num_classes, red_type, device)
    
    return torch.mean(loss)


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, 
                 use_kl_div=True, red_type=None, device=None):
    """
    A wrapper function that calculates the EDL loss using the log formulation.

    Args:
        output (torch.Tensor): The model's raw output (logits), shape [B, num_classes].
        target (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The annealing step.
        use_kl_div (bool): Whether to use KL divergence regularization.
        red_type (str, optional): Type of RED loss to add. If None, no RED loss is used.
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The mean EDL log loss (scalar).
    """
    if not device:
        device = alpha.device
    
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, use_kl_div, device)
    
    if red_type is not None and len(red_type) > 0:
        loss = loss + edl_red_loss(target, alpha, num_classes, red_type, device)
    
    return torch.mean(loss)


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, 
                     use_kl_div=True, red_type=None, device=None):
    """
    A wrapper function that calculates the EDL loss using the digamma formulation.

    Args:
        output (torch.Tensor): The model's raw output (logits), shape [B, num_classes].
        target (torch.Tensor): The one-hot encoded ground truth labels, shape [B, num_classes].
        epoch_num (int): The current epoch number.
        num_classes (int): The number of classes.
        annealing_step (int): The annealing step.
        use_kl_div (bool): Whether to use KL divergence regularization.
        red_type (str, optional): Type of RED loss to add. If None, no RED loss is used.
        device (torch.device, optional): The device to run the calculations on.

    Returns:
        torch.Tensor: The mean EDL digamma loss (scalar).
    """
    if not device:
        device = alpha.device
    
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, use_kl_div, device)
    
    if red_type is not None and len(red_type) > 0:
        loss = loss + edl_red_loss(target, alpha, num_classes, red_type, device)
    
    return torch.mean(loss)


@MODELS.register_module()
class EDLLoss(nn.Module):
    """
    Evidential Deep Learning (EDL) loss compatible with MMDetection framework.

    This loss function is used for classification tasks where the model outputs
    the parameters of a Dirichlet distribution, representing the evidence for each class.

    Args:
        loss_type (str): The type of EDL loss to use. Options are 'log', 'digamma', 'mse'.
        loss_weight (float): The weight of the loss.
        annealing_step (int): The number of epochs for the annealing coefficient to reach 1.
        use_kl_div (bool): Whether to use KL divergence regularization. Default: True.
        red_type (str, optional): Type of RED loss regularization. Options include:
            - 'log-alpha': Pandey et al., ICML 2023
            - 'expec-alpha': Expectation-based
            - 'bayes-alpha-digamma': Digamma-based Bayes risk
            - 'bayes-alpha-log': Log-based Bayes risk
            - None: No RED loss
        reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'mean'.
        
    References:
        - Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
        - Pandey et al., "Towards Better Uncertainty Quantification", ICML 2023
    """
    
    def __init__(self,
                 loss_type='mse',
                 loss_weight=1.0,
                 annealing_step=10,
                 use_kl_div=True,
                 red_type=None,
                 reduction='mean',
                 branch='bag',
                 separate='II',
                 aggregate='mean'):
        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.annealing_step = annealing_step
        self.use_kl_div = use_kl_div
        self.red_type = red_type
        self.reduction = reduction
        self.branch = branch          # 'bag' | 'instance'
        self.separate = separate      # 'II' | 'I' | 'F' | 'B'
        self.aggregate = aggregate    # 'mean' | 'max' | 'diweight'
        
        assert loss_type in ['log', 'digamma', 'mse'], \
            f"loss_type must be one of ['log', 'digamma', 'mse'], but got {loss_type}"
        
        if red_type is not None:
            assert red_type in ['log-alpha', 'expec-alpha', 'bayes-alpha-digamma', 'bayes-alpha-log'], \
                f"red_type must be one of ['log-alpha', 'expec-alpha', 'bayes-alpha-digamma', 'bayes-alpha-log'] or None"

    def _select_base_loss(self, func_name):
        if func_name == 'log':
            return edl_log_loss
        elif func_name == 'digamma':
            return edl_digamma_loss
        elif func_name == 'mse':
            return edl_mse_loss
        else:
            raise NotImplementedError

    def forward(self,
                output,
                target,
                epoch_num,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ins_weight=None,
                ins_target=None,
                c_fisher=None):
        """
        output: Tensor [B, C] (bag logits) or list of Tensor [1, N, C] (instance alpha/logits)
        target: Tensor [B, C] one-hot or [B, 1] label
        ins_weight: list of Tensor for diweight aggregation, align with output list
        ins_target: list of Tensor for 'F' fully-supervised instances
        """
        reduction = reduction_override if reduction_override else self.reduction
        loss_fn = self._select_base_loss(self.loss_type)
        num_classes = target.shape[-1] if target.dim() > 1 else None

        # branch: bag (standard) ------------------------------------------------
        if self.branch == 'bag' and isinstance(output, torch.Tensor):
            loss = loss_fn(
                output, target, epoch_num, output.shape[-1],
                self.annealing_step, self.use_kl_div, self.red_type,
                device=output.device
            )

        # branch: instance (MIL) ------------------------------------------------
        elif self.branch == 'instance' and isinstance(output, list):
            # output here应为 alpha（或 logits 已转 alpha）；若是 logits 需先 relu+1
            ins_alpha = []
            for out in output:
                ins_alpha.append(out)

            if self.separate in ['F', 'B']:
                # 逐实例损失：全监督(F) 或 继承袋标签(B)
                n_sample = len(ins_alpha)
                ins_losses = []
                for i in range(n_sample):
                    cur_alpha = ins_alpha[i].squeeze(0)  # [N, C]
                    num_cls_ins = cur_alpha.shape[-1]
                    if self.separate == 'F':
                        assert isinstance(ins_target, list), "`ins_target` needed for separate='F'"
                        cur_t = ins_target[i].squeeze(0).to(cur_alpha.device)
                        cur_t = F.one_hot(cur_t.long(), num_cls_ins).to(cur_alpha.dtype)
                    else:  # 'B'
                        cur_t = target[[i], :].to(cur_alpha.device)
                        cur_t = cur_t * torch.ones((cur_alpha.shape[0], num_cls_ins), device=cur_alpha.device, dtype=cur_alpha.dtype)
                    # 可选正包权重
                    if self.aggregate == 'diweight' and target[i, 1].item() == 1 and ins_weight is not None:
                        w = ins_weight[i].squeeze(0)
                        prob = w / w.sum(-1, keepdim=True)
                        norm_prob = prob / prob.sum(0, keepdim=True)
                        ins_w = norm_prob[:, [1]] * cur_alpha.shape[0]
                    else:
                        ins_w = torch.ones((cur_alpha.shape[0], 1), device=cur_alpha.device)
                    base = edl_loss(torch.digamma if self.loss_type == 'digamma' else torch.log,
                                    cur_t, cur_alpha, epoch_num, num_cls_ins,
                                    self.annealing_step, self.use_kl_div, device=cur_alpha.device) \
                           if self.loss_type in ['log', 'digamma'] else \
                           mse_loss(cur_t, cur_alpha, epoch_num, num_cls_ins,
                                    self.annealing_step, self.use_kl_div, device=cur_alpha.device)
                    cur_loss = base
                    if self.red_type:
                        cur_loss = cur_loss + edl_red_loss(cur_t, cur_alpha, num_cls_ins, self.red_type, cur_alpha.device)
                    ins_losses.append((ins_w * cur_loss).mean())
                loss = sum(ins_losses) / len(ins_losses)

            elif self.separate in ['II', 'I']:
                pos_alpha, pos_target, neg_alpha, neg_target = compute_auxiliary_output(
                    ins_alpha, target, separate=self.separate, aggregate=self.aggregate, weight=ins_weight
                )
                n_sample, pos_loss, neg_loss = 0, 0, 0
                # 正包（袋级聚合）
                if pos_alpha is not None:
                    n_sample += len(pos_alpha)
                    base = mse_loss(pos_target, pos_alpha, epoch_num, pos_alpha.shape[-1],
                                    self.annealing_step, self.use_kl_div, device=pos_alpha.device) \
                           if self.loss_type == 'mse' else \
                           edl_loss(torch.digamma if self.loss_type == 'digamma' else torch.log,
                                    pos_target, pos_alpha, epoch_num, pos_alpha.shape[-1],
                                    self.annealing_step, self.use_kl_div, device=pos_alpha.device)
                    pos_loss = base
                    if self.red_type:
                        pos_loss = pos_loss + edl_red_loss(pos_target, pos_alpha, pos_alpha.shape[-1], self.red_type, pos_alpha.device)
                    pos_loss = pos_loss.sum()

                # 负包（实例级）
                if neg_alpha is not None:
                    n_sample += len(neg_alpha)
                    for i in range(len(neg_alpha)):
                        cur_alpha = neg_alpha[i]
                        cur_t = neg_target[i]
                        base = mse_loss(cur_t, cur_alpha, epoch_num, cur_alpha.shape[-1],
                                        self.annealing_step, self.use_kl_div, device=cur_alpha.device) \
                               if self.loss_type == 'mse' else \
                               edl_loss(torch.digamma if self.loss_type == 'digamma' else torch.log,
                                        cur_t, cur_alpha, epoch_num, cur_alpha.shape[-1],
                                        self.annealing_step, self.use_kl_div, device=cur_alpha.device)
                        cur_loss = base
                        if self.red_type:
                            cur_loss = cur_loss + edl_red_loss(cur_t, cur_alpha, cur_alpha.shape[-1], self.red_type, cur_alpha.device)
                        neg_loss = neg_loss + cur_loss.mean()
                loss = (pos_loss + neg_loss) / max(n_sample, 1)
            else:
                raise NotImplementedError(f"separate={self.separate} not supported.")
        else:
            raise TypeError("output must be Tensor for branch='bag' or list for branch='instance'.")

        # 样本权重 & reduction
        if weight is not None and isinstance(loss, torch.Tensor):
            if weight.dim() == 1:
                weight = weight.unsqueeze(-1)
            loss = loss * weight

        if isinstance(loss, torch.Tensor):
            if reduction == 'mean':
                loss = loss.mean() if avg_factor is None else loss.sum() / avg_factor
            elif reduction == 'sum':
                loss = loss.sum()

        return self.loss_weight * loss
