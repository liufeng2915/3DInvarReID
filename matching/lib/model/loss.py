
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn import init


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)

        return y

class CrossEntropyWithLabelSmooth(nn.Module):
    """ Cross entropy loss with label smoothing regularization.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. In CVPR, 2016.
    Equation:
        y = (1 - epsilon) * y + epsilon / K.

    Args:
        epsilon (float): a hyper-parameter in the above equation.
    """
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        _, num_classes = inputs.size()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)

        # dist.reduce_scatter(grad_out, list(grads))
        # grad_out.div_(dist.get_world_size())

        grad_out[:] = grads[dist.get_rank()]

        return grad_out

'''
class TripletLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normlize
        inputs = F.normalize(inputs, p=2, dim=1)

        # compute distance
        dist = 1 - torch.matmul(inputs, inputs.t()) # values in [0, 2]

        # get positive and negative masks
        targets, gallery_targets = targets.view(-1,1), targets.view(-1,1)
        mask_pos = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_neg = 1 - mask_pos

        # For each anchor, find the hardest positive and negative pairs
        dist_ap, _ = torch.max((dist - mask_neg * 99999999.), dim=1)
        dist_an, _ = torch.min((dist + mask_pos * 99999999.), dim=1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss

'''
class TripletLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """

        n = inputs.size(0)
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss

class RGBLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 50
        self.l1_loss_mean = nn.L1Loss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt[network_object_mask & object_mask]
        rgb_loss = self.l2_loss(rgb_values, rgb_gt) / float(object_mask.shape[0] * object_mask.shape[1])
        return rgb_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0]*object_mask.shape[1])

        return mask_loss

    def forward(self, rgb, esti_rgb, object_mask, network_object_mask, esti_occ, eval_rgb_only=False):

        rgb_loss = self.get_rgb_loss(esti_rgb, rgb, network_object_mask, object_mask)
        if eval_rgb_only:
            return rgb_loss
        mask_loss = self.get_mask_loss(esti_occ, network_object_mask, object_mask)

        return rgb_loss, mask_loss