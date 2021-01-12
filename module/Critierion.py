import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function



class BCELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, model_output, data_input):
        masks_probs_flat = model_output.view(-1)
        true_masks_flat = data_input.float().view(-1)
        loss = self.loss_fn(masks_probs_flat, true_masks_flat)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


#
# # https://github.com/pytorch/pytorch/issues/1249
# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size(0)
#     m1 = pred.view(num, -1)  # Flatten
#     m2 = target.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#
#     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


device = 0


class ReconLoss(nn.Module):
    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        if self.masked:
            masks = data_input['masks']
            return self.loss_fn(outputs * (1 - masks), targets * (1 - masks))
        else:
            return self.loss_fn(outputs, targets)


class VGGLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()

    def vgg_loss(self, output, target):
        output_feature = self.vgg(output.repeat(1, 3, 1, 1))
        target_feature = self.vgg(target.repeat(1, 3, 1, 1))
        loss = (
                self.l1_loss(output_feature.relu2_2, target_feature.relu2_2)
                + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3)
                + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        )
        return loss

    def forward(self, targets, outputs):
        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(-1)):
            mean_image_loss.append(
                self.vgg_loss(outputs[..., frame_idx], targets[..., frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class StyleLoss(nn.Module):
    def __init__(self, vgg, original_channel_norm=True):
        super().__init__()
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()
        self.original_channel_norm = original_channel_norm

    # From https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    # Implement "Image Inpainting for Irregular Holes Using Partial Convolutions", Liu et al., 2018
    def style_loss(self, output, target):
        output_features = self.vgg(output.repeat(1, 3, 1, 1))
        target_features = self.vgg(target.repeat(1, 3, 1, 1))
        layers = ['relu2_2', 'relu3_3', 'relu4_3']  # n_channel: 128 (=2 ** 7), 256 (=2 ** 8), 512 (=2 ** 9)
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)  # original design (avoid too small loss)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, targets, outputs):
        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(-1)):
            mean_image_loss.append(
                self.style_loss(outputs[..., frame_idx], targets[..., frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class L1LossMaskedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, masked):
        l1_sum = self.l1(x * masked, y * masked)
        return l1_sum / torch.sum(masked)


class L2LossMaskedMean(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.l2 = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, mask):
        l2_sum = self.l2(x * mask, y * mask)
        return l2_sum / torch.sum(mask)


class CompleteFramesReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        masks = data_input['masks']
        return self.loss_fn(outputs, targets, masks)


# From https://github.com/jxgu1016/Total_Variation_Loss.pytorch
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data_input, model_output):
        # View 3D data as 2D
        outputs = model_output['outputs']
        B, L, C, H, W = outputs.shape
        x = outputs.view([B * L, C, H, W])

        masks = data_input['masks']
        masks = masks.view([B * L, -1])
        mask_areas = masks.sum(dim=1)

        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum(1).sum(1).sum(1)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum(1).sum(1).sum(1)
        return ((h_tv + w_tv) / mask_areas).mean()


# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

        elif type == 'l1':
            self.criterion = nn.L1Loss()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
