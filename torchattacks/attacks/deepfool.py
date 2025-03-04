import torch
import torch.nn as nn

from ..attack import Attack
from tqdm import tqdm

class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]

    Distance Measure : L2 Linf

    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, norm='L2', eps=0.1, steps=50, overshoot=0, tqdm=False):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.eps = eps
        self.norm = norm
        self._supported_mode = ['default']
        self.tqdm = tqdm

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        # while (True in correct) and (curr_steps < self.steps):
        if self.tqdm:
            loop_range = tqdm(range(self.steps))
        else:
            loop_range = range(self.steps)
            
        for curr_step in loop_range:
            if not (True in correct):
                break
            for idx in range(batch_size):
                ori_image = images[idx]
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                if (self.norm == "Linf" and torch.norm(ori_image - adv_image, p=float('inf')) > self.eps) or \
                        (self.norm == "L2" and torch.norm(ori_image - adv_image, p=2) > self.eps):
                    correct[idx] = False
                    continue
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        if self.norm == "L2":
            value = torch.abs(f_prime) \
                    / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
            _, hat_L = torch.min(value, 0)

            delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                    / (torch.norm(w_prime[hat_L], p=2)**2))
        elif self.norm == "Linf":
            value = torch.abs(f_prime) \
                    / torch.norm(nn.Flatten()(w_prime), p=1, dim=1)
            _, hat_L = torch.min(value, 0)

            delta = (torch.abs(f_prime[hat_L])*torch.sign(w_prime[hat_L])) \
                    / torch.norm(w_prime[hat_L], p=1)
            
        target_label = hat_L if hat_L < label else hat_L+1

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
