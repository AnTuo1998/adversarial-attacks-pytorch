import torch
import torch.nn as nn

from ..attack import Attack


class NIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)

    Shape:
        - x: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        >>> x_adv = attack(x, labels)

    """

    def __init__(self, model, eps=8/255, steps=10, alpha=2/255, decay=1.0):
        super().__init__("NIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = eps / steps
        self._supported_mode = ['default', 'targeted']

    def forward(self, x, labels):
        r"""
        Overridden.
        """
        x = x.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(x, labels)

        momentum = torch.zeros_like(x).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        x_adv = x.clone().detach()

        for _ in range(self.steps):
            x_nes = x_adv + self.alpha * self.decay * momentum

            x_nes.requires_grad = True
            outputs = self.model(x_nes)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, x_nes,
                                       retain_graph=False, create_graph=False)[0]
            grad = grad / torch.norm(grad, p=1, dim=1, keepdim=True)
            # grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)

            grad = grad + momentum * self.decay
            momentum = grad

            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(x_adv - x,
                                min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, min=0, max=1).detach()

        return x_adv
