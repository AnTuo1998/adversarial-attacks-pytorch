import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.1):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss(reduction="sum")

        images.requires_grad = True
        outputs = self.model(images)

        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        # cost.backward()
        # grad = images.grad.clone().detach()
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images


# def fgsm(model, x, y, eps= 0.1):
#     loss = nn.CrossEntropyLoss()
#     x = x.clone().detach()
#     x.requires_grad = True
#     outputs = model(x)
#     cost = loss(outputs, y)
#     cost.backward()
#     grad = x.grad.clone().detach()
#     x_adv = x + eps * torch.sign(grad)
#     return x_adv


# Y_pred = torch.zeros_like(Y_test)
# for i in tqdm(range(nTest)):
#     X_adv = fgsm(model, X_test[i].unsqueeze(0), Y_test[i].unsqueeze(0))
#     Y_pred[i] = model(X_adv).argmax()
# acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
# print("Test accuracy is", acc)
