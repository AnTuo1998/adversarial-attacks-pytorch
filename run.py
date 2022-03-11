from re import X
import numpy as np
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import MultiLayerPerceptron
from torchattacks import *

torch.set_printoptions(precision=8)
path = "data"
X_test = np.load(os.path.join(path, "X_test.npy"))
Y_test = np.load(os.path.join(path, "Y_test.npy"))
X_test = torch.from_numpy(X_test).double()
Y_test = torch.from_numpy(Y_test)
nTest = len(X_test)
model = MultiLayerPerceptron()


# Y_pred = np.zeros(nTest)
# for i in tqdm(range(nTest)):
#     x, y = X_test[i], Y_test[i]
#     x_adv = attack_method.forward(x, y)
#     Y_pred[i] = model.predict(x_adv)

# print(X_test.shape)
# print(X_test[0].unsqueeze(0).shape)
# X_adv = attack_method.forward(X_test[0].unsqueeze(0), 
#                               Y_test[0].unsqueeze(0))

# Y_pred = model(X_adv).argmax(dim=1)
# print(Y_pred)


# logit = model(X_test[0].unsqueeze(0))
# print(logit)
# # print(F.softmax(logit, dim=1))
# target = torch.zeros_like(logit)
# target[0, Y_test[0]] = 1
# print(torch.sum(F.log_softmax(logit, dim=1) * target))

# X_adv = attack_method.forward(X_test[0].unsqueeze(0),
#                               Y_test[0].unsqueeze(0))


# print(model.fc4.weight.grad)
# logit = model(X_test)
# # print(logit)
# # print(F.softmax(logit, dim=1))
# target = torch.zeros_like(logit)
# target[0, Y_test] = 1
# print(-torch.sum(F.log_softmax(logit, dim=1) * target) / nTest)
# X_adv = attack_method.forward(X_test, Y_test)
# print(model.fc1.weight.shape)

eps = 0.2
# for eps in [0.2, 0.15, 0.1, 0.05]:
#     print("eps=", eps)

attack_method = SparseFool(model=model)
X_adv = attack_method.forward(X_test, Y_test)
Y_pred = model(X_adv).argmax(dim=1)
acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
print("Test accuracy is", acc)


    # attack_method = PGD(model=model, eps=eps)
    # X_adv = attack_method.forward(X_test, Y_test)
    # Y_pred = model(X_adv).argmax(dim=1)
    # acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
    # print("Test accuracy is", acc)


    # attack_method = FAB(model=model, eps=eps)
    # X_adv = attack_method.forward(X_test, Y_test)
    # # print(torch.norm(X_test-X_adv, p=float("inf")))
    # Y_pred = model(X_adv).argmax(dim=1)
    # acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
    # print("Test accuracy is", acc)

    # attack_method = TPGD(model=model, eps=eps)
    # X_adv = attack_method.forward(X_test, Y_test)
    # # print(torch.norm(X_test-X_adv, p=float("inf")))
    # Y_pred = model(X_adv).argmax(dim=1)
    # acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
    # print("Test accuracy is", acc)

    # attack_method = EOTPGD(model=model, eps=eps)
    # X_adv = attack_method.forward(X_test, Y_test)
    # # print(torch.norm(X_test-X_adv, p=float("inf")))
    # Y_pred = model(X_adv).argmax(dim=1)
    # acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
    # print("Test accuracy is", acc)

    # attack_method = Jitter(model=model, eps=eps)
    # X_adv = attack_method.forward(X_test, Y_test)
    # print(torch.norm(X_test-X_adv, p=float("inf")))
    # Y_pred = model(X_adv).argmax(dim=1)
    # acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
    # print("Test accuracy is", acc)




# attack_method = FGSM(model=model, eps=0.1)
# Y_pred = torch.zeros_like(Y_test)
# for i in tqdm(range(nTest)):
#     X_adv = attack_method.forward(X_test[i].unsqueeze(0),
#                                   Y_test[i].unsqueeze(0))
#     Y_pred[i] = model(X_adv).argmax()
# acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
# print("Test accuracy is", acc)




  


#test the accuracy of the mdoel 0.976
# Y_pred = model(X_test).argmax(dim=1)
# acc = torch.sum((Y_pred == Y_test)).item() * 1.0 / nTest
# print("Test accuracy is", acc)
