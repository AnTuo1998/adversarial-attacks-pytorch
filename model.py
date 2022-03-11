import numpy as np
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiLayerPerceptron(nn.Module):
    def __init__(self, load = True):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(784, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        
        if load:
            self.load_param()

    def load_param(self, path="data"):
        
        params = {}
        param_names = ["fc1.weight", "fc1.bias",
                    "fc2.weight", "fc2.bias",
                    "fc3.weight", "fc3.bias",
                    "fc4.weight", "fc4.bias"]

        for name in param_names:
            params[name] = np.load(os.path.join(path, 
                                                f"{name}.npy")).T
            # print(name, params[name].shape)

        with torch.no_grad():
            # self.fc1.weight.copy_(torch.from_numpy(
            #     params["fc1.weight"]).double())
            # self.fc1.bias.copy_(torch.from_numpy(
            #     params["fc1.bias"]).double())
            self.fc1.weight = nn.Parameter(
                torch.from_numpy(params["fc1.weight"]).double())
            self.fc1.bias = nn.Parameter(
                torch.from_numpy(params["fc1.bias"]).double())
            self.fc2.weight = nn.Parameter(
                torch.from_numpy(params["fc2.weight"]).double())
            self.fc2.bias = nn.Parameter(
                torch.from_numpy(params["fc2.bias"]).double())
            self.fc3.weight = nn.Parameter(
                torch.from_numpy(params["fc3.weight"]).double())
            self.fc3.bias = nn.Parameter(
                torch.from_numpy(params["fc3.bias"]).double())
            self.fc4.weight = nn.Parameter(
                torch.from_numpy(params["fc4.weight"]).double())
            self.fc4.bias = nn.Parameter(
                torch.from_numpy(params["fc4.bias"]).double())
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.softmax(self.fc4(x), dim=1)
        x = self.fc4(x)
        return x
            
    # def load_param(self):
    #     '''
    #     This method loads the weights and biases of a trained model.
    #     '''
        
    #     params = {}
    #     param_names = ["fc1.weight", "fc1.bias",
    #                    "fc2.weight", "fc2.bias",
    #                    "fc3.weight", "fc3.bias",
    #                    "fc4.weight", "fc4.bias"]

    #     for name in param_names:
    #         params[name] = np.load(os.path.join("data",
    #                                             f"{name}.npy"))
    #     # print(params["fc1.weight"][0,0])
        
    #     self.W1 = torch.from_numpy(params["fc1.weight"]).double()
    #     self.b1 = torch.from_numpy(params["fc1.bias"]).double()
    #     self.W2 = torch.from_numpy(params["fc2.weight"]).double()
    #     self.b2 = torch.from_numpy(params["fc2.bias"]).double()
    #     self.W3 = torch.from_numpy(params["fc3.weight"]).double()
    #     self.b3 = torch.from_numpy(params["fc3.bias"]).double()
    #     self.W4 = torch.from_numpy(params["fc4.weight"]).double()
    #     self.b4 = torch.from_numpy(params["fc4.bias"]).double()

    # # def gradient(self, x, y):
    # #     grad = F.softmax(self.z4, dim = 1)
    # #     grad[:, y] -= 1
    # #     # print(self.p)
    # #     # print(grad)
    # #     grad = grad @ self.W4.T
    # #     grad = grad @ torch.diag((self.h3 > 0).squeeze().double()) @ self.W3.T
    # #     grad = grad @ torch.diag((self.h2 > 0).squeeze().double()) @ self.W2.T
    # #     grad = grad @ torch.diag((self.h1 > 0).squeeze().double()) @ self.W1.T
    # #     return grad
        
    # def forward(self, x):
    #     '''
    #     This method finds the predicted probability vector of an input
    #     image x.
        
    #     Input
    #         x: a single image vector in ndarray format
    #     Ouput
    #         a vector in ndarray format representing the predicted class
    #         probability of x.
            
    #     Intermediate results are stored as class attributes.
    #     You might need them for gradient computation.
    #     '''
    #     W1, W2, W3, W4 = self.W1, self.W2, self.W3, self.W4
    #     b1, b2, b3, b4 = self.b1, self.b2, self.b3, self.b4

    #     self.z1 = torch.matmul(x, W1)+b1
    #     self.h1 = F.relu(self.z1)
    #     self.z2 = torch.matmul(self.h1, W2)+b2
    #     self.h2 = F.relu(self.z2)
    #     self.z3 = torch.matmul(self.h2, W3)+b3
    #     self.h3 = F.relu(self.z3)
    #     self.z4 = torch.matmul(self.h3, W4)+b4
    #     # self.p = softmax(self.z4)

    #     return self.z4
    

        
            
# if __name__ == "__main__":
#     mlp = MultiLayerPerceptron(load=False)
    # print(mlp)
    # for param in mlp.parameters():
    #     print(param.size())
    # data = torch.randn((2, 784))
    # print(mlp(data))
    # mlp.load_param()
    # print(mlp(data))
