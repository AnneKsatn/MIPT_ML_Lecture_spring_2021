#Архитектура сети

import torch


class SimpleNLP(torch.nn.Module):

    def __init__(self, layers_list):
        super(SimpleNLP, self).__init__()

        layers = []

        for layer in layers_list:
            layers.append(torch.nn.Linear(layer[0], layer[1]))
            # as parameter
            layers.append(torch.nn.Sigmoid())

        layers.append(torch.nn.Softmax())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    layers_list_example = [(2, 5), (5, 3), (3, 6), (6, 2)]
    example_net = SimpleNLP(layers_list_example)

    print(example_net.net)
    x = torch.rand((1, 2))
    print(x)

    print(example_net.net(x, ))