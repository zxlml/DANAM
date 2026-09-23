# coding=utf-8
"""DANAM model components (PyTorch implementation).

Implements the additive hypothesis space of the paper:
    f(x) = sum_{j=1}^{p} f_j(x_j),  f_j in H_j          (Eq. 6)
where each f_j is a dedicated small neural network (subnetwork) that
takes a single feature as input. Each subnetwork corresponds to one
group of parameters theta_j used by the group-sparsity regularizer.
"""

import numpy as np
import torch
import torch.nn as nn

TfInput = None  # kept for backward compatibility of imports


def exu(x, weight, bias):
    """ExU hidden unit: exp(w) * (x - b)."""
    return torch.exp(weight) * (x - bias)


def relu(x, weight, bias):
    """Standard hidden unit with ReLU activation."""
    return torch.relu(weight * (x - bias))


def relu_n(x, n=1.0):
    """ReLU activation clipped at n (ReLU-n)."""
    return torch.clamp(x, 0.0, n)


class ActivationLayer(nn.Module):
    """Custom activation layer to support ExU hidden units."""

    def __init__(self, num_units, activation='exu', trainable=True):
        super().__init__()
        self.num_units = num_units
        if activation == 'relu':
            self._activation = relu
            beta_init_mean = 0.0
        elif activation == 'exu':
            self._activation = lambda x, weight, bias: relu_n(exu(x, weight, bias))
            beta_init_mean = 4.0
        else:
            raise ValueError('{} is not a valid activation'.format(activation))
        # beta: (1, num_units); c: (1, num_units) centering parameters
        self._beta = nn.Parameter(torch.empty(1, num_units))
        self._c = nn.Parameter(torch.empty(1, num_units))
        self._beta_init_mean = beta_init_mean
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.5
        nn.init.trunc_normal_(self._beta, mean=self._beta_init_mean, std=std,
                              a=self._beta_init_mean - 2 * std,
                              b=self._beta_init_mean + 2 * std)
        nn.init.trunc_normal_(self._c, mean=0.0, std=0.5, a=-1.0, b=1.0)

    def forward(self, x):
        """x: (batch, 1) -> activations: (batch, num_units)."""
        # Clamp beta for numerical stability of exp()
        beta = torch.clamp(self._beta, -4.0, 4.0)
        return self._activation(x, beta, self._c)


class FeatureNN(nn.Module):
    """Neural network for a single feature (one parameter group theta_j)."""

    def __init__(self, num_units, dropout=0.1, trainable=True, shallow=True,
                 feature_num=0, activation='exu'):
        super().__init__()
        self._num_units = num_units
        self._dropout = dropout
        self._shallow = shallow
        self._feature_num = feature_num
        self.hidden_layers = nn.ModuleList([
            ActivationLayer(num_units, activation=activation)
        ])
        if not shallow:
            self.hidden_layers.append(nn.Linear(num_units, 64))
            self.hidden_layers.append(nn.Linear(64, 32))
        self.linear = nn.Linear(num_units if shallow else 32, 1, bias=False)
        # Zero-init the output layer so that the initial model output is 0
        # for every sample. With the distribution-aware (EML/EDF) loss this
        # is important: a large random constant offset at initialization
        # pre-favors one class and the asymmetric confidence weighting then
        # amplifies it (rich-get-richer collapse).
        nn.init.zeros_(self.linear.weight)
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x, training=True):
        """x: (batch, 1) -> contribution f_j(x_j): (batch,)."""
        h = x
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if not self._shallow and i < len(self.hidden_layers) - 1:
                h = torch.relu(h)
            h = nn.functional.dropout(h, p=self._dropout, training=training)
        return self.linear(h).squeeze(-1)


class NAM(nn.Module):
    """Neural Additive Model / DANAM backbone (Eq. 6)."""

    def __init__(self, num_inputs, num_units, trainable=True, shallow=True,
                 feature_dropout=0.0, dropout=0.1, **kwargs):
        super().__init__()
        self._num_inputs = num_inputs
        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        else:
            self._num_units = [num_units for _ in range(num_inputs)]
        self._shallow = shallow
        self._feature_dropout = feature_dropout
        self._dropout = dropout
        self._kwargs = kwargs
        self.feature_nns = nn.ModuleList([
            FeatureNN(num_units=self._num_units[i],
                      dropout=dropout,
                      trainable=trainable,
                      shallow=shallow,
                      feature_num=i,
                      **kwargs)
            for i in range(num_inputs)
        ])
        self._bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, training=True):
        """x: (batch, p) -> logits (batch,)."""
        individual_outputs = self.calc_outputs(x, training=training)
        stacked_out = torch.stack(individual_outputs, dim=-1)  # (batch, p)
        if self._feature_dropout > 0.0:
            stacked_out = nn.functional.dropout(stacked_out,
                                                p=self._feature_dropout,
                                                training=training)
        return stacked_out.sum(dim=-1) + self._bias

    def calc_outputs(self, x, training=True):
        """Per-feature contributions f_j(x_j)."""
        return [self.feature_nns[j](x[:, j:j + 1], training=training)
                for j in range(self._num_inputs)]

    def feature_param_groups(self):
        """Returns one parameter list per subnetwork (group theta_j).

        The global bias is not part of any group and is excluded from the
        group-sparsity penalty.
        """
        return [list(fnn.parameters()) for fnn in self.feature_nns]


class DNN(nn.Module):
    """Deep Neural Network baseline with 10 hidden layers."""

    def __init__(self, trainable=True, dropout=0.15):
        super().__init__()
        self._dropout = dropout
        layers = []
        for i in range(10):
            layers += [nn.Linear(100, 100) if i else nn.LazyLinear(100),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.hidden_layers = nn.Sequential(*layers)
        self.linear = nn.LazyLinear(1)

    def forward(self, x, training=True):
        h = self.hidden_layers(x)
        return self.linear(h).squeeze(-1)
