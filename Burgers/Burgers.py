"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import deepxde as dde
import numpy as np
import torch

from FFnet import FFnet
from torchsummary import summary

def gen_testdata():
    data = np.load("../dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
)

# Use FFnet instead of FNN for comparison
x_train, _ = data.train_x, data.train_y

# Analyze training data uniqueness
print(f"Training data shape: {x_train.shape}")
print(f"Total training points: {x_train.shape[0]}")

# Check unique values in each dimension
x_unique = np.unique(x_train[:, 0])
t_unique = np.unique(x_train[:, 1])
print(f"Unique x values: {len(x_unique)} (range: [{x_unique.min():.4f}, {x_unique.max():.4f}])")
print(f"Unique t values: {len(t_unique)} (range: [{t_unique.min():.4f}, {t_unique.max():.4f}])")

if __name__ == "__main__":
    # Check unique (x,t) pairs
    unique_pairs, counts = np.unique(x_train, axis=0, return_counts=True)
    net = FFnet(
        layer_size=[2] + [20] * 3 + [1],
        activation="tanh",
        initializer="Glorot normal",
        ff_num=10,
        ff_radius=0,
        ff_intensity=0.9,
        init_data=torch.tensor(x_train),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(net.to(device), input_size=(2,))

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    model.train(iterations=10000, batch_size=151, display_every=1000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    # """Backend supported: pytorch"""
    # # Run NNCG after Adam and L-BFGS
    # dde.optimizers.set_NNCG_options(rank=50, mu=1e-1)
    # model.compile("NNCG")
    # losshistory, train_state = model.train(iterations=1000, display_every=100)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    X, y_true = gen_testdata()
    y_pred = model.predict(X)
    f = model.predict(X, operator=pde)
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
    # np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
