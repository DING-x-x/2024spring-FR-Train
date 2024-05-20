from argparse import Namespace
import torch
import matplotlib.pyplot as plt

# 根据不同的数据集设置对应的超参
def setopt(y_val, datatype, dataname):
    if datatype == "clean" and dataname == "synthetic":
        train_opt = Namespace(val=len(y_val), n_epochs=4000, k=3, lr_g=0.005, lr_f=0.01, lr_r=0.001)
        lambda_f_set = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]  # 公平性鉴别器的不同程度
        lambda_r = 0.1  # 鲁棒性鉴别器超参数，调整鲁棒性程度
    elif datatype == "poisoned" and dataname == "synthetic":
        train_opt = Namespace(val=len(y_val), n_epochs=10000, k=5, lr_g=0.001, lr_f=0.001, lr_r=0.001)
        lambda_f_set = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.52]  # 公平性鉴别器的不同程度
        lambda_r = 0.4  # 鲁棒性鉴别器超参数，调整鲁棒性程度
    elif datatype == "clean" and dataname == "adult":
        train_opt = Namespace(val=len(y_val), n_epochs=3000, k=3, lr_g=0.005, lr_f=0.01, lr_r=0.001)
        lambda_f_set = [0.2, 0.4, 0.6, 0.8, 0.85]  # 公平性鉴别器的不同程度
        lambda_r = 0.1  # 鲁棒性鉴别器超参数，调整鲁棒性程度
    elif datatype == "poisoned" and dataname == "adult":
        train_opt = Namespace(val=len(y_val), n_epochs=4000, k=5, lr_g=0.001, lr_f=0.001, lr_r=0.001)
        lambda_f_set = [0.2, 0.4, 0.5, 0.59]  # 公平性鉴别器的不同程度
        lambda_r = 0.4  # 鲁棒性鉴别器超参数，调整鲁棒性程度
    elif datatype == "clean" and dataname == "german":
        train_opt = Namespace(val=len(y_val), n_epochs=1000, k=3, lr_g=0.01, lr_f=0.001, lr_r=0.001)
        lambda_f_set = [0.2, 0.4, 0.6, 0.8, 0.85]  # 公平性鉴别器的不同程度
        lambda_r = 0.05  # 鲁棒性鉴别器超参数，调整鲁棒性程度
    elif datatype == "poisoned" and dataname == "german":
        train_opt = Namespace(val=len(y_val), n_epochs=3000, k=5, lr_g=0.001, lr_f=0.001, lr_r=0.001)
        lambda_f_set = [0.2, 0.4, 0.5, 0.6]  # 公平性鉴别器的不同程度
        lambda_r = 0.4  # 鲁棒性鉴别器超参数，调整鲁棒性程度

    return train_opt, lambda_f_set, lambda_r


def test_model(model_, X, y, s1):
    """
    用于模型性能测试

    Args:
        model_: 需要进行性能度量的模型
        X: 测试集特征
        y: 测试集标签
        s1: 测试集敏感属性

    Returns:
        模型准确率、Disparate Impact、Statistical Parity（后加）
    """

    model_.eval()

    y_hat = model_(X).squeeze()
    prediction = (y_hat > 0.0).int().squeeze()
    y = (y > 0.0).int()

    z_0_mask = (s1 == 0)
    z_1_mask = (s1 == 1)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))
    z_0_mask = (s1 == 0).squeeze()
    z_1_mask = (s1 == 1).squeeze()

    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1

    y_1_z_0_mask = ((y == 1.0) & (s1 == 0.0)).squeeze()
    y_1_z_1_mask = ((y == 1.0) & (s1 == 1.0)).squeeze()

    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))

    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1

    test_acc = torch.sum((prediction == y)).float() / len(y)
    print("Test accuracy: {}".format(test_acc))
    print("P(y_hat=1 | z=0) = {:.3f}, P(y_hat=1 | z=1) = {:.3f}".format(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1))
    print(
        "P(y_hat=1 | y=1, z=0) = {:.3f}, P(y_hat=1 | y=1, z=1) = {:.3f}".format(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1))
    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    print("Disparate Impact ratio = {:.3f}".format(min_dp / max_dp))
    print("Demographic Parity = {:.3f}".format(max_dp - min_dp))
    #     min_eo = min(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1)
    #     max_eo = max(Pr_y_hat_1_y_1_z_0, Pr_y_hat_1_y_1_z_1)
    #     print("Equal Opportunity ratio = {:.3f}".format(min_eo/max_eo))
    return test_acc, min_dp / max_dp, max_dp - min_dp


def plot_losses(c_losses, d_f_losses, d_r_losses, lambda_f, lambda_r):
    """
    绘制训练过程中分类器、公平性鉴别器和鲁棒性鉴别器的损失变化曲线

    Args:
        c_losses: 分类器损失列表
        d_f_losses: 公平性鉴别器损失列表
        d_r_losses: 鲁棒性鉴别器损失列表
        lambda_f: 公平性超参数
        lambda_r: 鲁棒性超参数
    """
    epochs = range(len(c_losses))

    plt.figure(figsize=(12, 6))

    # 分类器损失
    plt.plot(epochs, c_losses, label='Classifier Loss (c_losses)', color='blue')

    # 公平性鉴别器损失
    plt.plot(range(len(d_f_losses)), d_f_losses, label='Fairness Discriminator Loss (d_f_losses)', color='green')

    # 鲁棒性鉴别器损失
    plt.plot(range(len(d_r_losses)), d_r_losses, label='Robustness Discriminator Loss (d_r_losses)', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Losses During Training (lambda_f={lambda_f}, lambda_r={lambda_r})')
    plt.legend()
    plt.grid(True)
    plt.show()