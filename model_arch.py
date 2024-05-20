import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def weights_init_normal(m):
    """
    初始化模型权重

    Args:
        m: 需要进行初始化的模型
    Returns:
        None.
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)


# 合成数据
class Classifier_synthetic(nn.Module):
    """
    分类器，用于预测数据标签
    (ref: FR-Train paper, Section 3)

    Attributes:
        model: 分类模型
    """

    def __init__(self):
        super(Classifier_synthetic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 1)
        )

    def forward(self, input_data):
        """
        Args:
            input_data: 数据输入

        Returns:
            预测标签
        """
        output = self.model(input_data)
        return output


# 合成数据 公平鉴别器
class DiscriminatorF_synthetic(nn.Module):

    def __init__(self):
        super(DiscriminatorF_synthetic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_data):
        """
        Args:
            input_data: 输入数据

        Returns:
            predicted_z: 预测的敏感属性值
        """

        predicted_z = self.model(input_data)
        return predicted_z


# 合成数据 鲁棒性鉴别器
class DiscriminatorR_synthetic(nn.Module):

    def __init__(self):
        super(DiscriminatorR_synthetic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_data):
        """
        Args:
            input_data: 输入数据

        Returns:
            validity: 判断是否为poisoned data
        """

        validity = self.model(input_data)
        return validity


# adult数据
class Classifier_adult(nn.Module):
    """
    分类器，用于预测数据标签
    (ref: FR-Train paper, Section 3)

    Attributes:
        model: 分类模型
    """

    def __init__(self):
        super(Classifier_adult, self).__init__()

        self.fc1 = nn.Linear(13, 64)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(64, 32)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(32, 1)  # 第二个隐藏层到输出层

    def forward(self, input_data):
        """
        Args:
            input_data: 数据输入

        Returns:
            预测标签
        """
        x = F.relu(self.fc1(input_data))
        x = F.relu(self.fc2(x))
        y_pred = self.fc3(x)

        return y_pred


# adult公平鉴别器
class DiscriminatorF_adult(nn.Module):

    def __init__(self):
        super(DiscriminatorF_adult, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_data):
        """
        Args:
            input_data: 输入数据

        Returns:
            predicted_z: 预测的敏感属性值
        """

        predicted_z = self.model(input_data)
        return predicted_z


# adult 鲁棒性鉴别器
class DiscriminatorR_adult(nn.Module):

    def __init__(self):
        super(DiscriminatorR_adult, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(14),
            nn.Linear(14, 1),  # 输入层到隐藏层
            nn.Sigmoid(),
        )

    def forward(self, input_data):

        validity = self.model(input_data)
        return validity

# german数据
class Classifier_german(nn.Module):
    """
    分类器，用于预测数据标签
    (ref: FR-Train paper, Section 3)

    Attributes:
        model: 分类模型
    """

    def __init__(self):
        super(Classifier_german, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(20),
            nn.Linear(20, 64),  # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_data):
        """
        Args:
            input_data: 数据输入

        Returns:
            预测标签
        """

        y_pred = self.model(input_data)

        return y_pred


# 合成数据 公平鉴别器
class DiscriminatorF_german(nn.Module):

    def __init__(self):
        super(DiscriminatorF_german, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_data):
        """
        Args:
            input_data: 输入数据

        Returns:
            predicted_z: 预测的敏感属性值
        """

        predicted_z = self.model(input_data)
        return predicted_z


# 合成数据 鲁棒性鉴别器
class DiscriminatorR_german(nn.Module):

    def __init__(self):
        super(DiscriminatorR_german, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(21),
            nn.Linear(21, 1),  # 输入层到隐藏层
            nn.Sigmoid(),
        )

    def forward(self, input_data):

        validity = self.model(input_data)
        return validity
