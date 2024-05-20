from model_arch import *
from utils import *


def train_model(dataname, train_tensors, val_tensors, test_tensors, train_opt, lambda_f, lambda_r, seed):
    """
    定义模型训练过程

    Args:
        dataname: 数据集名称
        train_tensors: 训练集数据
        val_tensors: 验证集数据
        test_tensors: 测试集数据
        train_opt: 超参数集合，包含验证集大小、鉴别器参与比例、轮数、学习率
        lambda_f: 公平性超参，用于控制公平性程度 (ref: FR-Train paper, Section 3.3).
        lambda_r: 鲁棒性超参，用于控制鲁棒性程度 (ref: FR-Train paper, Section 3.3).
        seed: 随机数种子（便于复现）

    Returns:
       不同超参数下的模型表现
    """

    XS_train = train_tensors.XS_train.cuda()
    y_train = train_tensors.y_train.cuda().view(-1, 1)
    s1_train = train_tensors.s1_train.cuda().view(-1, 1)

    XS_val = val_tensors.XS_val.cuda()
    y_val = val_tensors.y_val.cuda()
    s1_val = val_tensors.s1_val.cuda()

    XS_test = test_tensors.XS_test.cuda()
    y_test = test_tensors.y_test.cuda()
    s1_test = test_tensors.s1_test.cuda()

    # 保存模型表现
    test_result = []

    val = train_opt.val  # 验证集大小
    k = train_opt.k  # 分类器与鉴别器更新比例
    n_epochs = train_opt.n_epochs  # 训练轮数

    XSY_val = torch.cat([XS_val, y_val.view(y_val.shape[0], 1)], dim=1).cuda()
    XSY_val_data = XSY_val[:val]

    # 存储各个模型损失值
    c_losses = []
    d_f_losses = []
    d_r_losses = []

    # 使用BCE作为损失函数
    bce_loss = torch.nn.BCELoss()

    # 定义一个分类器，两个鉴别器（分别用于公平性与鲁棒性）
    if dataname == "synthetic":
        classifier = Classifier_synthetic().cuda()
        discriminator_F = DiscriminatorF_synthetic().cuda()
        discriminator_R = DiscriminatorR_synthetic().cuda()
    elif dataname == "adult":
        classifier = Classifier_adult().cuda()
        discriminator_F = DiscriminatorF_adult().cuda()
        discriminator_R = DiscriminatorR_adult().cuda()
    elif dataname == "german":
        classifier = Classifier_german().cuda()
        discriminator_F = DiscriminatorF_german().cuda()
        discriminator_R = DiscriminatorR_german().cuda()

    # 初始化各个模型权重
    torch.manual_seed(seed)
    classifier.apply(weights_init_normal)
    discriminator_F.apply(weights_init_normal)
    discriminator_R.apply(weights_init_normal)

    # 定义模型优化器
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=train_opt.lr_g)
    optimizer_D_F = torch.optim.SGD(discriminator_F.parameters(), lr=train_opt.lr_f)
    optimizer_D_R = torch.optim.SGD(discriminator_R.parameters(), lr=train_opt.lr_r)

    train_len = XS_train.shape[0]
    val_len = XSY_val.shape[0]

    # 鲁棒性鉴别器的真实标签（即是否为合成数据）
    Tensor = torch.cuda.FloatTensor
    valid = Variable(Tensor(train_len, 1).fill_(1.0), requires_grad=False)
    generated = Variable(Tensor(train_len, 1).fill_(0.0), requires_grad=False)
    fake = Variable(Tensor(train_len, 1).fill_(0.0), requires_grad=False)
    clean = Variable(Tensor(val_len, 1).fill_(1.0), requires_grad=False)

    r_weight = torch.ones_like(y_train, requires_grad=False).float()
    r_ones = torch.ones_like(y_train, requires_grad=False).float()

    for epoch in range(n_epochs):

        # 前500轮内消除梯度，过早的鉴别器干扰容易影响分类器性能
        if epoch % k == 0 or epoch < 500:
            optimizer_C.zero_grad()

        gen_y = classifier(XS_train)
        gen_data = torch.cat([XS_train, gen_y.detach().reshape((gen_y.shape[0], 1))], dim=1)

        # 训练公平性鉴别器
        optimizer_D_F.zero_grad()

        # 鉴别器通过分类结果预测敏感属性值
        d_f_loss = bce_loss(discriminator_F(gen_y.detach()), s1_train)
        d_f_loss.backward()
        d_f_losses.append(d_f_loss)
        optimizer_D_F.step()

        # 训练鲁棒性鉴别器
        optimizer_D_R.zero_grad()

        # 鉴别器主要用于分辨当前数据是验证集还是训练集
        clean_loss = bce_loss(discriminator_R(XSY_val_data), clean)
        poison_loss = bce_loss(discriminator_R(gen_data.detach()), fake)
        d_r_loss = 0.5 * (clean_loss + poison_loss)

        d_r_loss.backward()
        d_r_losses.append(d_r_loss)
        optimizer_D_R.step()

        #  更新分类器参数，计算分类器性能
        #  前500轮不考虑鉴别器的影响，后续每间隔k轮结合鉴别器更新分类器参数
        if epoch < 500:
            c_loss = bce_loss((F.tanh(gen_y) + 1) / 2, (y_train + 1) / 2)
            c_loss.backward()
            optimizer_C.step()

        elif epoch % k == 0:
            r_decision = discriminator_R(gen_data)
            r_gen = bce_loss(r_decision, generated)

            if epoch % 100 == 0:
                loss_ratio = (c_losses[-1] / d_r_losses[-1]).detach()
                a = 1 / (1 + torch.exp(-(loss_ratio - 3)))
                b = 1 - a
                r_weight_tmp = r_decision.detach().squeeze()
                r_weight = a * r_weight_tmp + b * r_ones

            f_cost = F.binary_cross_entropy(discriminator_F(gen_y), s1_train, reduction="none").squeeze()
            c_cost = F.binary_cross_entropy_with_logits(gen_y.squeeze(), (y_train.squeeze() + 1) / 2,
                                                        reduction="none").squeeze()

            f_gen = torch.mean(f_cost * r_weight)
            c_loss = (1 - lambda_f - lambda_r) * torch.mean(c_cost * r_weight) - lambda_r * r_gen - lambda_f * f_gen

            c_loss.backward()
            optimizer_C.step()

        c_losses.append(c_loss)

        if epoch % 200 == 0:
            print(
                "[Lambda_f: %1f] [Epoch %d/%d] [D_F loss: %f] [D_R loss: %f] [c loss: %f]"
                % (lambda_f, epoch, n_epochs, d_f_losses[-1], d_r_losses[-1], c_losses[-1])
            )

    #     torch.save(classifier.state_dict(), './FR-Train_on_clean_synthetic.pth')
    tmp = test_model(classifier, XS_test, y_test, s1_test)
    test_result.append([lambda_f, lambda_r, tmp[0].item(), tmp[1], tmp[2]])
    # plot_losses(c_losses, d_f_losses, d_r_losses, lambda_f, lambda_r)

    return test_result

