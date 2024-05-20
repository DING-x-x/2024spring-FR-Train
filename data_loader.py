import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


# 加载数据
def getdata(dataname, datatype):
    if dataname == "synthetic":
        XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test = getSyntheticData(datatype)
    elif dataname == "adult":
        XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test = getAdultData(datatype)
    elif dataname == "german":
        XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test = getGermanData(datatype)

    return XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test


# 加载合成数据
def getSyntheticData(datatype):
    num_train = 2000
    num_val1 = 200
    num_val2 = 500
    num_test = 1000

    X = np.load('datasets/synthetic_dataset/X_synthetic.npy')
    y = np.load('datasets/synthetic_dataset/y_synthetic.npy')
    y_poi = np.load('datasets/synthetic_dataset/y_poi.npy')
    s1 = np.load('datasets/synthetic_dataset/s1_synthetic.npy')

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    y_poi = torch.FloatTensor(y_poi)
    s1 = torch.FloatTensor(s1)

    X_train = X[:num_train - num_val1]
    if datatype == "clean":
        y_train = y[:num_train - num_val1]  # Clean label
    else:
        y_train = y_poi[:num_train - num_val1]  # Poisoned label
    s1_train = s1[:num_train - num_val1]

    X_val = X[num_train: num_train + num_val1]
    y_val = y[num_train: num_train + num_val1]
    s1_val = s1[num_train: num_train + num_val1]

    X_test = X[num_train + num_val1 + num_val2: num_train + num_val1 + num_val2 + num_test]
    y_test = y[num_train + num_val1 + num_val2: num_train + num_val1 + num_val2 + num_test]
    s1_test = s1[num_train + num_val1 + num_val2: num_train + num_val1 + num_val2 + num_test]

    XS_train = torch.cat([X_train, s1_train.reshape((s1_train.shape[0], 1))], dim=1)
    XS_val = torch.cat([X_val, s1_val.reshape((s1_val.shape[0], 1))], dim=1)
    XS_test = torch.cat([X_test, s1_test.reshape((s1_test.shape[0], 1))], dim=1)

    return XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test


# 加载adult数据集
def getAdultData(datatype):
    train_data_url = "datasets/public_dataset/adult/adult.data"
    test_data_url = "datasets/public_dataset/adult/adult.test"

    column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                    "hours_per_week", "native_country", "income"]

    train_data = pd.read_csv(train_data_url, names=column_names)
    test_data = pd.read_csv(test_data_url, names=column_names, skiprows=1)

    train_data = train_data.applymap(lambda x: x.strip() if type(x) is str else x)  # 如果数据两端包含空格，则去掉空格
    test_data = test_data.applymap(lambda x: x.strip() if type(x) is str else x)  # 如果数据两端包含空格，则去掉空格
    train_data = train_data[train_data.ne("?").all(axis=1)].reset_index(drop=True)
    test_data = test_data[test_data.ne("?").all(axis=1)].reset_index(drop=True)

    # 删除fnlwgt属性 属性对结果没用用处
    train_data.drop("fnlwgt", axis=1, inplace=True)
    test_data.drop("fnlwgt", axis=1, inplace=True)

    # 将 income 转为二值：'<=50K' 转为 0， '>50K' 转为 1
    train_data['income'] = train_data['income'].apply(lambda x: -1 if x == '<=50K' else 1)
    test_data['income'] = test_data['income'].apply(lambda x: -1 if x == '<=50K.' else 1)

    # 将 sex 转为二值
    train_data['sex'] = train_data['sex'].apply(lambda x: 0 if x == 'Female' else 1)
    test_data['sex'] = test_data['sex'].apply(lambda x: 0 if x == 'Female' else 1)

    # 将 age 进行离散化
    age_bins = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
    train_data['age'] = pd.cut(train_data['age'], bins=age_bins, labels=False, right=False)
    test_data['age'] = pd.cut(test_data['age'], bins=age_bins, labels=False, right=False)

    categorical_cols = ["workclass", "education", "marital_status", "occupation",
                        "relationship", "race", "native_country"]

    for col in categorical_cols:
        le = LabelEncoder()
        le.fit_transform(train_data[col])
        train_data[col] = le.transform(train_data[col])
        test_data[col] = le.transform(test_data[col])

    # 划分训练集和验证集
    train_data, val_data = train_test_split(train_data, test_size=0.5, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    test_data, data = train_test_split(test_data, test_size=0.5, random_state=42)
    # 在训练集中反转10%的 sex=1 的标签
    sex_1_indices = train_data[train_data['sex'] == 1].index
    num_to_flip = int(0.1 * len(sex_1_indices))
    flip_indices = sex_1_indices[:num_to_flip]

    poi_data = train_data.copy()
    poi_data.loc[flip_indices, 'income'] = -1*  poi_data.loc[flip_indices, 'income']

    if datatype == "clean":
        y_train = train_data["income"]
    else:
        y_train = poi_data["income"]
    y_val = val_data["income"]
    y_test = test_data["income"]

    s1_train = train_data["sex"]
    s1_val = val_data["sex"]
    s1_test = test_data["sex"]

    XS_train = train_data.drop("income",axis = 1)
    XS_val = val_data.drop("income",axis = 1)
    XS_test = test_data.drop("income",axis = 1)

    # 将 DataFrame 转换为 torch.FloatTensor
    XS_train = torch.FloatTensor(XS_train.to_numpy())
    XS_val = torch.FloatTensor(XS_val.to_numpy())
    XS_test = torch.FloatTensor(XS_test.to_numpy())

    y_train = torch.FloatTensor(y_train.to_numpy())
    y_val = torch.FloatTensor(y_val.to_numpy())
    y_test = torch.FloatTensor(y_test.to_numpy())

    s1_train = torch.FloatTensor(s1_train.to_numpy())
    s1_val = torch.FloatTensor(s1_val.to_numpy())
    s1_test = torch.FloatTensor(s1_test.to_numpy())

    return XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test


# German Credit数据集
def getGermanData(datatype):
    # 数据集 URL
    data_url = "datasets/public_dataset/german/german.data"

    # 特征名称
    column_names = ["Status", "Duration", "Credit_history", "Purpose", "Credit_amount",
                    "Savings", "Employment", "Installment_rate", "Personal_status_sex",
                    "Other_debtors", "Present_residence", "Property", "Age", "Other_installment_plans",
                    "Housing", "Number_of_credits", "Job", "Liable_people", "Telephone", "Foreign_worker", "Target"]

    # 读取数据集
    data = pd.read_csv(data_url, delimiter=' ', header=None, names=column_names)

    # 将目标变量转为二值：1表示好客户，2表示坏客户，转为-1和1
    data['Target'] = data['Target'].apply(lambda x: 1 if x == 1 else -1)

    # 将 Personal_status_sex 转为二值
    data['Personal_status_sex'] = data['Personal_status_sex'].apply(lambda x: 0 if 'A92' in x or 'A95' in x else 1)

    # 将 Age 进行离散化
    age_bins = [0, 25, 35, 45, 55, 65, 75]
    data['Age'] = pd.cut(data['Age'], bins=age_bins, labels=False, right=True)

    # 处理类别特征
    categorical_cols = ["Status", "Credit_history", "Purpose", "Savings", "Employment",
                        "Other_debtors", "Property", "Other_installment_plans", "Housing", "Job",
                        "Telephone", "Foreign_worker"]

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # 划分训练集和验证集
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # 在训练集中反转10%的 Personal_status_sex = 1 的标签
    sex_1_indices = train_data[train_data['Personal_status_sex'] == 1].index
    num_to_flip = int(0.1 * len(sex_1_indices))
    flip_indices = sex_1_indices[:num_to_flip]

    poi_data = train_data.copy()
    poi_data.loc[flip_indices, 'Target'] = (- 1) * poi_data.loc[flip_indices, 'Target']

    if datatype == "clean":
        y_train = train_data["Target"]
    else:
        y_train = poi_data["Target"]
    y_val = val_data["Target"]
    y_test = test_data["Target"]

    s1_train = train_data["Personal_status_sex"]
    s1_val = val_data["Personal_status_sex"]
    s1_test = test_data["Personal_status_sex"]

    XS_train = train_data.drop("Target", axis=1)
    XS_val = val_data.drop("Target", axis=1)
    XS_test = test_data.drop("Target", axis=1)

    # 将 DataFrame 转换为 torch.FloatTensor
    XS_train = torch.FloatTensor(XS_train.to_numpy())
    XS_val = torch.FloatTensor(XS_val.to_numpy())
    XS_test = torch.FloatTensor(XS_test.to_numpy())

    y_train = torch.FloatTensor(y_train.to_numpy())
    y_val = torch.FloatTensor(y_val.to_numpy())
    y_test = torch.FloatTensor(y_test.to_numpy())

    s1_train = torch.FloatTensor(s1_train.to_numpy())
    s1_val = torch.FloatTensor(s1_val.to_numpy())
    s1_test = torch.FloatTensor(s1_test.to_numpy())

    return XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test


