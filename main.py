from argparse import Namespace
from model_arch import *
from training_process import *
from Args import *
from data_loader import *
from utils import *

args = MyArgs().parse_args()

# 不同数据集与不同数据类型的运行参数
# python main.py --dataname synthetic --type clean
# python main.py --dataname synthetic --type poisoned
# python main.py --dataname adult --type clean
# python main.py --dataname adult --type poisoned
# python main.py --dataname german --type clean
# python main.py --dataname german --type poisoned

if __name__ == '__main__':

    XS_train, XS_val, XS_test, y_train, y_val, y_test, s1_train, s1_val, s1_test = getdata(args["dataname"],args["type"])

    print("--------------------- Number of Data -------------------------")
    print(
        "Train data : %d, Validation data : %d, Test data : %d "
        % (len(y_train), len(y_val), len(y_test))
    )
    print("--------------------------------------------------------------")

    train_result = []
    train_tensors = Namespace(XS_train=XS_train, y_train=y_train, s1_train=s1_train)
    val_tensors = Namespace(XS_val=XS_val, y_val=y_val, s1_val=s1_val)
    test_tensors = Namespace(XS_test=XS_test, y_test=y_test, s1_test=s1_test)

    train_opt,lambda_f_set,lambda_r = setopt(y_val,args["type"],args["dataname"])

    seed = 1

    torch.cuda.empty_cache()
    for lambda_f in lambda_f_set:
        train_result.append(
            train_model(args['dataname'],train_tensors, val_tensors, test_tensors, train_opt, lambda_f=lambda_f, lambda_r=lambda_r,
                        seed=seed))

    print("--------------------------------------------------------------------------------")
    print(f"------------------ Training Results of FR-Train on {args['type']} {args['dataname']} data ------------------")
    for i in range(len(train_result)):
        print(
            "[Lambda_f: %.2f] [Lambda_r: %.2f] Accuracy : %.3f, Disparate Impact : %.3f, Demographic Parity : %.3f "
            % (train_result[i][0][0], train_result[i][0][1], train_result[i][0][2], train_result[i][0][3], train_result[i][0][4])
        )
    print("--------------------------------------------------------------------------------")
