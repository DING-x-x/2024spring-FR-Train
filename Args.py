import argparse


class MyArgs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()  # create an ArgumentParser object
        self.parser.add_argument("-d","--dataname",
                                 choices=['synthetic','adult','german'],
                                 default='synthetic',
                                 help='choose which dataset to use')
        self.parser.add_argument("-t","--type",
                                 choices=["clean","poisoned"],
                                 default='clean',
                                 help="choose which type dataset to use")

    def parse_args(self):
        return vars(self.parser.parse_args())
