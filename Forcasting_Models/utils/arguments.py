import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--primarycategory", nargs="+", required=False)
    parser.add_argument("--secondarycategory", nargs="+", required=False)
    parser.add_argument("--companyid", nargs="+", required=False, default=[15515])

    args = parser.parse_args()

    return args
