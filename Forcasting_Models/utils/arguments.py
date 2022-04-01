import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="fb", help="Model to use", choices=["fb", "informer", "lstm"])
    parser.add_argument("--primarycategory", nargs="+", required=False)
    parser.add_argument("--secondarycategory", nargs="+", required=False)
    parser.add_argument("--companyid", nargs="+", required=False, default=[15515])
    parser.add_argument("--timeunit", required=False, default='H')
    parser.add_argument("--time", required=False, default='0001-01-01 00:00:00')
    parser.add_argument("--seasonality_mode", required=False, default='additive')
    parser.add_argument("--yearly_seasonality", required=False, default=False)
    parser.add_argument("--weekly_seasonality", required=False, default=False)
    parser.add_argument("--daily_seasonality", required=False, default=False)
    parser.add_argument("--include_history", required=False, default=False)
    parser.add_argument("--predict_periods", required=False, default=1000)


    args = parser.parse_args()

    return args
