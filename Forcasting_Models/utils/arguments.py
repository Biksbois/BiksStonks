import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to use",
        choices=["all", "fb", "informer", "lstm", "arima"],
    )
    parser.add_argument("--primarycategory", nargs="+", required=False)
    parser.add_argument("--secondarycategory", nargs="+", required=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use_args", type=str, default="True")
    parser.add_argument("--save_data", type=str, default="True")
    parser.add_argument("--companyid", nargs="+", required=False, default=[15515])
    parser.add_argument("--timeunit", required=False, default="1H")
    parser.add_argument("--time", required=False, default="0001-01-01 00:00:00")
    parser.add_argument("--seasonality_mode", required=False, default="additive")
    parser.add_argument("--yearly_seasonality", required=False, default=False)
    parser.add_argument("--weekly_seasonality", required=False, default=False)
    parser.add_argument("--daily_seasonality", required=False, default=False)
    parser.add_argument("--include_history", required=False, default=True)
    # parser.add_argument("--predict_periods", required=False, default=1000)
    parser.add_argument("--horizon", required=False, default="48 hours")
    parser.add_argument("--period", required=False, default="24 hours")
    parser.add_argument("--initial", required=False, default="72 hours")
    parser.add_argument("--use_sentiment", required=False, default=False)
    parser.add_argument(
        "--columns",
        nargs="+",
        required=False,
        default=["close", "open", "high", "low", "volume"],
    )

    args = parser.parse_args()

    return args
