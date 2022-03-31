from cv2 import triangulatePoints
import utils.DatasetAccess as db_access
import utils.preprocess as preprocess
import utils.arguments as arg
import FbProphet.fbprophet as fb
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def ensure_valid_values(input_values, actual_values, value_type):
    for value in input_values:
        if not value in actual_values:
            raise Exception(f"Value '{value}' is not a valid {value_type}")
    return True


if __name__ == "__main__":
    arguments = arg.get_arguments()

    connection = db_access.get_connection()

    primary_category = db_access.get_primay_category(connection)
    secondary_category = db_access.get_secondary_category(connection)
    company_id = db_access.get_companyid(connection)

    if arguments.primarycategory:
        if ensure_valid_values(
            arguments.primarycategory, primary_category, "primary category"
        ):
            print(
                f"Models will be trained on companies with primary category in {arguments.primarycategory}"
            )
                    
    elif arguments.secondarycategory:
        if ensure_valid_values(
            arguments.secondarycategory, secondary_category, "secondary category"
        ):
            print(
                f"Models will be trained on companies with secondary category in {arguments.secondarycategory}"
            )
    elif arguments.companyid:
        if ensure_valid_values(arguments.companyid, company_id, "companyid"):
            print(
                f"Models will be trained on companies with company id in {arguments.companyid}"
            )
    else:
        print("No information was provided. No models will be trained.")


    data = db_access.get_data_for_datasetid(
        datasetid=arguments.companyid[0],
        conn=connection,
        interval=arguments.timeunit,
        time=arguments.time,
    )
    print(data.head())

    data = preprocess.rename_dataset_columns(data)
    training, testing = preprocess.get_split_data(data)
    
    model = fb.model_fit(
        training,
        yearly_seasonality=arguments.yearly_seasonality,
        weekly_seasonality=arguments.weekly_seasonality,
        daily_seasonality=arguments.daily_seasonality,
        seasonality_mode=arguments.seasonality_mode,
    )

    print("model has been trained, now predicting..")

    future = fb.get_future_df(  
        model,
        period=arguments.predict_periods,
        freq=arguments.timeunit,
        include_history=arguments.include_history,
    )

    forecast = fb.make_prediction(
        model,
        training,
    )

    fb.plot_forecast(
        model,
        forecast,
        testing,
    )
    print("done!")
