import utils.DatasetAccess as db_access
import utils.arguments as arg


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

    print("done!")
