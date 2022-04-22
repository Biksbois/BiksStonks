
companies = [
    {"kind":"secondarycategory", "value":["bank"]},
    {"kind":"companyid", "value":[15629, 15611, 15521, 48755, 6041]}
]

granularities = ["5T", "15T", "30T", "45T", "1H", "12H", "1D"]

columns = [
    ['close'],
    ["close", "volume"],
    ["close", "open", "volume"],
    ["close", "open", "high", "low", "volume"],
]

periods = [
    { "from_date" : "2012-04-01 00:00:00", "to_date" : "2014-04-01 00:00:00"},
    { "from_date" : "2014-04-01 00:00:00", "to_date" : "2016-04-01 00:00:00"},
    { "from_date" : "2016-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2018-04-01 00:00:00", "to_date" : "2020-04-01 00:00:00"},
    { "from_date" : "2020-04-01 00:00:00", "to_date" : "2022-04-01 00:00:00"},
]

def overwrite_arguments(arguments, granularity, column, period, company):
    arguments.companyid = None
    arguments.primarycategory = None
    arguments.secondarycategory = None

    if company["kind"] == "secondarycategory":
        arguments.secondarycategory
    elif company["kind"] == "primarycategory":
        arguments.primarycategory
    elif company["kind"] == "companyid":
        arguments.companyid
    else:
        print(f"error. kinds not valid {company['kind']}")
    


