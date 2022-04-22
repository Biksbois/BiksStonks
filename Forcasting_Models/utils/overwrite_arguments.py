
companies = [
    {"kind":"secondarycategory", "value":["bank"]},
    {"kind":"companyid", "value":[15629, 15611, 15521, 48755, 6041]}
]

granularities = [
    {"horizon":"120 minutes", "period": "120 minutes", "initial":"800000 minutes", "gran":"5T"}, 
    {"horizon":"360 minutes", "period": "360 minutes", "initial":"800000 minutes", "gran":"15T"}, 
    {"horizon":"720 minutes", "period": "720 minutes", "initial":"800000 minutes", "gran":"30T"}, 
    {"horizon":"1080 minutes", "period": "1080 minutes", "initial":"800000 minutes", "gran":"45T"}, 
    {"horizon":"6 hours", "period": "6 hours", "initial":"14000 hours", "gran":"1H"}, 
    {"horizon":"100 hours", "period": "100 hours", "initial":"14000 hours", "gran":"12H"}, 
    {"horizon":"7 days", "period": "7 days", "initial":"365 days", "gran":"1D"}
]

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
        arguments.secondarycategory = company["value"]
    elif company["kind"] == "primarycategory":
        arguments.primarycategory= company["value"]
    elif company["kind"] == "companyid":
        arguments.companyid= company["value"]
    else:
        print(f"error. kinds not valid {company['kind']}")
    
    arguments.initial =granularity["initial"]
    arguments.horizon = granularity["horizon"]
    arguments.period = granularity["period"]
    arguments.timeunit = granularity["gran"]
    
    arguments.columns = column

    return arguments, period["from_date"], period["to_date"]
    

    


