from datetime import datetime

companies_nn = [
    {"kind": "companyid", "value" : ['15594', '15521', '15554']},
    {"kind": "companyid", "value" : ['15594']},
    # {"kind":"secondarycategory", "value":["bank"]},
    # {"kind":"companyid", "value":['15629', '15611', '15521', '48755', '6041']}
]

companies_stat = [
    {"kind": "companyid", "value" : ['15594']},
    # {"kind":"secondarycategory", "value":["bank"]},
    # {"kind":"companyid", "value":['15629', '15611', '15521', '48755', '6041']}
]

granularities = [
    {"horizon":"24 hours", "period": "24 hours", "initial":"72 hours", "gran":"1H"}, 
    {"horizon":"120 minutes", "period": "120 minutes", "initial":"360 minutes", "gran":"30T"}, 
    {"horizon":"120 minutes", "period": "120 minutes", "initial":"360 minutes", "gran":"5T"}, 
    # {"horizon":"360 minutes", "period": "360 minutes", "initial":"400000 minutes", "gran":"15T"}, 
    # # {"horizon":"100 minutes", "period": "100 minutes", "initial":"5800 minutes", "gran":"45T"}, 
    # {"horizon":"84 hours", "period": "84 hours", "initial":"1440 hours", "gran":"12H"}, 
    # {"horizon":"7 days", "period": "7 days", "initial":"30 days", "gran":"1D"}
]

columns_nn = [
    ['close'],
    # ["close", "volume"],
    ["close", "open", "volume"],
    # ["close", "open", "high", "low", "volume"],
]

columns_stat = [
    ['close'],
    # ["close", "open", "high", "low", "volume"],
]



periods_nn = [
    # { "from_date" : "2012-04-01 00:00:00", "to_date" : "2015-04-01 00:00:00"},
    # { "from_date" : "2013-04-01 00:00:00", "to_date" : "2015-04-01 00:00:00"},
    # { "from_date" : "2014-04-01 00:00:00", "to_date" : "2015-04-01 00:00:00"},

    { "from_date" : "2014-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2016-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2015-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2017-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},

    { "from_date" : "2017-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
    { "from_date" : "2018-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
    { "from_date" : "2019-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
    { "from_date" : "2020-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
]

periods_stat = [
    { "from_date" : "2014-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2015-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2016-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},
    { "from_date" : "2017-04-01 00:00:00", "to_date" : "2018-04-01 00:00:00"},

    { "from_date" : "2017-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
    { "from_date" : "2018-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
    { "from_date" : "2019-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
    { "from_date" : "2020-04-01 00:00:00", "to_date" : "2021-04-01 00:00:00"},
]

def calculate_fb_arguments(arguments, gran, from_date, to_date):
    unit = gran[-1]
    if len(gran) > 1:
        value = int(gran[:-1])
    else:
        value = 1
    
    from_date_obj = datetime.strptime(from_date, '%Y-%m-%d %H:%M:%S')
    to_date_obj = datetime.strptime(to_date, '%Y-%m-%d %H:%M:%S')

    days = (to_date_obj - from_date_obj).days * (5/7)

    if unit == 'H':
        datapoints = (days * 8) / value
    if unit == 'T':
        datapoints = (days * 8 * 60) / value
    if unit == 'D':
        datapoints = days

    initial_train_ratio = 0.8
    forecasts_to_make = 50

    initial = initial_train_ratio * datapoints
    horizon = (datapoints - (initial)) / forecasts_to_make
    period = horizon

    arguments.initial = "72 hours"#_add_unit(int(initial), unit)
    arguments.horizon = "24 hours"#_add_unit(int(horizon), unit)
    arguments.period = "24 hours"#_add_unit(int(period), unit)
    
def _add_unit(msg, unit):
    if unit == 'H':
        return f"{msg} hours"
    elif unit == 'T':
        return f"{msg} minutes"
    elif unit == 'D':
        return f"{msg} days"
    else:
        print(f"{unit} not a valid unit")
        exit()

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
    

    arguments.timeunit = granularity["gran"]
    # calculate_fb_arguments(arguments, granularity["gran"], period["from_date"], period["to_date"])
    
    arguments.columns = column

    return arguments, period["from_date"], period["to_date"]


    

    


