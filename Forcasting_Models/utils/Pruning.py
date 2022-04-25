from datetime import datetime

def convert_time_unit(time_str):
    unit = time_str[-1]
    num = None
    if len(time_str) != 1:
        num = int(time_str[:-1])
    if unit == 't' or unit == 'T':
        if num != None:
            return 60/num
        return 60
    elif unit == 'h' or unit == 'H':
        if num != None:
            return 1/num
        return 1
    elif unit == 'd' or unit == 'D':
        if num != None:
            return (1/7)/num
        return (1/7)
    else:
        return num

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return abs((d2 - d1).days)

def calc_expected_datapoint(days, interval):
    interval = convert_time_unit(interval)
    working_days = 250
    Total_days = 365
    market_open = 9
    market_close = 16
    HolidayMultiplier = working_days / Total_days
    hour_in_day = market_close-market_open
    expected_datapoints = (days*HolidayMultiplier) * hour_in_day * interval
    return expected_datapoints

def is_there_enough_points(from_date,to_date,num,tolerance,interval):
    days = days_between(from_date,to_date)
    expected_datapoints = calc_expected_datapoint(days,interval)
    if(expected_datapoints*tolerance < num):
        return True
    else:
        return False