from datetime import date


def age(dob, scan_date):

    f_date = date(2014, 7, 2)
    l_date = date(2014, 7, 11)
    delta = l_date - f_date
    print(delta.days)

age("2020-11-12","f")