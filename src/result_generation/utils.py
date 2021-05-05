from datetime import datetime

MIN_HEIGHT = 45
MAX_HEIGHT = 120
MAX_AGE = 1856.0


def calculate_age(dob, scan_date):
    date_dob = datetime.strptime(dob, "%Y-%m-%d")
    date_scan = datetime.strptime(scan_date, '%Y-%m-%dT%H:%M:%SZ')
    delta = date_scan - date_dob
    return delta.days
