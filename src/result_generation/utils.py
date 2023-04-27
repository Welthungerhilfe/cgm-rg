from datetime import datetime
from constants import AGE_LT_2, STANDING_SCAN_STEPS
import numpy as np

MIN_HEIGHT = 45
MAX_HEIGHT = 120
MAX_AGE = 1856.0


def calculate_age(dob, scan_date):
    date_dob = datetime.strptime(dob, "%Y-%m-%d")
    date_scan = datetime.strptime(scan_date, '%Y-%m-%dT%H:%M:%SZ')
    delta = date_scan - date_dob
    return delta.days


def remove_outliers(arr):
    arr = arr.flatten().tolist()
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    result = [nums for nums in arr if lower <= nums <= upper]
    arr_new = np.array(result).reshape(len(result), 1)
    return arr_new


def is_child_standing_age_lt_2(age_in_days, scan_type):
    child_standing_age_lt_2 = False
    if int(age_in_days) < AGE_LT_2 and scan_type in STANDING_SCAN_STEPS:
        child_standing_age_lt_2 = True
    return child_standing_age_lt_2
