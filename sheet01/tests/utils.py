""" module which contains helper functions for testing
"""
def is_list_of_floats(lst):
    if not isinstance(lst, list):
        return False
    for item in lst:
        if not isinstance(item, float):
            return False
    return True

def is_float_between_0_and_1(value):
    if isinstance(value, float):
        if value >= 0 and value <= 1:
            return True
    return False

def is_positive_integer(value):
    if isinstance(value, int) and value > 0:
        return True
    return False

def is_positive_float(value):
    if isinstance(value, float) and value > 0:
        return True
    return False

def check_floats_between_zero_and_one(lst):
    for item in lst:
        if isinstance(item, float) and 0 < item < 1:
            return True
    return False