"""Utility functions for robust redundant calibration"""

import numpy

import pyuvdata
from pyuvdata import utils as uvutils


def find_nearest(array, value):
    """Find nearest value in array

    Returns nearest value in array, and index of that nearest value
    """
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return array[idx], idx


def jd_to_lst(JD_time, telescope='HERA'):
    """Converts fractional JDs of HERAData object into LSTs"""
    lat_lon_alt = pyuvdata.telescopes.get_telescope(telescope).telescope_location_lat_lon_alt_degrees
    lst = uvutils.get_lst_for_time([JD_time], *lat_lon_alt)
    return lst[0]
