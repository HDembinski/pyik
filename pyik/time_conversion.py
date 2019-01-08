# -*- coding: utf-8 -*-
"""Time conversion utilities."""
from __future__ import print_function

def utc_to_gps(utcsec):
    """
    Convert UTC seconds to GPS seconds.

    Parameters
    ----------
    utcsec: int
      Time in utc seconds.

    Returns
    -------
    Time in GPS seconds.

    Notes
    -----
    The code is ported from Offline.

    Examples
    --------
    >>> utc_to_gps(315964800) # Jan 6th, 1980
    0
    """

    import time

    def seconds(year, month, day):
        return time.mktime((year, month, day, 0, 0, 0, 0, 0, 0)) - time.timezone

    kUTCGPSOffset0 = (10 * 365 + 7) * 24 * 3600
    kLeapSecondList = (
        (seconds(1981, 7, 1), 1),  # 1 JUL 1981
        (seconds(1982, 7, 1), 2),  # 1 JUL 1982
        (seconds(1983, 7, 1), 3),  # 1 JUL 1983
        (seconds(1985, 7, 1), 4),  # 1 JUL 1985
        (seconds(1988, 1, 1), 5),  # 1 JAN 1988
        (seconds(1990, 1, 1), 6),  # 1 JAN 1990
        (seconds(1991, 1, 1), 7),  # 1 JAN 1991
        (seconds(1992, 7, 1), 8),  # 1 JUL 1992
        (seconds(1993, 7, 1), 9),  # 1 JUL 1993
        (seconds(1994, 7, 1), 10),  # 1 JUL 1994
        (seconds(1996, 1, 1), 11),  # 1 JAN 1996
        (seconds(1997, 7, 1), 12),  # 1 JUL 1997
        (seconds(1999, 1, 1), 13),  # 1 JAN 1999
        # DV: 2000 IS a leap year since it is divisible by 400,
        #     ie leap years here are 2000 and 2004 -> leap days = 6
        (seconds(2006, 1, 1), 14),  # 1 JAN 2006
        (seconds(2009, 1, 1), 15),  # 1 JAN 2009, from IERS Bulletin C No. 36
        (seconds(2012, 7, 1), 16),  # 1 JUL 2012, from IERS Bulletin C No. 43
        # 1 JUL 2015, from IERS Bulletin C No. 49
        # (https://hpiers.obspm.fr/iers/bul/bulc/bulletinc.49)
        (seconds(2015, 7, 1), 17),
        (seconds(2017, 1, 1), 18)   # 1 JAN 2017, from IERS Bulletin C No. 52
    )

    leapSeconds = 0
    for x in reversed(kLeapSecondList):
        if utcsec >= x[0]:
            leapSeconds = x[1]
            break

    return utcsec - kUTCGPSOffset0 + leapSeconds


def gps_to_utc(gpssec):
    """
    Convert GPS seconds to UTC seconds.

    Parameters
    ----------
    gpssec: int
      Time in GPS seconds.

    Returns
    -------
    Time in UTC seconds.

    Notes
    -----
    The code is ported from Offline.

    Examples
    --------
    >>> gps_to_utc(0) # Jan 6th, 1980
    315964800
    """

    kSecPerDay = 24 * 3600
    kUTCGPSOffset0 = (10 * 365 + 7) * kSecPerDay

    kLeapSecondList = (
        ((361 + 0 * 365 + 0 + 181) * kSecPerDay + 0, 1),  # 1 JUL 1981
        ((361 + 1 * 365 + 0 + 181) * kSecPerDay + 1, 2),  # 1 JUL 1982
        ((361 + 2 * 365 + 0 + 181) * kSecPerDay + 2, 3),  # 1 JUL 1983
        ((361 + 4 * 365 + 1 + 181) * kSecPerDay + 3, 4),  # 1 JUL 1985
        ((361 + 7 * 365 + 1) * kSecPerDay + 4, 5),  # 1 JAN 1988
        ((361 + 9 * 365 + 2) * kSecPerDay + 5, 6),  # 1 JAN 1990
        ((361 + 10 * 365 + 2) * kSecPerDay + 6, 7),  # 1 JAN 1991
        ((361 + 11 * 365 + 3 + 181) * kSecPerDay + 7, 8),  # 1 JUL 1992
        ((361 + 12 * 365 + 3 + 181) * kSecPerDay + 8, 9),  # 1 JUL 1993
        ((361 + 13 * 365 + 3 + 181) * kSecPerDay + 9, 10),  # 1 JUL 1994
        ((361 + 15 * 365 + 3) * kSecPerDay + 10, 11),  # 1 JAN 1996
        ((361 + 16 * 365 + 4 + 181) * kSecPerDay + 11, 12),  # 1 JUL 1997
        ((361 + 18 * 365 + 4) * kSecPerDay + 12, 13),  # 1 JAN 1999
        # DV: 2000 IS a leap year since it is divisible by 400,
        #     ie leap years here are 2000 and 2004 -> leap days = 6
        ((361 + 25 * 365 + 6) * kSecPerDay + 13, 14),  # 1 JAN 2006
        ((361 + 28 * 365 + 7) * kSecPerDay + 14, 15),  # 1 JAN 2009
        ((361 + 31 * 365 + 8 + 181) * kSecPerDay + 15, 16),  # 1 JUL 2012
        ((361 + 34 * 365 + 8 + 181) * kSecPerDay + 16, 17),  # 1 JUL 2015
        ((361 + 36 * 365 + 9) * kSecPerDay + 17, 18)  # 1 JAN 2017
    )

    leapSeconds = 0
    for x in reversed(kLeapSecondList):
        if gpssec >= x[0]:
            leapSeconds = x[1]
            break

    return gpssec + kUTCGPSOffset0 - leapSeconds


def mjd(year, month, day, hour=0, minute=0, second=0):
    """
    Calculate the modified Julian date from an ordinary date and time.

    Notes
    -----
    The formulas are taken from Wikipedia.

    Example
    -------
    >>> mjd(2000, 1, 1)
    51544.0
    """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + \
        y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (hour - 12) / 24. + minute / 1400. + second / 86400.
    mjd = jd - 2400000.5
    return mjd


def UTCSecondToLocalDatetime(utcsec, timezone="Europe/Berlin"):
    """
    Convert utc second to local datetime (specify correct local timezone with string).

    Parameters
    ----------
    utcsec: int
      Time in utc seconds (unix time).

    timezone: string
      Time zone string compatible with pytz format

    Returns
    -------
      dt: Datetime object corresponding to local time.

    Notes
    -----
    Adapted from a stackoverflow answer.

    To list all available time zones:
    >> import pytz
    >> pytz.all_timezones

    To print the returned datetime object in a certain format:
    >> from pyik.time_conversion import UTCSecondToLocalDatetime
    >> dt = UTCSecondToLocalDatetime(1484314559)
    >> dt.strftime("%d/%m/%Y, %H:%M:%S")
    """

    import pytz
    from datetime import datetime

    local_tz = pytz.timezone(timezone)
    utc_dt = datetime.utcfromtimestamp(utcsec)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)

    return local_tz.normalize(local_dt)


def DatetimeToGPSSeconds(dt, timezone=None):
    """
    Convert a datetime expressed as python datetime object to (int) gps seconds.

    Parameters
    ----------
    dt: python datetime object

    timezone: string, timezone object or None (default)
      This is the time zone the datetime dt is specified in
      If None, assumes that dt is naive, i.e. represents UTC time

    Returns
    --------
      gps: time in gps seconds

    """
    import calendar
    import pytz
    from datetime import datetime

    if dt.tzinfo is not None:
        print("Warning! Your datetime is timezone-aware. This function assumes the timezone that is passed to the function and won't do a conversion between timezones!")

    if timezone is None:

        ts = calendar.timegm(dt.utctimetuple())

    else:

        if isinstance(timezone, str):
            tz = pytz.timezone(timezone)
        else:
            tz = timezone

        dt = dt.replace(tzinfo=tz)
        utc_naive = dt.replace(tzinfo=None) - dt.utcoffset()
        ts = (utc_naive - datetime(1970, 1, 1)).total_seconds()

    return int(utc_to_gps(ts))
