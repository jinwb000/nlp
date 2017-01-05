from datetime import time
from itertools import chain

from pandas.tslib import Timestamp
from pytz import timezone

from zipline.utils.calendars import TradingCalendar
from zipline.utils.calendars.trading_calendar import HolidayCalendar
from zipline.utils.calendars.us_holidays import (
    USNewYearsDay,
    )


class SSEExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for Shanghai Stock Exchange.

    Open Time: 9:30am, Asia/Shanghai
    Close Time: 3:00pm, Asia/Shanghai

    https://www.theice.com/publicdocs/futures_us/ICE_Futures_US_Regular_Trading_Hours.pdf # noqa
    """
    @property
    def name(self):
        return "SSE"

    @property
    def tz(self):
        return timezone("Asia/Shanghai")

    @property
    def open_time(self):
        return time(9, 30)

    @property
    def close_time(self):
        return time(15)

    @property
    def open_offset(self):
        return -1

    @property
    def adhoc_holidays(self):
        return list(chain(
            [Timestamp('2012-10-29', tz='UTC')]
        ))

    @property
    def regular_holidays(self):
        # https://www.theice.com/publicdocs/futures_us/exchange_notices/NewExNot2016Holidays.pdf # noqa
        return HolidayCalendar([
            USNewYearsDay,
        ])
