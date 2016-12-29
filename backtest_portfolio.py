#coding=utf8
import datetime
import numpy as np
import pandas as pd
import talib as ta
from zipline.api import order, record, symbol
from zipline.utils.factory import load_bars_from_yahoo

def prepare_data(tickers, start='', end=''):
    assert isinstance(tickers, list)
    if not end:
        now = datetime.datetime.now()
        end = now.strftime('%Y%m%d')
    if not start:
        start = (datetime.datetime.strptime(end, '%Y%m%d')+datetime.timedelta(days=-365*5)).strftime('%Y%m%d')
    tmpStart = datetime.datetime.strptime(start, '%Y%m%d')
    ystart = datetime.datetime(tmpStart.year, tmpStart.month, tmpStart.day, 0, 0, 0 ,0, pytz.utc)
    tmpEnd = datetime.datetime.strptime(end, '%Y%m%d')
    yend = datetime.datetime(tmpEnd.year, tmpEnd.month, tmpEnd.day, 0, 0, 0, 0, pytz.utc)
    stockDf = load_bars_from_yahoo(stocks=tickers, start=ystart, end=yend, adjusted=True)
    return stockDf.dropna()

def predict(inputHisDf, windowSize=20):
    closeSeries = inputHisDf.tail(windowSize)['close']
    std = closeSeries.std()
    stdRate = std/closeSeries.mean()
    if std < 0.05 or std > 50 or stdRate < 0.005 or stdRate > 0.12:
        return 0
    trendWindow = 10
    tmpStdGap = stdRate*100

    stdGap = 1.25 if tmpStdGap < 1 else 1.5 if tmpStdGap < 1.5 else 2
    hisDf = inputHisDf.tail(windowSize+1)
    pU, pM, pL = ta.BBANDS(hisDf['close'].head(windowSize).astype(float).values, timeperiod=trendWindow, nbdevup=stdGap, nbdevdn=stdGap)
    volU, volM, volL = ta.BBANDS(hisDf['volume'].head(windowSize).astype(float).values, timeperiod=trendWindow, nbdevup=stdGap, nbdevdn=stdGap)
    preP = hisDf['close'].iat[-2]
    curP = hisDf['close'].iat[-1]

    preV = hisDf['volume'].iat[-2]
    curV = hisDf['volume'].iat[-1]

    pUSlope = _array_slope(pU[-trendWindow:])
    pMSlope = _array_slope(pM[-trendWindow:])
    volUSlope = _array_slope(volU[-trendWindow:])
    volMSlope = _array_slope(volM[-trendWindow:])
    if volMSlope > 0: #goes upper with larger std
        if curP > pL[-1] and preP < pL[-1]:
            return 1
    if curP < pU[-1] and preP > pU[-1]:
        return -1
    '''
    if pUSlope > 0 and pMSlope > 0 and pUSlope-pMSlope > 0: #goes upper with larger std
        if curP > pU[-1] and preP < pU[-1]:
            return 1
        elif curP < pU[-1] and preP > pU[-1]:
            return -1
        elif curP < pL[-1] and preP > pL[-1]:
            return -1
        elif curP > pL[-1] and preP < pL[-1]:
            return 1

    elif pUSlope < 0 and pMSlope < 0 and pUSlope-pMSlope < 0: #goes down with small std
        if curP > pL[-1] and preP < pL[-1]:
            return 1
        elif curP < pL[-1] and preP > pL[-1]:
            return -1
    if volUSlope > 0 and volMSlope > 0 and volUSlope-volMSlope > 0:
        if curP > pL[-1] and preP < pL[-1]:
            return 1
        if curP > pM[-1] and pMSlope > 0:
            return 1
        if curP < pU[-1] and preP > pU[-1]:
            return -1
        if curP < pM[-1] and pMSlope < 0:
            return -1
    elif volUSlope < 0 and volMSlope < 0 and volUSlope-volMSlope < 0:
        if curP < pU[-1] and preP > pU[-1]:
            return -1
    '''
    return 0

def _array_slope(series):
    if isinstance(series, list) or isinstance(series, np.ndarray):
        series = pd.Series(series)
    assert isinstance(series, pd.Series)
    X = pd.Series(range(len(series)))
    return X.corr(series)



def initialize(context, xdata=None, xticker=None, xstart='20130101', xend='20161207', window=20, mincorr=0.9):
    context.ticker = xticker
    context.i = 0
    context.window = window
    context.mincorr = mincorr
    context.pos = dict(zip(context.ticker, [0]*len(context.ticker)))
    context.pos_bar = dict(zip(context.ticker, [0]*len(context.ticker)))
    context.pos_price = dict(zip(context.ticker, [0]*len(context.ticker)))
    context.max_profit = 0.1
    context.max_loss = 0.05
    context.max_hold = 3
    context.max_position_rate = 0.9
    if xdata is not None:
        context.data = xdata
    else:
        context.data = prepare_data(context.ticker, start=xstart, end=xend)

def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 500:
        return

    for t in context.ticker:
        handle_data_func(context, t, data)

def handle_data_func(context, t, data):
    curSym = symbol(t)
    df = context.data[t]
    df = df[df.index <= np.datetime64(data.current_dt)]
    pred = predict(df, windowSize=context.window)
    curLot = int(context.portfolio.starting_cash*context.max_position_rate/len(context.ticker)/data.current(curSym, 'price'))
    if pred > 0:
        if context.pos[t] <= 0:
            order(curSym, curLot)
            context.pos[t] += curLot
            context.pos_bar[t] = context.i
            context.pos_price[t] = data.current(curSym, 'price')
    elif pred < 0:
        if context.pos[t] > 0:
            order(curSym, -1*context.pos[t])
            context.pos[t] = 0
            context.pos_bar[t] = 0
            context.pos_price[t] = 0
    else:
        if context.pos[t] > 0:
            if context.i - context.pos_bar[t] > context.max_hold:
                order(curSym, -1*context.pos[t])
                context.pos[t] = 0
                context.pos_bar[t] = 0
                context.pos_price[t] = 0
        if context.pos[t] > 0:
            ret = data.current(curSym, 'high')/context.pos_price[t] -1
            if ret > context.max_profit or ret < - context.max_loss:
                order(curSym, -1*context.pos[t])
                context.pos[t] = 0
                context.pos_bar[t] = 0
                context.pos_price[t] = 0

    # Save values for later inspection
    kargs = {t:data.current(curSym, "price"),
            'pred':pred*5,
            'pos':context.pos,
            }
    record(**kargs)

# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(results=None, symbol=None):
    import matplotlib.pyplot as plt
    import logbook
    logbook.StderrHandler().push_application()
    log = logbook.Logger('Algorithm')

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')

    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('Price (USD)')

    # If data has been record()ed, then plot it.
    # Otherwise, log the fact that no data has been recorded.
    if (symbol in results):
        results[symbol].plot(ax=ax2)
        #results[['short_mavg', 'long_mavg']].plot(ax=ax2)

        trans = results.ix[[t != [] for t in results.transactions]]
        buys = trans.ix[[t[0]['amount'] > 0 for t in
                         trans.transactions]]
        sells = trans.ix[
            [t[0]['amount'] < 0 for t in trans.transactions]]
        #ax2.plot(buys.index, results.short_mavg.ix[buys.index], '^', markersize=10, color='m')
        #ax2.plot(sells.index, results.short_mavg.ix[sells.index], 'v', markersize=10, color='k')
        results.pred.plot(ax=ax2)
        #results.pos.plot(ax=ax2)
        ax2.plot(buys.index, results['pred'].ix[buys.index], '^', markersize=10, color='r')
        ax2.plot(sells.index, results['pred'].ix[sells.index], 'v', markersize=10, color='g')
        plt.legend(loc=0)
        print results
    else:
        msg = 'short_mavg & long_mavg data not captured using record().'
        ax2.annotate(msg, xy=(0.1, 0.5))
        log.info(msg)

    plt.show()


if __name__ =='__main__':
    import sys
    import pytz
    import matplotlib.pyplot as plt
    from zipline import TradingAlgorithm
    from zipline.utils.factory import load_from_yahoo

    import argparse
    parser = argparse.ArgumentParser(description='predict/test using similarity-prediction')
    parser.add_argument('-t', '--ticker', action='store', default='AAPL', help='tickers to predict/test')
    parser.add_argument('-m', '--mamethod', action='store', choices=['ema','ma'], default='ema', help='ma method to pre-process the Close/Volume')
    parser.add_argument('-p', '--maperiod', action='store', type=int, default=20, help='period to ma Close/Volume')
    parser.add_argument('-w', '--window', action='store', type=int, default=20, help='window size to match')
    parser.add_argument('-a', '--lookahead', action='store', type=int, default=1, help='days to lookahead when predict')
    parser.add_argument('-c', '--mincorr', action='store', type=float, default=0.9, help='days to lookahead when predict')
    parser.add_argument('-b', '--begin', action='store', type=str, default='20100101', help='start of the market data')
    parser.add_argument('-e', '--end', action='store', type=str, default='20161221', help='end of the market data')
    args = parser.parse_args()

    #start = datetime.datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
    #end = datetime.datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
    #data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start, end=end, adjusted=False)
    tickers = [t.strip() for t in args.ticker.split(',') if t.strip()]

    data = prepare_data(tickers, start=args.begin, end=args.end)

    algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data, capital_base=50000, xdata=data, xticker=tickers, xstart=args.begin, xend=args.end, window=args.window)
    res = algo.run(data).dropna()
    analyze(res, tickers[0])
