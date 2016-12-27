#coding=utf8
import datetime
import pandas as pd
import talib as ta
from zipline.api import order, record, symbol
from auto_regression_predictor import predict

def prepare_data(ticker, maMethod='ema', maPeriod=20, lookAheadDays=3, start='', end='', useYahoo=True):
    if not end:
        now = datetime.datetime.now()
        end = now.strftime('%Y%m%d')
    if not start:
        start = (datetime.datetime.strptime(end, '%Y%m%d')+datetime.timedelta(days=-365*5)).strftime('%Y%m%d')
    if useYahoo:
        from zipline.utils.factory import load_bars_from_yahoo
        tmpStart = datetime.datetime.strptime(start, '%Y%m%d')
        ystart = datetime.datetime(tmpStart.year, tmpStart.month, tmpStart.day, 0,0,0,0,pytz.utc)
        tmpEnd = datetime.datetime.strptime(end, '%Y%m%d')
        yend = datetime.datetime(tmpEnd.year, tmpEnd.month, tmpEnd.day, 0,0,0,0,pytz.utc)
        stockDf = load_bars_from_yahoo(stocks=[ticker], start=ystart, end=yend, adjusted=True)[ticker].reset_index()
        stockDf['TradeDate'] = stockDf['Date'].apply(lambda x:x.strftime('%Y%m%d'))
        stockDf = stockDf[['open', 'high', 'low', 'close', 'volume','TradeDate']]
        stockDf.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
    else:
        sys.path.insert(0, '/home/jinwb/code/IIA/jsforesight/datamodel')
        from TickerEodModel import TickerEodModel
        eodM=TickerEodModel('testEventDbConfigKey')
        stockDf = eodM.get_eod(ticker,start,end)
    #emaDf=pd.DataFrame(index=pd.DatetimeIndex(stockDf['TradeDate'], tz=pytz.utc))
    emaDf = pd.DataFrame()
    emaDf['OrigClose']=stockDf['Close']
    emaDf['OrigVolume']=stockDf['Volume']
    emaDf['TradeDate']=stockDf['TradeDate']
    emaDf['PctChg'] = stockDf['Close'].pct_change(periods=lookAheadDays)
    emaDf['index'] = stockDf.index
    if maMethod.lower() == 'ema':
        emaDf['Close']=ta.EMA(stockDf['Close'].values, maPeriod)
        emaDf['Volume']=ta.EMA(stockDf['Volume'].values, maPeriod)
    else:
        emaDf['Close']=ta.MA(stockDf['Close'].values, maPeriod)
        emaDf['Volume']=ta.MA(stockDf['Volume'].values, maPeriod)
    emaDf['open'] = stockDf['Open']
    emaDf['high'] = stockDf['High']
    emaDf['low'] = stockDf['Low']
    emaDf['close'] = stockDf['Close']
    emaDf['volume'] = stockDf['Volume']
    emaDf.set_index(pd.DatetimeIndex(emaDf['TradeDate'], tz=pytz.utc), inplace=True)
    res = pd.Panel({ticker:emaDf.dropna()})
    return res


def initialize(context, xdata=None, xticker=None, xstart='20130101', xend='20161207', lookahead=1, window=5, mincorr=0.9, onlypositivecorr=True):
    context.ticker = xticker
    context.sym = symbol(xticker)
    context.i = 0
    context.lookahead = lookahead
    context.window = window
    context.mincorr = mincorr
    context.onlypositivecorr = onlypositivecorr
    context.pos = 0
    context.pos_bar = 0
    context.pos_price = 0
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

    df = context.data[context.ticker]
    df = df[df['TradeDate']<data.current_dt.strftime('%Y%m%d')]
    pred = predict(df, context.lookahead, context.window, context.mincorr, context.onlypositivecorr)
    curLot = int(context.portfolio.starting_cash*context.max_position_rate/data.current(context.sym, 'price'))
    if pred > 0:
        if context.pos <= 0:
            order(context.sym, curLot)
            context.pos += curLot
            context.pos_bar = context.i
            context.pos_price = data.current(context.sym, 'price')
    elif pred < 0:
        if context.pos > 0:
            order(context.sym, -1*context.pos)
            context.pos = 0
            context.pos_bar = 0
            context.pos_price = 0
    else:
        if context.pos > 0:
            if context.i - context.pos_bar > context.max_hold:
                order(context.sym, -1*context.pos)
                context.pos = 0
                context.pos_bar = 0
                context.pos_price = 0
        if context.pos > 0:
            ret = data.current(context.sym, 'high')/context.pos_price -1
            if ret > context.max_profit or ret < - context.max_loss:
                order(context.sym, -1*context.pos)
                context.pos = 0
                context.pos_bar = 0
                context.pos_price = 0

    # Save values for later inspection
    kargs = {context.ticker:data.current(context.sym, "price"),
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
    parser.add_argument('-s', '--testsize', action='store', type=int, default=50, help='period to test')
    parser.add_argument('-b', '--begin', action='store', type=str, default='20100101', help='start of the market data')
    parser.add_argument('-e', '--end', action='store', type=str, default='20161221', help='end of the market data')
    parser.add_argument('-o', '--onlypositivecorr', action='store_true', default=False)
    parser.add_argument('-u', '--usamarket', action='store_true', default=True)
    args = parser.parse_args()

    #start = datetime.datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
    #end = datetime.datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
    #data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start, end=end, adjusted=False)

    data = prepare_data(args.ticker, args.mamethod, args.maperiod, lookAheadDays=args.lookahead, start=args.begin, end=args.end, useYahoo=args.usamarket)

    algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data, capital_base=50000, xdata=data, xticker=args.ticker, xstart=args.begin, xend=args.end, lookahead=args.lookahead, window=args.window, mincorr=args.mincorr, onlypositivecorr=args.onlypositivecorr)
    res = algo.run(data).dropna()
    analyze(res, args.ticker)
