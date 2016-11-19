# coding=utf8
 
import sys
import datetime
import pytz
import numpy as np
import pandas as pd
import talib as ta
from scipy import stats

def similar(ndarr, hisDf, curDf, cor=0.9, onlyPositiveCorr=True):
    tmpDf=hisDf.loc[map(int, ndarr)]
    closeCor=stats.pearsonr(tmpDf['Close'].values, curDf['Close'].values)[0]
    volCor=stats.pearsonr(tmpDf['Volume'].values, curDf['Volume'].values)[0]
    if onlyPositiveCorr:
        return 1 if closeCor>cor and volCor>cor else 0
    else:
        if closeCor>cor and volCor>cor:
            return 1
        elif closeCor<-cor and volCor<-cor:
            return -1
        else:
            return 0
 
def predict(hisDf, lookAheadDays=3, windowSize=20, minCorr=0.9, onlyPositiveCorr=True):
    ecurDf=hisDf[-windowSize:]
    ehisDf=hisDf[:-windowSize]
    if pd.__version__ < '0.18':
        hisSim=pd.rolling_apply(ehisDf['index'], windowSize, similar, args=(ehisDf,ecurDf,minCorr,onlyPositiveCorr))
    else:
        hisSim=ehisDf['index'].rolling(center=False,window=windowSize).apply(func=similar, args=(ehisDf,ecurDf,minCorr,onlyPositiveCorr))
    hisSim=hisSim[hisSim.index<len(hisDf)-lookAheadDays]
    positiveSim=hisDf.iloc[hisSim[hisSim>0].index+lookAheadDays]['Close'].values/hisDf.iloc[hisSim[hisSim>0].index]['Close']-1
    if onlyPositiveCorr:
        return positiveSim.median()
    else:
        negtiveSim=hisDf.iloc[hisSim[hisSim<0].index+lookAheadDays]['Close'].values/hisDf.iloc[hisSim[hisSim<0].index]['Close']-1
        negtiveSim*=-1
        sim = pd.concat([positiveSim, negtiveSim])
        return sim.median()
 
def test(maDf, testSize=50, lookAheadDays=3, windowSize=20, minCorr=0.9, onlyPositiveCorr=True):
    right = 0.0
    unpredicable = 0.0
    for i in range(-testSize-lookAheadDays,-lookAheadDays,1):
        testDf=maDf[:i]
        predictedChg = predict(testDf, lookAheadDays, windowSize, minCorr, onlyPositiveCorr)
        length = len(maDf)
        realChg = maDf.at[length+i+lookAheadDays,'Close']/maDf.at[length+i,'Close'] -1
        dt = maDf.at[length+i, 'TradeDate']
        predictDt = maDf.at[length+i+lookAheadDays, 'TradeDate']
        print 'today:%s %s predict:%s %s predict chg:%s real chg:%s' % (dt,maDf.at[length+i,'Close'], predictDt,maDf.at[length+i+lookAheadDays,'Close'], predictedChg, realChg)
        if str(predictedChg) == 'nan' or predictedChg is np.nan:
            unpredicable += 1
        if predictedChg*realChg > 0:
            right += 1
    return unpredicable, right
 
 
def prepare_data(ticker, maMethod='ema', maPeriod=20, start='', end='', useYahoo=False):
    if not end:
        now = datetime.datetime.now()
        end = now.strftime('%Y%m%d')
    if not start:
        start = (datetime.datetime.now()+datetime.timedelta(days=-365*5)).strftime('%Y%m%d')
    if useYahoo:
        from zipline.utils.factory import load_bars_from_yahoo
        start = datetime.datetime.strptime(start, '%Y%m%d')
        start = datetime.datetime(start.year, start.month, start.day, 0,0,0,0,pytz.utc)
        end = datetime.datetime.strptime(end, '%Y%m%d')
        end = datetime.datetime(end.year, end.month, end.day, 0,0,0,0,pytz.utc)
        stockDf = load_bars_from_yahoo(stocks=[ticker], start=start, end=end, adjusted=False)[ticker].reset_index()
        stockDf['TradeDate'] = stockDf['Date'].apply(lambda x:x.strftime('%Y%m%d'))
        stockDf = stockDf[['open', 'high', 'low', 'close', 'volume','TradeDate']]
        stockDf.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
    else:
        sys.path.insert(0, '/home/jinwb/code/IIA/jsforesight/datamodel')
        from TickerEodModel import TickerEodModel
        eodM=TickerEodModel('testEventDbConfigKey')
        stockDf = eodM.get_eod(ticker,start,end)
    print stockDf.tail(10)
    emaDf=pd.DataFrame(index=stockDf.index)
    emaDf['Close']=stockDf['Close']
    emaDf['TradeDate']=stockDf['TradeDate']
    emaDf['index']=emaDf.index
    if maMethod.lower() == 'ema':
        emaDf['Close']=ta.EMA(stockDf['Close'].values, maPeriod)
        emaDf['Volume']=ta.EMA(stockDf['Volume'].values, maPeriod)
    else:
        emaDf['Close']=ta.MA(stockDf['Close'].values, maPeriod)
        emaDf['Volume']=ta.MA(stockDf['Volume'].values, maPeriod)
    return emaDf
 
 
if __name__ == '_main__':
    import argparse
    parser = argparse.ArgumentParser(description='predict/test using similarity-prediction')
    parser.add_argument('-t', '--ticker', action='store', default='000001.SZ', help='tickers to predict/test')
    parser.add_argument('-m', '--mamethod', action='store', choices=['ema','ma'], default='ema', help='ma method to pre-process the Close/Volume')
    parser.add_argument('-p', '--maperiod', action='store', type=int, default=20, help='period to ma Close/Volume')
    parser.add_argument('-w', '--window', action='store', type=int, default=20, help='window size to match')
    parser.add_argument('-a', '--lookahead', action='store', type=int, default=3, help='days to lookahead when predict')
    parser.add_argument('-c', '--mincorr', action='store', type=float, default=0.9, help='days to lookahead when predict')
    parser.add_argument('-s', '--testsize', action='store', type=int, default=50, help='period to test')
    parser.add_argument('-b', '--begin', action='store', type=str, default='19900101', help='start of the market data')
    parser.add_argument('-e', '--end', action='store', type=str, default='29900101', help='end of the market data')
    parser.add_argument('-o', '--onlypositivecorr', action='store_true', default=False)
    parser.add_argument('-u', '--usamarket', action='store_true', default=False)
    args = parser.parse_args()
 
    df = prepare_data(args.ticker, args.mamethod, args.maperiod, start=args.begin, end=args.end, useYahoo=args.usamarket)
    if args.testsize<=0:
        pred = predict(df, args.lookahead, args.window, args.mincorr, args.onlypositivecorr)
        print 'today:%s predict %s days later chg:%s' % (df.at[len(df)-1, 'TradeDate'], args.lookahead, pred)
    else:
        unpredicted, right = test(df, args.testsize, args.lookahead, args.window, args.mincorr, args.onlypositivecorr)
        print 'ticker:%s, ma period:%s, ma method:%s, testSize: %s predicts:%s, Right Rate: %s, Total Rate:%s' % (args.ticker, args.maperiod, args.mamethod, args.testsize,args.testsize-unpredicted, right/(args.testsize-unpredicted), right/args.testsize)
