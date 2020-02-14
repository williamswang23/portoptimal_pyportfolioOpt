"""author:williams_wang"""



##
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import quandl


quandl.ApiConfig.api_key="your API"

##
"""
input all stock names of US listed companies which you want to analyse
"""
stockname_input= ['AAPL','AMZN','GOOGL','FB','GM','AMD','MSFT','MCD','MMM','SHLD','WMT','INTC']


def stockpridownload (stockname):
    ic=((len(stockname))-1)
    df_port=pd.DataFrame()
    idx_sn=1
    pricex=quandl.get_table('WIKI/PRICES', ticker=stockname[0],
                             qopts={'columns': ['date', 'adj_close']},
                             date={'gte': '2017-1-1', 'lte': '2019-12-31'}, paginate=True)
    df_port=pd.concat([df_port,pricex],axis=1)
    for idx_sn in range(ic):
        pricex=quandl.get_table('WIKI/PRICES', ticker = stockname[idx_sn],
                        qopts = { 'columns': ['adj_close'] },
                        date={'gte': '2017-1-1', 'lte': '2019-12-31'}, paginate=True);
        pd.DataFrame(pricex);
        pprix=pricex['adj_close'];
        df_port=pd.concat([df_port,pprix],axis=1);
        idx_sn=idx_sn+1;

    df_port=df_port.set_index('date')
    df_port.columns=stockname

    return df_port


portdf=stockpridownload(stockname_input)
portdf.head()

##
def dict_print(df_dict):
    for key,value in df_dict.items():
        print('{key}:{value}'.format(key=key,value=value))
    return
##
"""
portdf.to_csv("files/date.csv")
"""


##

mu = expected_returns.mean_historical_return(portdf)
S = risk_models.sample_cov(portdf)
print(mu)
print(S)


##
"""strategy-1"""
"""max sharpe method optimal portfolio"""
ef_maxsp= EfficientFrontier(mu, S)
raw_weights = ef_maxsp.max_sharpe()
cleaned_weights = ef_maxsp.clean_weights()
"""ef.save_weights_to_file("/files/weights.csv")  # saves to file"""


print("the weights of max sharpe portfolio")
dict_print(cleaned_weights)
ef_maxsp.portfolio_performance(verbose=True)

##
"""strategy-2"""
# A long/short portfolio maximising return for a target volatility of 10%,
# with a shrunk covariance matrix risk model
shrink = risk_models.CovarianceShrinkage(portdf)
S = shrink.ledoit_wolf()
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
weights_ls = ef.efficient_risk(target_risk=0.10)
print("the weights of long/short strategy portfolio is ")
dict_print(weights_ls)
ef.portfolio_performance(verbose=True)
##
"""strategy-3"""
# A market-neutral Markowitz portfolio finding the minimum volatility
# for a target return of 20%
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
weights_mv = ef.efficient_return(target_return=0.20, market_neutral=True)
print("the weights of minimum volatility portfolio is ")
dict_print(weights_mv)
ef.portfolio_performance(verbose=True)
##

