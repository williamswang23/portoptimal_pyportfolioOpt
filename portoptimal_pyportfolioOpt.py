##
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import quandl
quandl.ApiConfig.api_key="Q4uQPxfNSDW28sSj-5V9"

##
"""
put into each stock name of US market which you want to analyse
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
portdf.to_csv("/Users/wangjiatao/Documents/Project_ww/optimal_port_G/date.csv")



##

mu = expected_returns.mean_historical_return(portdf)
S = risk_models.sample_cov(portdf)

"""max sharpe method optimal portfolio"""
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("/Users/wangjiatao/Documents/Project_ww/optimal_port_G/weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


