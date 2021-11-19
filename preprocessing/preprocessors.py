import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df,extra_ind):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())
    #print("printing stock object:")
    #print("stock: " + str(stock))
    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()
    mom = pd.DataFrame()
    if (extra_ind):
        boll = pd.DataFrame()
        sma = pd.DataFrame()
        ema = pd.DataFrame()
        mstd = pd.DataFrame()
        


    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)
        if extra_ind:
            ## bollinger band
            temp_bb = stock[stock.tic == unique_ticker[i]]['boll']
            temp_bb = pd.DataFrame(temp_bb)
            bb = boll.append(temp_bb, ignore_index=True)
            ## Simple Moving Average
            temp_sma = stock[stock.tic == unique_ticker[i]]['open_2_sma']
            temp_sma = pd.DataFrame(temp_sma)
            sma = sma.append(temp_sma, ignore_index=True)
            ## Exponential Moving Average
            temp_ema = stock[stock.tic == unique_ticker[i]]['open_2_ema']
            temp_ema = pd.DataFrame(temp_ema)
            ema = ema.append(temp_ema, ignore_index=True)
            ## moving standard deviation
            temp_mstd = stock[stock.tic == unique_ticker[i]]['open_2_mstd']
            temp_mstd = pd.DataFrame(temp_mstd)
            mstd = mstd.append(temp_mstd, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx
    if extra_ind:
        #df['boll'] = bb
        df['sma'] = sma
        df['ema'] = ema
        df['mstd'] = mstd
    return df



def preprocess_data(small,no_ind,extra_ind):
    """data preprocessing pipeline"""

    print("loading dataset " + str(config.TRAINING_DATA_FILE))
    if (small):
        print("using smaller dataset")
        df = load_dataset(file_name=config.SMALL_TRAINING_DATA_FILE)
    else:
        df = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # get data after 2009
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final=df_preprocess
    if not no_ind:
        df_final=add_technical_indicator(df_preprocess,extra_ind)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df


def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index

def add_systemic_risk(df):
    """
    add systemic risk index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    systemic_risk_index = calcualte_systemic_risk(df)
    df = df.merge(systemic_risk_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_systemic_risk(df):
    """calculate systemic risk index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    
    ###
    print(df_price_pivot)
    
    # start after a year
    ### SHORTER TIME WINDOW MAY BE MORE RESPONSIVE
    ### HYPERPARAMETER #1
    start = 252
    systemic_risk_index = [0]*start
    #systemic_risk_index = [0]
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[(i-start):i] for n in df_price_pivot.index ]]
        ### Use the past 252 days of price history to calculate systemic risk
        
        cov_temp = hist_price.cov() ### THIS IS THE COVARIANCE MATRIX
        
        ### Resources:
        # https://stats.stackexchange.com/questions/346692/how-does-eigenvalues-measure-variance-along-the-principal-components-in-pca
        # https://stackoverflow.com/questions/54538232/finding-eigenvalues-of-covariance-matrix?rq=1
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.htmlhttps://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
        
        eigenvals_temp, eigenvects_trash = np.linalg.eig(cov_temp)
        # The eigenvalues are automatically ordered from largest to smallest
        
        ### HYPERPARAMETER #2
        # Number of eigenvalues to use in the absorption ratio is heuristically 1/5 of the number of assets (Kritzman et al 2010)
        n_eigs = 6 # For DJIA30
        
        systemic_risk_temp = sum(eigenvals_temp[0:(n_eigs-1)])/sum(eigenvals_temp)    
        
        systemic_risk_index.append(systemic_risk_temp)
    
    
    systemic_risk_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'systemic_risk':systemic_risk_index})
    return systemic_risk_index








