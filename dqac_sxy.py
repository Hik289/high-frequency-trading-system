from  dqt2 import make_ic, get_group_pnls, make_long_short, make_weighted_ic
import pandas as pd
import numpy as np
import torch
from copy import deepcopy as copy

#############################################################################################
##operators
#############################################################################################
#rank
def RANK(df, device = 'cpu'):
    '''
    calc the rank of cross section data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)rank result, cross_section result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.argsort(torch.argsort(temp))/ columns.shape[0]
    torch.cuda.empty_cache()
        
    return results, index, columns

#delta
def DELTA(df, rolling_date, device = 'cpu'):
    '''
    calc the delta(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
    @return:tuple
        ((1)delta result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date+1
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    rw_t = torch.FloatTensor(rw).to(calc_device)
    
    results = rw_t[:,window - 1,:]-rw_t[:,0,:]   
    offset = torch.full((rolling_date, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#max
def MAX(df, device = 'cpu'):
    '''
    calculate the maximum of cross section data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)maximum result, cross_section result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    temp[torch.isnan(temp)] = -torch.inf

    results= torch.tile(torch.max(temp,axis = 1)[0],(temp.shape[1],1)).transpose(0,1)
    torch.cuda.empty_cache()
        
    return results, index, columns

#min
def MIN(df, device = 'cpu'):
    '''
    calculate the minimum of cross section data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1) minimum result, cross_section result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    temp[torch.isnan(temp)] = torch.inf

    results= torch.tile(torch.min(temp,axis = 1)[0],(temp.shape[1],1)).transpose(0,1)
    torch.cuda.empty_cache()
        
    return results, index, columns

#if
def IF(condition, x, y, device = 'cpu'):
    '''
    compare the condition of x and y, return the condition anywhere,
    if condition ,return x, else return y, anywhere satisfy the condition
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        condition:bool dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)condition result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(condition) == pd.core.frame.DataFrame:
        columns = condition.columns
        condition_index = condition.index
        condition_values = condition.values
    elif type(condition) == tuple:
        columns = condition[2]
        condition_index = condition[1]
        condition_values = condition[0].cpu().numpy()
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('x and y index not the same!')
    else:
        index = x_index
        
    if (condition_index != x_index).any():
        raise('x and condition index not the same!')
    else:
        index = x_index
        
    if (y_index != condition_index).any():
        raise('y and condition index not the same!')
    else:
        index = x_index
        
    temp_x = torch.FloatTensor(x_values).to(device)
    temp_y = torch.FloatTensor(y_values).to(device)
    temp_condition = torch.BoolTensor(condition_values).to(device)    
    results = torch.full(temp_x.shape, torch.nan)
    results[temp_condition] = temp_x[temp_condition]
    results[~temp_condition] = temp_y[~temp_condition]
    
    torch.cuda.empty_cache()    
    return results, index, columns    

#sum
def SUM(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the sum(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)sum result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.sum(temp, axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#mean
def MEAN(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the mean(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)mean result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.mean(temp, axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#std
def STD(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the standard deviaiotn (rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)std result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.std(temp, axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    if window > 1:
        results = results*np.sqrt(window*(window-1))/(window-1)
    else:
        results = results*0
    '''
        as we checked, dataframe.std() is different from torch.std(), 
        so we apply compensation constant
    '''
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#diff
def DIFF(df, rolling_date, device = 'cpu'):
    '''
    calculate the difference(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
    @return:tuple
        ((1)difference result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date+1
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    rw_t = torch.FloatTensor(rw).to(calc_device)
    
    results = rw_t[:,window - 1,:]-rw_t[:,0,:]   
    offset = torch.full((rolling_date, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#median
def MEDIAN(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the median (rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)median result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.median(temp, axis = 1)[0]
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#tsrank
def TSRANK(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the tsrank (rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)tsrank result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.argsort(torch.argsort(temp,axis = 1),axis = 1)[:,window-1,:]
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    if window > 1:
        results = results*2/(window-1) -1
    else:
        results = results*0
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#sign
def SIGN(df, device = 'cpu'):
    '''
    calculate the sign value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)sign result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.sign(temp)
    torch.cuda.empty_cache()
        
    return results, index, columns

#kurt
def KURT(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the kurtosis(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:4, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)kurtosis result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    
    temp_mean = torch.mean(temp, axis = 1)
    temp_var = torch.var(temp, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    results = (torch.mean((temp.transpose(0,1)-temp_mean)**4,axis = 0)/temp_var**2)
    
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    if window > 3:
        results = results*(window**2-1)/(window-2)/\
            (window-3)-3*(window-1)**2/(window-2)/(window-3)
    else:
        results = results*0
    '''
        as we checked, dataframe.kurt() is unbiased kurtosis using Fisher's definition
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1, 
        so we apply compensation constant, and +3 for fisher model.
    '''
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns   

#skew
def SKEW(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the skewness(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:3, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)skewness result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    
    temp_mean = torch.mean(temp, axis = 1)
    temp_var = torch.var(temp, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    results = (torch.mean((temp.transpose(0,1)-temp_mean)**3,axis = 0)/temp_var**(3/2))
    
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
        
    if window > 2:
        results = results*((window-1)*window)**0.5 / (window-2)
    else:
        results = results*0
    '''
        as we checked, dataframe.skew() is unbiased skewness 
        skewness (skewness of normal == 0.0). Normalized by N-1, 
        so we apply compensation constant
    '''
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#tsmin
def TSMIN(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the time series minimum (rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)minimum result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.min(temp, axis = 1)[0]
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#tsmax
def TSMAX(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the time series maximum (rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)amximum result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.max(temp, axis = 1)[0]
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#tot
def TOT(df, rolling_date, device = 'cpu'):
    '''
    calculate the change rate(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
    @return:tuple
        ((1)change rate TOT result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date+1
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    rw_t = torch.FloatTensor(rw).to(calc_device)
    
    results = (rw_t[:,window - 1,:]-rw_t[:,0,:])/ rw_t[:,0,:]  
    offset = torch.full((rolling_date, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#sigmoid
def SIGMOID(df, device = 'cpu'):
    '''
    calculate the sigmoid value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)sigmoid result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.sigmoid(temp)
    torch.cuda.empty_cache()
        
    return results, index, columns

#log
def LOG(df, device = 'cpu'):
    '''
    calculate the log value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)log result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.log(temp)
    torch.cuda.empty_cache()
        
    return results, index, columns

#abs
def ABS(df, device = 'cpu'):
    '''
    calculate the absolute value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)abs result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.abs(temp)
    torch.cuda.empty_cache()
        
    return results, index, columns

#sqrt
def SQRT(df, device = 'cpu'):
    '''
    calculate the sqrt value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)sqrt result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.sqrt(temp)
    torch.cuda.empty_cache()
        
    return results, index, columns

#lowday
def LOWDAY(df, rolling_date, min_period = None, pct = False, device = 'cpu'):
    '''
    calculate the time series minimum index(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
        pct: True if nomalize to (0,1)
    @return:tuple
        ((1)minimum index result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.argmin(temp, axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    if pct == True:
        results = results / window
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns     

#highday
def HIGHDAY(df, rolling_date, min_period = None, pct = False, device = 'cpu'):
    '''
    calculate the time series maximum index(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
        pct: True if nomalize to (0,1)
    @return:tuple
        ((1)maximum index result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.argmax(temp, axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan

    if pct == True:
        results = results / window
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#maxminnorm
def MAXMINNORM(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the time series maxmin normalization(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)maxmin normalization result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    result = torch.FloatTensor(values[window-1:]).to(device)
    results = (result- torch.min(temp,axis = 1)[0])/\
            (torch.max(temp, axis = 1)[0]-torch.min(temp, axis = 1)[0])
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#prod
def PROD(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the time series prod(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window

    @return:tuple
        ((1)prod index result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.prod(temp, axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#power
def POWER(df, n, device = 'cpu'):
    '''
    Function: compress dataframe data by row
    high frequncy operator, rolling_date can be seconds or minutes
    power(n) formula:

    .. math::
        \alpha = sign(rank\_i(\alpha)-0.5) * |{rank\_i(\alpha)-0.5|}^n
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        n: int
            The degree of data compression. 
            The greater the N, the greater the degree of compression
            The value of N cannot be 1. When n = 1, 
            the data will be close to uniform distribution after conversion
            
    @return:tuple
        ((1)rank result, cross_section result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    @Examples
    --------
    >>> TradingDay = Date().TraDate('2016-01-01', '2016-01-10')['D']
    >>> df = pd.DataFrame(np.random.randn(5,4), index = TradingDay, 
    ...              columns = ['000001', '000002', '000004', '000005'])
    >>> of = OperatorFun()
    >>> data = of.power_i(df, 2)
    >>> data
                    000001  000002  000004  000005
        TradingDay                                
        2016-01-04  0.0000  0.2500  0.0625 -0.0625
        2016-01-05  0.0000  0.2500  0.0625 -0.0625
        2016-01-06  0.2500 -0.0625  0.0625  0.0000
        2016-01-07  0.0000 -0.0625  0.0625  0.2500
        2016-01-08 -0.0625  0.0000  0.2500  0.0625
        
    '''
    
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    ranks = torch.argsort(torch.argsort(temp))/ columns.shape[0]
    results = torch.sign(ranks-0.5)*(torch.abs(ranks-0.5))**n
    
    torch.cuda.empty_cache()
    
    return results, index, columns  

#corr
def CORR(x, y, rolling_date, na_threshold=True, device = 'cpu'):
    '''
    calculate the time series correlation of x and y
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
        na_threshold: True, if there are more than 2/3 of x or y are Nan, 
            the corr value is Nan.
    @return:tuple
        ((1)correlation time series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    
    temp_x[torch.isnan(temp_x)] = 0
    temp_y[torch.isnan(temp_y)] = 0
    temp_mean_x = torch.mean(temp_x, axis = 1)
    temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    results = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                             (temp_y.transpose(0,1)-temp_mean_y),axis = 0)/\
                   temp_var_x**0.5/ temp_var_y**0.5)
    
    if na_threshold == True:
        results[torch.sum(torch.isnan(temp_x),axis = 1)> 2/3* window] = torch.nan
        results[torch.sum(torch.isnan(temp_y),axis = 1)> 2/3* window] = torch.nan  
        
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#cov
def COV(x, y, rolling_date, na_threshold=True, device = 'cpu'):
    '''
    calculate the time series coviarance of x and y
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        na_threshold: True, if there are more than 2/3 of x or y are Nan, 
            the cov value is Nan.
    @return:tuple
        ((1)coviarance time series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    
    temp_x[torch.isnan(temp_x)] = 0
    temp_y[torch.isnan(temp_y)] = 0
    temp_mean_x = torch.mean(temp_x, axis = 1)
    #temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    #temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    results = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                             (temp_y.transpose(0,1)-temp_mean_y),axis = 0))
        
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#zscore
def ZSCORE(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calculate the time series zscore normalization(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)zscore normalization result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    result = torch.FloatTensor(values[window-1:]).to(device)
    
    temp_var = torch.var(temp, axis = 1)
    results = (result- torch.mean(temp,axis = 1))/temp_var**0.5
    '''
        as we checked, torch.var() is biased variance,
        but dataframe zscore result is also biased different from df.std(),
        so we delete compensation constant.
    '''
    
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#scale
def SCALE(df, a = 1,device = 'cpu'):
    '''
    calculate the scale of cross section data, the sumation is a
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        a: scale total values
    @return:tuple
        ((1)scale result, cross_section result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    temp[torch.isnan(temp)] = 0
    results = (temp.T/ torch.sum(temp, axis = 1)*a).T
    torch.cuda.empty_cache()
        
    return results, index, columns
    
#signedpower
def SIGNEDPOWER(df, a = 2, device = 'cpu'):
    '''
    calculate the signed power value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        a: power exponent
    @return:tuple
        ((1)signed power result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.sign(temp)*(torch.abs(temp))**a
    torch.cuda.empty_cache()
        
    return results, index, columns
    
#quantile
def QUANTILE(df, rolling_date, p, min_period = None, device = 'cpu'):
    '''
    calculate the time series quantile(rolling_date) of time series data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
        p: a quantile number between (0,1)
    @return:tuple
        ((1)quantile p result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.quantile(temp, q = p,axis = 1)
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#slog
def SLOG(df, device = 'cpu'):
    '''
    calculate the signed log value of all data
    high frequncy operator, rolling_date can be seconds or minutes
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)signed log result, torch.FloatTensor, placed on device;
         (2) or df.index;
         (3) or df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(calc_device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    results = torch.sign(temp)*torch.log(temp)
    torch.cuda.empty_cache()
        
    return results, index, columns
    
#tsbeta
def TSBETA(x, y, rolling_date, device = 'cpu'):
    '''
    calculate the time series OLS beta of x and y
    belong to TSOLS category
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
    @return:tuple
        ((1)OLS beta time series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    

    temp_mean_x = torch.mean(temp_x, axis = 1)
    temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    #temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    beta = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                    (temp_y.transpose(0,1)-temp_mean_y),axis = 0))/\
                    (temp_var_x)   
    results = beta
        
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#tscons
def TSCONS(x, y, rolling_date, device = 'cpu'):
    '''
    calculate the time series OLS constant of x and y
    belong to TSOLS category
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
    @return:tuple
        ((1)OLS constant time series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    

    temp_mean_x = torch.mean(temp_x, axis = 1)
    temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    #temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    beta = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                    (temp_y.transpose(0,1)-temp_mean_y),axis = 0))/\
                    (temp_var_x)   
    constant = temp_mean_y - beta*temp_mean_x
    
    results = constant
        
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns
        
#tsresstd
def TSRESSTD(x, y, rolling_date, device = 'cpu'):
    '''
    calculate the time series OLS residual std of x and y
    belong to TSOLS category
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows

    @return:tuple
        ((1)OLS residual std time series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    

    temp_mean_x = torch.mean(temp_x, axis = 1)
    temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    #temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    beta = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                    (temp_y.transpose(0,1)-temp_mean_y),axis = 0))/\
                    (temp_var_x)   
    constant = temp_mean_y - beta*temp_mean_x
      
    residual = temp_y.transpose(0,1) - beta*temp_x.transpose(0,1) - constant
    
    results = torch.std(residual, axis = 0)
    
    if window > 1:
        results = results*np.sqrt(window*(window-1))/(window)
    else:
        results = results*0
    '''
        as we checked, dataframe.std() is different from torch.std(), 
        so we apply compensation constant
    '''
        
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#tsreskew
def TSRESKEW(x, y, rolling_date, device = 'cpu'):
    '''
    calculate the time series OLS residual skew of x and y
    the skew of OLS residual is 0, so we take the last residual as skew values
    belong to TSOLS category
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:4, maximum rolling_date:df rows

    @return:tuple
        ((1)OLS residual skew time series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    

    temp_mean_x = torch.mean(temp_x, axis = 1)
    temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    #temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    beta = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                    (temp_y.transpose(0,1)-temp_mean_y),axis = 0))/\
                    (temp_var_x)   
    constant = temp_mean_y - beta*temp_mean_x
      
    residual = temp_y.transpose(0,1) - beta*temp_x.transpose(0,1) - constant
    
    results = residual[window-1,:,:]

    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    '''
        second step, calculate skew
    '''
    results = results.cpu().numpy()
    torch.cuda.empty_cache()
    
    rw_s = np.lib.stride_tricks.as_strided( \
        x=results,shape=(results.shape[0]-window+1,window,results.shape[1]),\
        strides=(results.strides[0],results.strides[0],results.strides[1])) 
    
    temp = torch.FloatTensor(rw_s).to(calc_device)
    
    temp_mean = torch.mean(temp, axis = 1)
    temp_var = torch.var(temp, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    results = (torch.mean((temp.transpose(0,1)-temp_mean)**4,axis = 0)/temp_var**2)
    
    if window > 3:
        results = results*(window**2-1)/(window-2)/\
            (window-3)-3*(window-1)**2/(window-2)/(window-3)
    else:
        results = results*0
    '''
        as we checked, dataframe.kurt() is unbiased kurtosis using Fisher's definition
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1, 
        so we apply compensation constant, and +3 for fisher model.
    '''
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#ir
def IR(df, rolling_date, min_period = None, device = 'cpu'):
    '''
    calc the mean/standard deviaiotn (rolling_date) of time series data
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows
        min_period: least numbers allowed in the rolling window
    @return:tuple
        ((1)mean/std result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    temp_mean = torch.mean(temp,axis = 1)
    temp_var = torch.var(temp, axis = 1)
    
    results = temp_mean/temp_var**0.5
    '''
        as we checked, torch.var() is biased variance 
        but this place we use biased variance to calculate IR(I dont know why)
        so we delete compensation constant.
    '''
    if min_period != None:
        results[torch.sum(torch.isnan(temp),axis = 1)> min_period] = torch.nan
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#sma
def SMA(df, rolling_date, a, device = 'cpu'):
    '''
    calc the decay exponent(rolling_date) of time series data
    
    same as this function:
    def SMA(close, n):
        weights = a**np.array(range(n, 0,-1))
        sum_weights = np.sum(weights)

        res = close.rolling(window=n).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
        return res
    
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days
        a: the exponent base,which is between (0,1)
    @return:tuple
        ((1)decaylinear result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
        
    if a<0 or a>1:
        raise ValueError('exponent base a must between 0 and 1')
        
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    weights = a**torch.FloatTensor(range(window, 0,-1))
    sum_weights = torch.sum(weights)
    
    results = torch.sum(temp.transpose(1,2)*weights, axis = 2)/sum_weights
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#wma
def WMA(df, rolling_date, device = 'cpu'):
    '''
    calc the anti-decaylinear(rolling_date) of time series data
    
    same as this function:
    def WMA(close, n):
        weights = np.array(range(1, n+1))
        sum_weights = np.sum(weights)

        res = close.rolling(window=n).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
        return res
    
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows

    @return:tuple
        ((1)anti-decaylinear result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    weights = torch.FloatTensor(range(window, 0,-1))
    sum_weights = torch.sum(weights)
    
    results = torch.sum(temp.transpose(1,2)*weights, axis = 2)/sum_weights
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#decaylinear
def DECAYLINEAR(df, rolling_date, device = 'cpu'):
    '''
    calc the decaylinear(rolling_date) of time series data
    
    same as this function:
    def decaylinear(close, n):
        weights = np.array(range(1, n+1))
        sum_weights = np.sum(weights)

        res = close.rolling(window=n).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
        return res
    
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:1, maximum rolling_date:df rows

    @return:tuple
        ((1)decaylinear result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    weights = torch.FloatTensor(range(1, window+1))
    sum_weights = torch.sum(weights)
    
    results = torch.sum(temp.transpose(1,2)*weights, axis = 2)/sum_weights
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#section_res
def SECTION_RES(y, x, device = 'cpu'):
    '''
    do linear regression of y on x, return the residual of cross section,
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)residual result cross section result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''

    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('x and y index not the same!')
    else:
        index = x_index
        
    temp_x = torch.FloatTensor(x_values).to(device)
    temp_y = torch.FloatTensor(y_values).to(device)
    temp_x[torch.isnan(temp_x)] = 0
    temp_y[torch.isnan(temp_y)] = 0
    temp_mean_x = torch.mean(temp_x,axis = 1)
    temp_mean_y = torch.mean(temp_y,axis = 1)    
    temp_var_x = torch.var(temp_x, axis = 1)*(temp_x.shape[1]-1)/temp_x.shape[1]
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''
    
    beta = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                    (temp_y.transpose(0,1)-temp_mean_y),axis = 0))/\
                    (temp_var_x)   
    constant = temp_mean_y - beta*temp_mean_x
      
    residual = (temp_y.transpose(0,1) - beta*temp_x.transpose(0,1) - constant).transpose(0,1)
    torch.cuda.empty_cache()
    
    results = residual
        
    return results, index, columns

#seciqr
def SECIQR(df, device = 'cpu'):
    '''
    interquartile range for each section row.(.75 quantile - .25 quantile)
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)IQR result cross section result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    temp[torch.isnan(temp)] = 0
    results = torch.quantile(temp,0.75,axis = 1)- torch.quantile(temp,0.25,axis = 1)
    results= torch.tile(results,(temp.shape[1],1)).transpose(0,1)
    torch.cuda.empty_cache()
        
    return results, index, columns

#tsiqr
def TSIQR(df, rolling_date = 22, device = 'cpu'):
    '''
    calc the interquartile range for rolling w window (0.75 quantile - 0.25 quantile)
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days
    @return:tuple
        ((1)IQR result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, values.shape[0]))
    else:
        window = rolling_date
    rw = np.lib.stride_tricks.as_strided( \
        x=values,shape=(values.shape[0]-window+1,window,values.shape[1]),\
        strides=(values.strides[0],values.strides[0],values.strides[1])) 
    
    temp = torch.FloatTensor(rw).to(calc_device)
    results = torch.quantile(temp, q = 0.75,axis = 1)-\
        torch.quantile(temp, q = 0.25,axis = 1)
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#seciqr_winsorize
def SECIQR_WINSORIZE(df, n, device = 'cpu'):
    '''
    interquartile range for each section row.(.75 quantile - .25 quantile)
    then winsorize by 0.75/0.25 +/- n_iqr * IQR
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        n: winsorize parameters 
        device: cpu or gpu:0, gpu:1,...
    @return:tuple
        ((1)IQR/winsorize result cross section result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        index = df.index
        values = df.values
    elif type(df) == tuple:
        columns = df[2]
        index = df[1]
        values = df[0].cpu().numpy()
        
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    temp = torch.FloatTensor(values).to(calc_device)
    temp[torch.isnan(temp)] = 0
    IQR = torch.quantile(temp,0.75,axis = 1)- torch.quantile(temp,0.25,axis = 1)
    IQR= torch.tile(IQR,(temp.shape[1],1)).transpose(0,1)
    down_win = (IQR.T*-n+torch.quantile(temp,0.25)).T
    up_win = (IQR.T*n+torch.quantile(temp,0.75)).T
    
    temp[temp>up_win] = up_win[temp>up_win]
    temp[temp<down_win] = down_win[temp<down_win]
    torch.cuda.empty_cache()
    
    results = temp
    return results, index, columns

#struct_mean
def STRUCT_MEAN(x, y, rolling_date = 22, lamb = 0.25, high_low = True, device = 'cpu'):
    '''
    calculate the struct_mean(rolling_date) of time series data
    rolling N of (mean(x[top lamb y]) - mean(x[bottom lamb y]))
    reverse factor 
    reference: 开源证券-《A股反转之力的微观来源》
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
        lamb: quantile percentage between(0,1)
        high_low: True ascending, False descending
    @return:tuple
        ((1)struct_mean result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index

    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    
    if high_low == True:
        up_y = temp_y>=torch.quantile(temp_y,1-lamb,axis = 1).unsqueeze(1)
        down_y = temp_y<=torch.quantile(temp_y,lamb,axis = 1).unsqueeze(1)
    else:
        up_y = temp_y>=torch.quantile(temp_y,lamb,axis = 1).unsqueeze(1)
        down_y = temp_y<=torch.quantile(temp_y,1-lamb,axis = 1).unsqueeze(1)        
    
    tmpx1 = copy(temp_x)
    tmpx1[~up_y] = 0
    temp_sum_up = torch.sum(tmpx1,axis = 1)
    tmpx1[tmpx1==0] = torch.nan
    temp_num_up = torch.sum(~torch.isnan(tmpx1),axis = 1)
    temp_mean_up = temp_sum_up/temp_num_up
    tmpx1 = copy(temp_x)
    tmpx1[~down_y] = 0
    temp_sum_down = torch.sum(tmpx1,axis = 1)
    tmpx1[tmpx1==0] = torch.nan
    temp_num_down = torch.sum(~torch.isnan(tmpx1),axis = 1)
    temp_mean_down = temp_sum_down/temp_num_down
    
    results = temp_mean_up - temp_mean_down
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#struct_sum
def STRUCT_SUM(x, y, rolling_date = 22, lamb = 0.25, high_low = True, device = 'cpu'):
    '''
    calculate the struct_sum(rolling_date) of time series data
    rolling N of (sum(x[top lamb y]) - sum(x[bottom lamb y]))
    reverse factor 
    reference: 开源证券-《A股反转之力的微观来源》
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
        lamb: quantile percentage between(0,1)
        high_low: True ascending, False descending
    @return:tuple
        ((1)struct_sum result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index

    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    
    if high_low == True:
        up_y = temp_y>=torch.quantile(temp_y,1-lamb,axis = 1).unsqueeze(1)
        down_y = temp_y<=torch.quantile(temp_y,lamb,axis = 1).unsqueeze(1)
    else:
        up_y = temp_y>=torch.quantile(temp_y,lamb,axis = 1).unsqueeze(1)
        down_y = temp_y<=torch.quantile(temp_y,1-lamb,axis = 1).unsqueeze(1)        
    
    tmpx1 = copy(temp_x)
    tmpx1[~up_y] = 0
    temp_sum_up = torch.sum(tmpx1,axis = 1)

    tmpx1 = copy(temp_x)
    tmpx1[~down_y] = 0
    temp_sum_down = torch.sum(tmpx1,axis = 1)
    
    results = temp_sum_up - temp_sum_down
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#rolling_regression
def rolling_regression(x, y, rolling_date, device = 'cpu'):
    '''
    calculate the time series OLS of x and y
    belong to TSOLS category
    Attention: this function returns all the results of OLS in tuple
                cannot combine with other operators
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
    @return:tuple
        ((1)OLS beta time series result, torch.FloatTensor, placed on device;
         (2)OLS constant time series result, torch.FloatTensor, placed on device;
         (3)OLS residual time series result, torch.FloatTensor, placed on device;
         (4)df.index;
         (5)df.columns;
        ) 
    
    '''
    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('index not the same!')
    else:
        index = x_index
    
    if rolling_date == 0 or x_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, x_values.shape[0]))
    else:
        window = rolling_date
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)
    
    rw_y = np.lib.stride_tricks.as_strided( \
        x=y_values,shape=(y_values.shape[0]-window+1,window,y_values.shape[1]),\
        strides=(y_values.strides[0],y_values.strides[0],y_values.strides[1])) 
    
    temp_y = torch.FloatTensor(rw_y).to(calc_device) 
    

    temp_mean_x = torch.mean(temp_x, axis = 1)
    temp_var_x = torch.var(temp_x, axis = 1)*(window-1)/window
    temp_mean_y = torch.mean(temp_y, axis = 1)
    #temp_var_y = torch.var(temp_y, axis = 1)*(window-1)/window
    '''
        as we checked, torch.var() is biased variance 
        so we apply compensation constant.
    '''

    beta = (torch.mean((temp_x.transpose(0,1)-temp_mean_x)*\
                    (temp_y.transpose(0,1)-temp_mean_y),axis = 0))/\
                    (temp_var_x)   
    constant = temp_mean_y - beta*temp_mean_x
      
    residual = temp_y.transpose(0,1) - beta*temp_x.transpose(0,1) - constant
    
    residual = residual[window-1,:,:]
    
    '''
        as we checked, dataframe.std() is different from torch.std(), 
        so we apply compensation constant
    '''
        
    offset = torch.full((window-1, beta.shape[1]),torch.nan).to(calc_device)
    beta = torch.vstack([offset,beta])
    constant = torch.vstack([offset,constant])
    residual = torch.vstack([offset,residual])
    torch.cuda.empty_cache()
    
    return beta,constant,residual, index, columns

#ud_std
def UD_STD(df, rolling_date = 22, up_down = True, device = 'cpu'):
    '''
    calculate the up or down standard deviaiotn (rolling_date) of time series data
    @param: 
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows
        up_down: False for only calculate up std, True for calculate up_std - down_std
    @return:tuple
        ((1)std or std difference result, time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        df_index = df.index
        df_values = df.values
    elif type(df) == tuple:
        columns = df[2]
        df_index = df[1]
        df_values = df[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')

    if rolling_date == 0 or df_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, df_values.shape[0]))
    else:
        window = rolling_date
    rw_df = np.lib.stride_tricks.as_strided( \
        x=df_values,shape=(df_values.shape[0]-window+1,window,df_values.shape[1]),\
        strides=(df_values.strides[0],df_values.strides[0],df_values.strides[1])) 
    
    temp_df = torch.FloatTensor(rw_df).to(calc_device)

    if up_down == True:
        temp = copy(temp_df)
        temp[temp_df <0] = 0    
        results_up = torch.std(temp, axis = 1)
        
        temp = copy(temp_df)
        temp[temp_df >0] = 0    
        results_down = torch.std(temp, axis = 1)
        results = results_up- results_down
        
    else:
        temp = copy(temp_df)
        temp[temp_df <0] = 0    
        results = torch.std(temp, axis = 1)        
    
    
    if window > 1:
        results = results*np.sqrt(window*(window-1))/window
    else:
        results = results*0
    '''
        as we checked, dataframe.std() is different from torch.std(), 
        so we apply compensation constant
    '''
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, df_index, columns

#ud_corr
def UD_CORR(x, df, rolling_date = 22, up_down = True, device = 'cpu'):
    '''
    calculate the up or down correlation (rolling_date) of time series data between x and y
        For the X and df sequences in the rolling N-cycle window, 
        first split them into two sequences according to the X values 
        greater than 0 (up) and less than 0 (down), 
        and tnen calculate the correlation coefficients of the two sequences.
        If high_low is True, return the difference between two sequences. 
        Otherwise, return the high sequence correlation coefficient.
    @param: 
        x: often return data which is previously known
            Generally, conforms to the positive distribution
            
            dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
            
        df:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        device: cpu or gpu:0, gpu:1,...
        rolling_date: rolling back days, minimum rolling_date:2, maximum rolling_date:df rows

        up_down: False for only calculate up correlation, 
            True for calculate up_corr-down_corr
    @return:tuple
        ((1)correlation or correlation difference result, 
            time_series result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''
    if type(df) == pd.core.frame.DataFrame:
        columns = df.columns
        df_index = df.index
        df_values = df.values
    elif type(df) == tuple:
        columns = df[2]
        df_index = df[1]
        df_values = df[0].cpu().numpy()
    if type(x) == pd.core.frame.DataFrame:
        x = x.loc[df.index]
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:

        x_index = x[1]
        x_values = x[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (df_index != x_index).any():
        raise('index not the same!')
    else:
        index = df_index

    if rolling_date == 0 or df_values.shape[0]< rolling_date:
        raise ValueError('Moving window (=%d) must between 1 and %d, inclusive'\
                         %(rolling_date, df_values.shape[0]))
    else:
        window = rolling_date
    rw_df = np.lib.stride_tricks.as_strided( \
        x=df_values,shape=(df_values.shape[0]-window+1,window,df_values.shape[1]),\
        strides=(df_values.strides[0],df_values.strides[0],df_values.strides[1])) 
    
    temp_df = torch.FloatTensor(rw_df).to(calc_device)
    rw_x = np.lib.stride_tricks.as_strided( \
        x=x_values,shape=(x_values.shape[0]-window+1,window,x_values.shape[1]),\
        strides=(x_values.strides[0],x_values.strides[0],x_values.strides[1])) 
    
    temp_x = torch.FloatTensor(rw_x).to(calc_device)

    temp_x[torch.isnan(temp_x)] = 0
    temp_df[torch.isnan(temp_df)] = 0
    if up_down == True:
        temp1 = copy(temp_df)
        temp1[temp_x <0] = 0   
        temp2 = copy(temp_x)
        temp2[temp_x <0] = 0
        temp_mean_x = torch.mean(temp1, axis = 1)
        temp_var_x = torch.var(temp1, axis = 1)*(window-1)/window
        temp_mean_y = torch.mean(temp2, axis = 1)
        temp_var_y = torch.var(temp2, axis = 1)*(window-1)/window
        '''
            as we checked, torch.var() is biased variance 
            so we apply compensation constant.
        '''
        corr_up = (torch.mean((temp1.transpose(0,1)-temp_mean_x)*\
                             (temp2.transpose(0,1)-temp_mean_y),axis = 0)/\
                       temp_var_x**0.5/ temp_var_y**0.5)
        corr_up[torch.isnan(corr_up)] = 0
        
        temp1 = copy(temp_df)
        temp1[temp_x >0] = 0   
        temp2 = copy(temp_x)
        temp2[temp_x >0] = 0
        temp_mean_x = torch.mean(temp1, axis = 1)
        temp_var_x = torch.var(temp1, axis = 1)*(window-1)/window
        temp_mean_y = torch.mean(temp2, axis = 1)
        temp_var_y = torch.var(temp2, axis = 1)*(window-1)/window
        '''
            as we checked, torch.var() is biased variance 
            so we apply compensation constant.
        '''
        corr_down = (torch.mean((temp1.transpose(0,1)-temp_mean_x)*\
                             (temp2.transpose(0,1)-temp_mean_y),axis = 0)/\
                       temp_var_x**0.5/ temp_var_y**0.5)
        corr_down[torch.isnan(corr_down)] = 0
        
        results = corr_up - corr_down
        
    else:
        temp1 = copy(temp_df)
        temp1[temp_x <0] = 0   
        temp2 = copy(temp_x)
        temp2[temp_x <0] = 0
        temp_mean_x = torch.mean(temp1, axis = 1)
        temp_var_x = torch.var(temp1, axis = 1)*(window-1)/window
        temp_mean_y = torch.mean(temp2, axis = 1)
        temp_var_y = torch.var(temp2, axis = 1)*(window-1)/window
        '''
            as we checked, torch.var() is biased variance 
            so we apply compensation constant.
        '''
        corr_up = (torch.mean((temp1.transpose(0,1)-temp_mean_x)*\
                             (temp2.transpose(0,1)-temp_mean_y),axis = 0)/\
                            temp_var_x**0.5/ temp_var_y**0.5)   
        corr_up[torch.isnan(corr_up)] = 0
        
        results = corr_up
    
    if window > 1:
        results = results*np.sqrt(window*(window-1))/window
    else:
        results = results*0
    '''
        as we checked, dataframe.std() is different from torch.std(), 
        so we apply compensation constant
    '''
    
    offset = torch.full((window-1, results.shape[1]),torch.nan).to(calc_device)
    results = torch.vstack([offset,results])
    torch.cuda.empty_cache()
    
    return results, index, columns

#struct_section_zscore
def STRUCT_SECTION_ZSCORE(x, y, n, lamb = 0.2, high_low = True, device = 'cpu'):
    '''
    reverse factor 
    reference: 开源证券-《A股反转之力的微观来源》
        First, do structural splitting,
        Calculate the mean value of lamb * n x before / after y as the benchmark 
            and record it as day_top, day_bottom
        Then calculate the cross-section zscore of both
        (day_top - mean(day_top)) / np.nanstd(day_top) - \
            (day_bottom - np.nanmean(day_bottom)) / np.nanstd(day_bottom)  
        If high_low = False, (day_top - mean(day_top)) / np.nanstd(day_top)
    
    @param: 
        x:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        y:dataframe or tuple((1),(2),(3))
            if dataframe: split to ((1),(2),(3))
            if tuple((1),(2),(3)) do operation on (1), and keep (2),(3) as results
        n: the n times standard of lamb
        lamb: rate between (0,1)
        high_low: True for day_up - day_bottom, False for day_up only
        device: cpu or gpu:0, gpu:1,...
        
    @return:tuple
        ((1)struct result, cross section result, torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
        ) 
    '''

    if type(x) == pd.core.frame.DataFrame:
        columns = x.columns
        x_index = x.index
        x_values = x.values
    elif type(x) == tuple:
        columns = x[2]
        x_index = x[1]
        x_values = x[0].cpu().numpy()
    if type(y) == pd.core.frame.DataFrame:

        y_index = y.index
        y_values = y.values
    elif type(y) == tuple:

        y_index = y[1]
        y_values = y[0].cpu().numpy()
    
    try:
        calc_device = torch.device(device)
    except:
        calc_device = torch.device('cpu')
    
    if (y_index != x_index).any():
        raise('x and y index not the same!')
    else:
        index = x_index
        
    temp_x = torch.FloatTensor(x_values).to(device)
    temp_y = torch.FloatTensor(y_values).to(device)
    temp_x[torch.isnan(temp_x)] = 0
    temp_y[torch.isnan(temp_y)] = 0
    
    if n*lamb >=1:
        raise ValueError('quantile is more than 1')
    up_y = temp_y>torch.quantile(temp_y,n*lamb,axis = 1).unsqueeze(1)
    down_y = temp_y<torch.quantile(temp_y,1-n*lamb,axis = 1).unsqueeze(1)
    
    if high_low == True:
        tmpx1 = copy(temp_x)
        tmpx1[~up_y] = 0
        temp_sum_up = torch.sum(tmpx1,axis = 1)
        temp_std_up = (torch.var(tmpx1, axis = 1)*\
                       (tmpx1.shape[1]-1)/tmpx1.shape[1])**0.5
        '''
            as we checked, torch.var() is biased variance 
            so we apply compensation constant.
        '''
        tmpx1[tmpx1==0] = torch.nan
        temp_num_up = torch.sum(~torch.isnan(tmpx1),axis = 1)
        temp_mean_up = temp_sum_up/temp_num_up
        section_up = ((temp_x.T-temp_mean_up)/temp_std_up).T
        
        tmpx1 = copy(temp_x)
        tmpx1[~down_y] = 0
        temp_sum_down = torch.sum(tmpx1,axis = 1)
        temp_std_down = (torch.var(tmpx1, axis = 1)*\
                         (tmpx1.shape[1]-1)/tmpx1.shape[1])**0.5
        '''
            as we checked, torch.var() is biased variance 
            so we apply compensation constant.
        '''
            
        tmpx1[tmpx1==0] = torch.nan
        temp_num_down = torch.sum(~torch.isnan(tmpx1),axis = 1)
        temp_mean_down = temp_sum_down/temp_num_down
        section_down = ((temp_x.T-temp_mean_down)/temp_std_down).T    

        results = section_up - section_down
        
    else:
        tmpx1 = copy(temp_x)
        tmpx1[~up_y] = 0
        temp_sum_up = torch.sum(tmpx1,axis = 1)
        temp_std_up = (torch.var(tmpx1, axis = 1)*\
                       (tmpx1.shape[1]-1)/tmpx1.shape[1])**0.5
        '''
            as we checked, torch.var() is biased variance 
            so we apply compensation constant.
        '''
        tmpx1[tmpx1==0] = torch.nan
        temp_num_up = torch.sum(~torch.isnan(tmpx1),axis = 1)
        temp_mean_up = temp_sum_up/temp_num_up        
        section_up = ((temp_x.T-temp_mean_up)/temp_std_up).T
        results = section_up
        
    torch.cuda.empty_cache()
        
    return results, index, columns

#############################################################
#backtest
#############################################################
#tensor_to_df
def tensor_to_df(tuples):
    '''
    make tuple((1),(2),(3)) to dataframe
    @param:tuple
        ((1)torch.FloatTensor, placed on device;
         (2)df.index;
         (3)df.columns;
    @return:
        dataframe
    '''
    values = tuples[0].cpu().numpy()
    df = pd.DataFrame(values,tuples[-2],tuples[-1])  
    return df

#make_ic
def MAKE_IC(factor, price, groupers=None, use_log_ret=False, plot=False):
    '''
        calling the dqt2 module make_ic function  
        
        get RankIC and RankICIR.
        factor: 2-d DataFrame of factors
        price: 2-d DataFrame of using prices, do not shift the data.
        groupers: divide into n groupers, return dict of IC and ICIR for each groups. 
                None - only return overall IC and ICIR tuple.
        use_log_ret: use log return of normal return.
        plot: plot cumsum ic
    '''
    factor_df = tensor_to_df(factor)
    make_ic(factor_df, price, groupers=groupers, use_log_ret=use_log_ret, plot=plot)
    
#get_group_pnls    
def GET_GROUP_PNLS(factor, price, industry=None,
                    divider_num=10,
                    use_log_return=True,
                    plot=True,):
    '''
        calling the dqt2 module get_group_pnls function    
    '''
    factor_df = tensor_to_df(factor)
    get_group_pnls(factor_df, price, industry=industry,
    divider_num=divider_num,
    use_log_return=use_log_return,
    plot=plot,)
    
#make_long_short    
def MAKE_LONG_SHORT(factor, open_arr, plot=True, divider_num=10):
    '''
        calling the dqt2 module make_long_short function    
    '''
    factor_df = tensor_to_df(factor)
    make_long_short(factor_df, open_arr, plot=plot, divider_num=divider_num)
    
#make_weighted_ic    
def MAKE_WEIGHTED_IC(factor_df, price_df, plot=False):
    '''
        calling the dqt2 module make_weighted_ic function
        
        the result of this function is a weighted pearson correlation, 
        not the spearman correlation we use in dqt2.make_ic.
        which weight is 𝑤𝑖 = 0.5**(𝑖−1/𝑛−1) , i = 1, … , n
        Mean[x, w] = ∑ 𝑤𝑖𝑥𝑖
        Mean[𝑦, 𝑤] = ∑ 𝑤𝑖𝑦
        Var[x, w] = ∑𝑤𝑖𝑥𝑖**2 − Mean[x, w]**2
        Var[𝑦, 𝑤] = ∑𝑤𝑖𝑦𝑖**2 − Mean[y, w]**2
        Cov[x, y, w] = ∑ 𝑤𝑖𝑥𝑖𝑦𝑖 − Mean[x, w] × Mean[y, w]
        Corr[x, y, w] = Cov[x, y, w] / (√Var[x, w] × √Var[y, w])
        @param factor_df:2-d DataFrame of factors
        @param price_df: 2-d DataFrame of open prices, do not shift the data
        @plot: plot cumsum ic, default False
        @return: a 2-element tuple (IC,ICIR)
    '''
    factor_df = tensor_to_df(factor)
    make_weighted_ic(factor_df, price_df, plot=plot)