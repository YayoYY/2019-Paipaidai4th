'''Here is some useful functions'''

from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

# 输入:'due_data:repay_date'，输出：repay_interval
def get_repay_due_gap(repay_interval):
    if repay_interval is np.nan:
        return np.nan
    else:
        dates = repay_interval.split(':')
        due_date = datetime.strptime(dates[0], '%Y-%m-%d')
        repay_date = datetime.strptime(dates[1], '%Y-%m-%d')
        repay_interval = (due_date - repay_date).days
        return repay_interval

# 输入:'auditing_data:repay_date' ，输出：repay_interval
def get_auditing_repay_gap(repay_interval):
    if repay_interval is np.nan:
        return np.nan
    else:
        dates = repay_interval.split(':')
        auditing_date = datetime.strptime(dates[0], '%Y-%m-%d')
        repay_date = datetime.strptime(dates[1], '%Y-%m-%d')
        repay_interval = (repay_date - auditing_date).days
        return repay_interval

# 输入：'xxxx-xx-xx'，输出： 'xxxxxxxx'
def get_time_string(str_time):
    return ''.join(str_time.split('-'))

# 输入：'xxxxxxxx:...:xxxxxxxx'，输出：最大的'xxxxxxxx'
def get_max_time(str_times):		
    str_times = [int(x) for x in str_times.split(':')]
    return str(max(str_times))

# 获取统计数据
def get_stat(data, aggname, colname, prefix):
    result = pd.DataFrame(index=data[aggname].unique())
    stats = ['_mean', '_max', '_min', '_25', '_50', '_75', '_mode']
    colnames = [prefix+x for x in stats]
    _mean = data.groupby(aggname)[colname].agg(lambda x: sum(x)/len(x))
    _max = data.groupby(aggname)[colname].agg(max)
    _min = data.groupby(aggname)[colname].agg(min)
    _25 = data.groupby(aggname)[colname].agg(lambda x:x.quantile(0.25))
    _50 = data.groupby(aggname)[colname].agg(lambda x:x.quantile(0.5))
    _75 = data.groupby(aggname)[colname].agg(lambda x:x.quantile(0.75))
    _mode = data.groupby(aggname)[colname].agg(lambda x:x.mode())
    for i in range(len(stats)):
        result[colnames[i]] = locals()[stats[i]]
    return result

# 获取统计数据_1
def get_stat_1(tags, tag_ratio, stat):
    tags = tags.split('|')[1:-1]
    repay_ratios = pd.Series([tag_ratio[tag] for tag in tags])
    if stat == '_mean':
        return repay_ratios.mean()
    elif stat == '_max':
        return repay_ratios.max()
    elif stat == '_min':
        return repay_ratios.min()
    else:
        return repay_ratios.quantile(0.5)

# 获取比例数据
def get_ratio(data, aggname, colname, featname):
    result = pd.DataFrame(index=data[aggname].unique())
    result[featname] = data.groupby(aggname)[colname].agg(lambda x:sum(x)/len(x))
    return result

# 输入：'xxxxxxxx:xxxxxxxx:xxxxxxxx:xxxxxxxx'，输出：最晚和最早之间的间隔
def get_max_time_gap(str_time):
    times = [int(x) for x in str_time.split(':')]
    min_time = datetime.strptime(str(min(times)), '%Y%m%d')
    max_time = datetime.strptime(str(max(times)), '%Y%m%d')
    return (max_time - min_time).days

# 获取最多或最少的行为
def get_most_beha(beha_counts, method):
    beha_counts = beha_counts.split('+')
    dict_beha_counts = {}
    for beha_count in beha_counts:
        beha, count = beha_count.split(':')
        dict_beha_counts[beha] = count
    if method == 'max':
        return max(dict_beha_counts, key=dict_beha_counts.get)
    else:
        return min(dict_beha_counts, key=dict_beha_counts.get)

# 输入：'xxxxxxxx:xxxxxxxx:xxxxxxxx:xxxxxxxx'，输出：每两个时间重最大/最小/平均时间间隔
def get_repay_interval_1(due_dates ,method):
    due_dates = due_dates.split(':')
    if len(due_dates) == 1:
        return np.nan
    else:
        due_dates = [datetime.strptime(due_date, '%Y%m%d') for due_date in sorted(due_dates)]
        intervals = [(due_dates[i+1]-due_dates[i]).days for i in range(len(due_dates)-1)]
    if method == 'min':
        return min(intervals)
    elif method == 'max':
        return max(intervals)
    else:
        return np.sum(intervals)/len(intervals)

# 输入：'xxxxxxxx:day'，输出：'xxxxxxxx'
def get_repay_date(x):
    due_date, gap = x.split(':')
    gap = int(gap)
    due_date = datetime.strptime(due_date, '%Y%m%d')
    repay_date = due_date - timedelta(days=gap)
    return datetime.strftime(repay_date, '%Y%m%d')

# 获取还款日期(1)
def get_repay_date_1(x):
    auditing_date, repay_date, due_date = x.split(':')
    if int(auditing_date) >= int(repay_date): # 早于成交日
        repay_date = str(auditing_date)
        repay_date = datetime.strptime(repay_date, '%Y%m%d') + timedelta(days=1)
        repay_date = datetime.strftime(repay_date, '%Y%m%d')
        return repay_date[:4]+'-'+repay_date[4:6]+'-'+repay_date[6:8]
    elif int(repay_date) > int(due_date): # 晚于还款日(不包括还款日)
        return np.nan
    else:
        repay_date = str(repay_date)
        return repay_date[:4]+'-'+repay_date[4:6]+'-'+repay_date[6:8]

# 用0填充缺失值
def fill0(df):
    df.loc[:, 'user_beha_3_count'].fillna(0, inplace=True)
    df.loc[:, 'user_beha_count'].fillna(0, inplace=True)
    df.loc[:, 'user_beha_1_count'].fillna(0, inplace=True)
    df.loc[:, 'user_repay_count'].fillna(0, inplace=True)
    df.loc[:, 'user_tag_len'].fillna(0, inplace=True)
    df.loc[:, 'user_repay_freq'].fillna(0, inplace=True)
    df.loc[:, 'most_100'].fillna(0, inplace=True)
    df.loc[:, 'user_beha_2_count'].fillna(0, inplace=True)
    return df

# 用随机森林填充缺失值
def fillrf(df, isnull_columns, notnull_columns):
    for column in isnull_columns:
        print('column', column)
        reg = joblib.load('rf/'+column+'.model')
        df.loc[df[column].isnull(), column] = reg.predict(df[df[column].isnull()][notnull_columns])
    return df

# 为DataFrame重命名
def rename(df, old, new):
    return df.rename(columns={old:new})
