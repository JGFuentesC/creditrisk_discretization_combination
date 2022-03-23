# -*- coding: utf-8 -*-
"""
This set of functions performs data cleansing and/or feature engineering tasks for selected credit
scoring datasets, all of the datasets except mexican credit (due to confidentiality issues)
are publicly available, the list of corresponding urls can be found on README.md file.
All raw and decompressed data should be located in a subfolder called ./data/raw all results
are saved in the folder ./data/clean in pickle format. If data is larger than 5000 records,
sampling without replacement will be performed. All continuous features are preserved using
letter A as prefix, target feature is coded as y, index for each record is coded as id.
You must Run the Jupyter Notebook `Clean Data`

Author: José Fuentes <jose.gustavo.fuentes@comunidad.unam.mx>

"""
import os
from functools import reduce

import numpy as np
import pandas as pd
from scipy.io import arff


def clean_german(file_name: str):
    """
    :param file_name: str, filename
    :return: None
    """
    df = pd.read_csv(os.path.join('data/raw', file_name), sep=',', header=None)

    l = [2, 5, 8, 11, 13, 16, 18, 25]
    df = df[l]
    df.columns = [f'A{i}' for i in l]
    df.rename(columns={'A25': 'y'}, inplace=True)
    df = df.dropna().reset_index(drop=True)
    df['y'] -= 1
    df['y'] = df['y'].astype(int)

    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index + 1)
    df.to_pickle(os.path.join('data/clean', 'german.pickle'))


def clean_australian(file_name: str):
    """
       :param file_name: str, filename
       :return: None
    """
    df = pd.read_csv(os.path.join('data/raw/', file_name), sep=' ', header=None)

    df.columns = ['A%d' % (i + 1) for i in range(len(df.columns))]

    var = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14', 'A15']

    df = df[var].rename(columns={'A15': 'y'})
    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index + 1)
    df.to_pickle(os.path.join('data/clean', 'australian.pickle'))


def clean_japan(file_name: str):
    """
       :param file_name: str, filename
       :return: None
    """
    df = pd.read_csv(os.path.join('data/raw/', file_name), header=None)
    df = df[[2, 7, 10, 14, 15]].copy()
    df[15].replace({'+': 0, '-': 1}, inplace=True)
    df.columns = ['A2', 'A7', 'A10', 'A14', 'y']
    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index + 1)
    df.to_pickle(os.path.join('data/clean', 'japan.pickle'))


def clean_polish(random_state: str):
    """
    :param random_state: np.random.RandomState for random sampling
    Takes all files in arff format inside data/raw
    :return: None
    """
    l = [x for x in os.listdir('data/raw') if 'arff' in x]
    l.sort()
    df = pd.concat(map(lambda a: pd.DataFrame(arff.loadarff(os.path.join('data/raw', a))[0]), l), ignore_index=True)
    df = df.sample(n=5000, random_state=random_state).reset_index(drop=True)
    df['y'] = df['class'].astype(int)
    df.drop('class', axis=1, inplace=True)
    var = [v.replace('Attr', 'A') for v in df.columns if v[0] == 'A']
    df.columns = var + ['y']
    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index + 1)
    df.to_pickle(os.path.join('data/clean', 'polish.pickle'))
    df.head()


def clean_mexico(file_name, random_state: str):
    """
       :param file_name: str, filename
       :param random_state: np.random.RandomState for random sampling
       :return: None
    """
    df = pd.read_sas(os.path.join('data/raw/', file_name))
    var = [v for v in df.columns if v[:2] == 'V_']
    df.rename(columns=dict(zip(var, ['A%d' % (i + 1) for i in range(len(var))])), inplace=True)
    df['y'] = df['TARGET'].astype(int)
    df.drop(['ID', 'TARGET'], axis=1, inplace=True)
    df = df.sample(n=5000, random_state=random_state)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index)
    df.to_pickle(os.path.join('data/clean', 'mexico.pickle'))


def clean_mortgage(file_name: str, random_state: np.random.RandomState):
    """
           :param file_name: str, filename
           :param random_state: np.random.RandomState for random sampling
           :return: None
    """
    df = pd.read_csv(os.path.join('data/raw', file_name))
    df.drop(['id', 'time', 'payoff_time', 'orig_time', 'first_time', 'mat_time', 'status_time'],
            axis=1, inplace=True)

    df.rename(columns={'default_time': 'y'}, inplace=True)
    var = [v for v in df.columns if v != 'y']
    df.rename(columns=dict(zip(var, ['A%d' % (i + 1) for i in range(len(var))])), inplace=True)
    df = df.sample(n=5000, random_state=random_state)
    df.reset_index(drop=True, inplace=True)
    df.insert(0, 'id', df.index)
    df.to_pickle(os.path.join('data/clean', 'mortgage.pickle'))


def clean_lending_club(file_name: str, random_state: np.random.RandomState):
    """
       :param file_name: str, filename
       :param random_state: np.random.RandomState for random sampling
       :return: None
    """
    id_ = 'id'
    tgt = 'loan_status'
    varf = ['issue_d', 'earliest_cr_line']
    varc = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment',
            'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
            'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'annual_inc_joint',
            'dti_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
            'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
            'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim',
            'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal',
            'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
            'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
            'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
            'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
            'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
            'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd',
            'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
            'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort',
            'total_bc_limit', 'total_il_high_credit_limit', 'revol_bal_joint'
            ]
    df = pd.read_csv(os.path.join('data/raw', file_name),
                     dtype=str, usecols=varc + [id_, tgt] + varf)
    df = df.sample(n=5000, random_state=random_state).reset_index(drop=True)
    df['y'] = (~df['loan_status'].isin(['Fully Paid', 'Current'])).astype(int)

    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['tob'] = (df['issue_d'] - df['earliest_cr_line']) / np.timedelta64(1, 'M')
    varc.append('tob')
    d = dict(zip(varc, ['A%d' % (i + 1) for i in range(len(varc))]))
    df.rename(columns=d, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index + 1
    df = df[[id_] + list(d.values()) + ['y']]
    df = df.astype(float)
    df['id'] = df['id'].astype(int)
    df.to_pickle(os.path.join('data/clean', 'lendinc_club.pickle'))


def clean_give_me_some_credit(file_name: str, random_state: np.random.RandomState):
    """
       :param file_name: str, filename
       :param random_state: np.random.RandomState for random sampling
       :return: None
    """
    df = pd.read_csv(os.path.join('data/raw', file_name))
    df = df.sample(n=5000, random_state=random_state)
    df.reset_index(drop=True, inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.insert(0, 'id', df.index + 1)
    df.rename(columns={'SeriousDlqin2yrs': 'y'}, inplace=True)
    varc = [v for v in df.columns if v not in ['id', 'y']]
    df[varc] = df[varc].astype(float)
    d = dict(zip(varc, ['A%d' % (i + 1) for i in range(len(varc))]))
    df.rename(columns=d, inplace=True)
    df = df[['id'] + list(d.values()) + ['y']]
    df.to_pickle(os.path.join('data/clean', 'give_me_some_credit.pickle'))


def clean_taiwan(file_name: str, random_state: np.random.RandomState):
    """
       :param file_name: str, filename
       :param random_state: np.random.RandomState for random sampling
       :return: None
    """

    id_ = 'id'
    df = pd.read_csv(os.path.join('data/raw', file_name))
    df = df.sample(n=5000, random_state=random_state).reset_index(drop=True)

    varc = ['BILL_AMT', 'PAY_AMT']
    varc = [['%s%d' % (v, i) for i in range(2, 7)] for v in varc]

    for i, l in enumerate(varc):
        df['A%d' % (i + 1)] = df[l].mean(axis=1)

    i = 2
    df['A3'] = 0
    for x, y in zip(*varc):
        df['A3'] += df[y] / df[x]
    df['A3'] /= 5
    df['A3'] = df['A3'].replace({np.inf: np.nan, -np.inf: np.nan})

    df['y'] = (df['PAY_0'] >= 2).astype(int)

    varc = ['PAY_%d' % i for i in range(2, 7)]

    for v in varc:
        df[v] = (df[v] >= 1).astype(int)

    df['A4'] = 0
    for x, y in zip(varc, varc[1:]):
        df['A4'] += (df[y] > df[x]).astype(int)

    df = df.rename(columns={'LIMIT_BAL': 'A5', 'AGE': 'A6'})

    varc = sorted([v for v in df.columns if v[0] == 'A'])
    df.insert(0, id_, df.index + 1)
    df = df[[id_] + varc + ['y']]
    df.to_pickle(os.path.join('data/clean', 'taiwan.pickle'))


def clean_hmeq(file_name: str):
    """
       :param file_name: str, filename
       :return: None
    """
    id_ = 'id'
    df = pd.read_csv(os.path.join('data/raw', file_name))
    var = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ',
           'CLNO', 'DEBTINC']
    vara = ['A%d' % (i + 1) for i in range(len(var))]
    df = df.rename(columns=dict(zip(var, vara)))
    df = df.rename(columns={'BAD': 'y'})
    df.insert(0, id_, df.index + 1)
    df = df[[id_] + vara + ['y']]
    df.to_pickle(os.path.join('data/clean', 'hmeq.pickle'))


def clean_farmers(file_name: str, random_state: np.random.RandomState):
    """
        :param file_name: str, filename
        :param random_state: np.random.RandomState for random sampling
        :return: None
     """
    df = pd.read_csv(os.path.join('data/raw', file_name), dtype=str)

    for v in df.columns:
        df[v] = pd.to_numeric(df[v].map(lambda x: x.replace(',', '')), errors='coerce')

    vobs = 6
    vdes = 1

    anchori = vobs
    anchorf = 51

    id_ = 'Farmer No.'

    def ing(df, k, anchor):
        var = ['Week%d' % i for i in range(anchor - k + 1, anchor + 1)]
        aux = df[[id_] + var].copy()
        aux['v_mean_%d' % k] = aux[var].mean(axis=1)
        aux['v_std_%d' % k] = aux[var].std(axis=1)
        aux['v_max_%d' % k] = aux[var].max(axis=1)
        aux['v_min_%d' % k] = aux[var].min(axis=1)
        return aux[[id_] + [v for v in aux if v[:2] == 'v_']].assign(anchor=anchor)

    X = reduce(lambda x, y: pd.merge(x, y, on=[id_, 'anchor'], how='outer'),
               map(lambda k: pd.concat(map(lambda anchor: ing(df, k, anchor),
                                           range(anchori, anchorf + 1)),
                                       ignore_index=True), range(2, 8, 2)))

    def tgt(df, anchor):
        aux = df[[id_, 'Week%d' % (anchor + vdes)]].copy()
        aux.rename(columns={'Week%d' % (anchor + vdes): 'y'}, inplace=True)
        return aux.assign(anchor=anchor)

    y = pd.concat(map(lambda anchor: tgt(df, anchor),
                      range(anchori, anchorf + 1)),
                  ignore_index=True)

    X = X.merge(y, on=[id_, 'anchor'], how='left')

    X['y'] = (X['y'] < X['v_mean_6']).astype(int)

    var = [v for v in X if v[:2] == 'v_']

    X.rename(columns=dict(zip(var, ['A%d' % (i + 1) for i in range(len(var))])), inplace=True)

    X.insert(0, 'id', X.index + 1)

    X = X[['id'] + ['A%d' % (i + 1) for i in range(len(var))] + ['y']]
    X = X.sample(n=5000, random_state=random_state).reset_index(drop=True)
    X.to_pickle(os.path.join('data/clean', 'farmers.pickle'))
    print(X.shape)
