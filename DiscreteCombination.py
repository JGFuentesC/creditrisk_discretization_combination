# -*- coding: utf-8 -*-
"""
This script contains the necessary functions and classes to perform
the combination of unsupervised discretization methods
in order to improve logistic regression predictive power.

Author: José Fuentes <jose.gustavo.fuentes@comunidad.unam.mx>

"""
import numpy as np
import pandas as pd
import asyncio
from functools import reduce
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd
import warnings
import os


def ks(x: pd.DataFrame, y: pd.DataFrame, model: LogisticRegression) -> pd.DataFrame:
    """
    Calculates KS statistic
    :param x: pandas.DataFrame with predictors Matrix
    :param y: pandas.DataFrame with target vector
    :param model: sklearn.LogisticRegression, trained model
    :return: float, kolmogorov-smirnov statistic
    """
    aux = x.copy()
    aux['p'] = y
    aux['p'] = model.predict_proba(x)[:, 1]
    aux['p'] = pd.cut(aux['p'], bins=np.arange(0, 2, 0.1), include_lowest=True).astype(str)
    aux['y'] = y
    aux.insert(0, 'id', aux.index + 1)
    aux = aux.pivot_table(index='p', columns='y', aggfunc='count', values='id', fill_value=0)
    for i in range(2):
        aux[i] /= aux[i].sum()
        aux[i] = aux[i].cumsum()
    aux['d'] = np.abs(aux[0] - aux[1])
    return aux['d'].max()


class DiscreteCombination:
    data = None
    name = None
    features = None
    discretized_data = None
    max_bins = None
    discretized_features = None
    __random_state = None
    sampling_unit = ['id']
    target = ['y']
    _all_data = None
    _xt = None
    _xv = None
    _yt = None
    _yv = None
    iv_values = None

    def __init__(self, path: str, max_bins: int, random_state: np.random.RandomState):
        """
        :param path: str, full path to the clean pickle data
        :param max_bins:int, maximum number of bins for discretization
        :param random_seed:int, random seed for results reproduction
        :return: None
        """
        self.data = pd.read_pickle(path)
        self.name = path.split('/')[-1].split('.')[0]
        self.features = self.data.filter(like='A').columns.tolist()
        self.max_bins = max_bins
        self.discretized_data = asyncio.run(self.__get_discretization())
        self.discretized_features = [c for c in self.discretized_data.columns if c[:2] == 'i_']
        self._all_data = self.data.merge(self.discretized_data, on=self.sampling_unit, how='inner')
        self.__random_state = random_state
        self.data_partition()
        self.iv_values = asyncio.run(self.__get_iv_calculation())

    @staticmethod
    async def __discretize(data: pd.DataFrame,
                           sampling_unit: list,
                           feature: str,
                           strategy: str,
                           n_bins: int) -> pd.DataFrame:
        """
        Performs discretization of continuous features.
        :param data: pandas.DataFrame with data to be discretized.
        :param sampling_unit: list, sampling unit.
        :param feature: str, feature to be discretized.
        :param strategy: str, strategies for discretization, 'uniform',
        'quantile', 'kmeans' and 'gaussian' available
        :param n_bins: int number of bins used for discretization.
        this threshold to be merged in order to provide sufficient frequency towards WoE encoding
        :returns pd.DataFrame with discretized data
        """

        name_interval = f'i_{feature}_{n_bins}_{strategy[:3]}'
        name_discrete = f'd_{feature}_{n_bins}_{strategy[:3]}'
        aux = data[sampling_unit + [feature]].fillna('00. MISSING').copy()
        missing_split = [x.reset_index(drop=True) for _, x in aux.groupby(aux[feature] != '00. MISSING')]
        if len(missing_split) == 2:
            miss, nomiss = missing_split
        else:
            nomiss, miss = missing_split[0], aux.loc[aux[feature] == '00. MISSING'].reset_index(drop=True)
        miss.rename(columns={feature: name_interval}, inplace=True)

        if strategy in ['uniform', 'quantile', 'kmeans']:
            kb = KBinsDiscretizer(strategy=strategy, n_bins=n_bins, encode='ordinal')
            nomiss[name_discrete] = kb.fit_transform(nomiss[[feature]])
            boundaries = list(sorted(list(kb.bin_edges_[0])))

            nomiss[name_interval] = pd.cut(nomiss[feature], bins=boundaries, labels=range(len(boundaries) - 1),
                                           include_lowest=True).astype(
                str)
            boundaries = list(map(lambda x: str(round(x, 3)), boundaries))

        elif strategy == 'gaussian':
            kb = GaussianMixture(n_components=n_bins, max_iter=len(data))
            nomiss[name_discrete] = kb.fit_predict(nomiss[[feature]])
            aux = nomiss[[name_discrete, feature]].groupby(name_discrete).agg(('min', 'max'))[feature]
            aux = aux.sort_values(by='min').reset_index()
            boundaries = [aux['min'].tolist()[0]] + aux['max'].tolist()
            boundaries = list(sorted(set(boundaries)))
            nomiss[name_interval] = pd.cut(nomiss[feature], bins=boundaries, labels=range(len(boundaries) - 1),
                                           include_lowest=True).astype(
                str)
            boundaries = list(map(lambda x: str(round(x, 3)), boundaries))
        labels = ['|'.join(edges) for edges in zip(boundaries, boundaries[1:])]
        labels = ['%02d. [%s]' % (i + 1, e) if i == 0 else '%02d. (%s]' % (i + 1, e) for i, e in
                  enumerate(labels)]
        bins = nomiss[name_interval].value_counts().sort_index().index
        nomiss[name_interval] = nomiss[name_interval].map(dict(zip(bins, labels)))

        if len(labels) != len(bins):
            nomiss[name_interval] = '99. FAIL TO DISCRETIZE'
            miss[name_interval] = '99. FAIL TO DISCRETIZE'
        return pd.concat([nomiss[sampling_unit + [name_interval]], miss], ignore_index=True)

    async def __get_discretization(self) -> pd.DataFrame:
        """
        Consolidates all discretization tasks
        :return: pd.DataFrame with all discrete features
        """
        return reduce(lambda x, y: pd.merge(x, y, on=self.sampling_unit, how='outer'),
                      await asyncio.gather(
                          *map(lambda z: self.__discretize(self.data, self.sampling_unit, *z),
                               [(feat, strategy, n_bins) for feat
                                in self.features for strategy in
                                ['uniform', 'quantile', 'kmeans',
                                 'gaussian'] for n_bins in
                                range(2, self.max_bins + 1)])))

    def data_partition(self):
        """
        Partitions data in a train-validation fashion (70%-30%)
        :return: None
        """
        x, y = self._all_data[self.sampling_unit + self.discretized_features], self._all_data[
            self.sampling_unit + self.target]
        self._xt, self._xv, self._yt, self._yv = train_test_split(x, y, train_size=0.7,
                                                                  random_state=self.__random_state)
        self._xt.reset_index(drop=True, inplace=True)
        self._xv.reset_index(drop=True, inplace=True)
        self._yt.reset_index(drop=True, inplace=True)
        self._yv.reset_index(drop=True, inplace=True)

    @staticmethod
    async def __calculate_iv(data: pd.DataFrame, feature: str, target: str) -> tuple:
        """
        Performs iv calculation for a given feature
        :param data: pandas.DataFrame containing discretized features
        :param feature: str, feature to be measured
        :param target: str, target feature used for IV calculation
        :return: tuple containing feature name and IV value
        """
        aux = data[[feature, target]].copy().assign(n=1)
        aux = aux.pivot_table(index=feature, columns=target, values='n', aggfunc='sum', fill_value=0)
        for i in range(2):
            aux[i] /= aux[i].sum()
        aux['w'] = np.log(aux[0] / aux[1])
        aux['iv'] = (aux[0] - aux[1]) * aux['w']
        if np.isinf(aux['iv'].sum()):
            iv = np.nan
        else:
            iv = aux['iv'].sum()
        return feature, iv

    async def __get_iv_calculation(self) -> pd.DataFrame:
        """
        Calculates every IV value for all features
        :return: None
        """
        param = [(d, f, t) for d in [self._all_data] for f in self.discretized_features for t in self.target]
        iv_values = await asyncio.gather(*map(lambda z: self.__calculate_iv(*z), param))
        iv_values = pd.DataFrame(iv_values, columns=['feature', 'iv'])
        iv_values.sort_values(by='iv', ascending=False, inplace=True)
        iv_values.reset_index(drop=True, inplace=True)
        iv_values['root_feature'] = iv_values['feature'].map(
            lambda x: "_".join(x.split('_')[1:-2]) if x[:2] == 'i_' else "_".join(x.split('_')[1:]))
        iv_values.sort_values(by=['root_feature', 'iv'], ascending=[True, False], inplace=True)
        iv_values.reset_index(drop=True, inplace=True)
        iv_values = iv_values.sort_values(by=['root_feature', 'feature']).reset_index(drop=True)
        iv_values['method'] = iv_values['feature'].map(lambda x: x.split('_')[-1])
        iv_values['bins'] = iv_values['feature'].map(lambda x: x.split('_')[-2])
        return iv_values

    @staticmethod
    async def __woe_encoding(data: pd.DataFrame, feature: str, target: str) -> tuple:
        """
          Encodes discrete feature into Weight of Evidence
          :param data: pandas.DataFrame containing discretized features
          :param feature: str, feature to be encoded
          :param target: str, target feature used for WoE encoding
          :return: tuple containing feature name WoE encoding dictionary
        """
        aux = data[[feature, target]].copy().reset_index(drop=True).assign(n=1)
        aux = aux.pivot_table(index=feature, columns=target, values='n', aggfunc='count', fill_value=0)
        for i in range(2):
            aux[i] /= aux[i].sum()
        aux['w'] = np.log(aux[0] / aux[1])
        if np.isinf(np.abs(aux['w']).max()):
            aux['w'] = 0
        return feature, aux['w'].to_dict()

    async def __get_woe_encoding(self, features: list) -> list:
        """
        Gets WoE encoding map
        :param features: features to map
        :return: list with tuples containing feature name and encoding dictionary
        """
        param = [(d, f, t) for d in [self._xt.merge(self._yt, on=self.sampling_unit, how='inner')] for f in
                 features for t in self.target]
        woe_encoding_map = await asyncio.gather(*map(lambda z: self.__woe_encoding(*z), param))
        return woe_encoding_map

    @staticmethod
    def performance_metrics(x: pd.DataFrame, y: pd.DataFrame, model: LogisticRegression, label: str) -> pd.DataFrame:
        """
        Calculates a variety of classification metrics
        :param x: pandas.DataFrame with predictors Matrix
        :param y: pandas.DataFrame with target vector
        :param model: sklearn.LogisticRegression, trained model
        :param label: str, label for dataframe
        :return: pandas.DataFrame
        """
        t = roc_auc_score(y_true=y,
                          y_score=model.predict_proba(x)[:, 1]), \
            accuracy_score(y_true=y, y_pred=model.predict(x)), \
            precision_score(y_true=y, y_pred=model.predict(x)), \
            recall_score(y_true=y, y_pred=model.predict(x)), \
            f1_score(y_true=y, y_pred=model.predict(x)), \
            ks(x, y, model)
        return pd.DataFrame([t], columns=['roc', 'acc', 'pre', 'rec', 'f1', 'ks']).assign(label=label)

    def discrete_competitive_combination(self) -> pd.DataFrame:
        """
        Performs Discrete Competitive Combination (DCC)
        :return: pd.DataFrame with computational experiment results
        """
        # Make a copy of partitioned data
        xt, xv, yt, yv = self._xt.copy(), self._xv.copy(), self._yt.copy(), self._yv.copy()
        init = datetime.now()
        # Select best method and best number of bins (DCC)
        iv_values = self.iv_values.copy().sort_values(by=['root_feature', 'iv'], ascending=[1, 0]).reset_index(
            drop=True)
        iv_values = iv_values.groupby('root_feature').first()
        features = list(iv_values['feature'])
        # Get WoE encoding
        woe_encoding_map = asyncio.run(self.__get_woe_encoding(features))
        # WoE Encoding
        for f, woe_map in woe_encoding_map:
            xt[f].replace(woe_map, inplace=True)
            xv[f].replace(woe_map, inplace=True)
            xt[f] = np.where(xt[f].map(type) == str, 0, xt[f])
            xv[f] = np.where(xv[f].map(type) == str, 0, xv[f])

        # Logistic Regression (with l2 regularization) Parameter Estimation
        lr = LogisticRegression(penalty='l2', C=1.0)
        lr.fit(xt[features], yt[self.target])
        results = pd.concat([self.performance_metrics(xt[features], yt[self.target], lr, f'dcc_{self.name}_train'),
                             self.performance_metrics(xv[features], yv[self.target], lr, f'dcc_{self.name}_validate')],
                            ignore_index=True)
        results.insert(0, 'completion_time', (datetime.now() - init).total_seconds() * 1000)
        return results

    def discrete_exhaustive_combination(self) -> pd.DataFrame:
        """
        Performs Discrete Exhaustive Combination (DEC)
        :return: pd.DataFrame with computational experiment results
        """
        # Make a copy of partitioned data
        xt, xv, yt, yv = self._xt.copy(), self._xv.copy(), self._yt.copy(), self._yv.copy()
        init = datetime.now()
        # Select best binning for each method, every method is present
        iv_values = self.iv_values.copy().sort_values(by=['method', 'root_feature', 'iv'],
                                                      ascending=[1, 1, 0]).reset_index(
            drop=True)
        iv_values = iv_values.groupby(['method', 'root_feature']).first()
        features = list(iv_values['feature'])
        # Get WoE encoding
        woe_encoding_map = asyncio.run(self.__get_woe_encoding(features))
        # WoE Encoding
        for f, woe_map in woe_encoding_map:
            xt[f].replace(woe_map, inplace=True)
            xv[f].replace(woe_map, inplace=True)
            xt[f] = np.where(xt[f].map(type) == str, 0, xt[f])
            xv[f] = np.where(xv[f].map(type) == str, 0, xv[f])

        # Logistic Regression (with l2 regularization) Parameter Estimation
        lr = LogisticRegression(penalty='l2', C=1.0)
        lr.fit(xt[features], yt[self.target])
        results = pd.concat([self.performance_metrics(xt[features], yt[self.target], lr, f'dec_{self.name}_train'),
                             self.performance_metrics(xv[features], yv[self.target], lr, f'dec_{self.name}_validate')],
                            ignore_index=True)
        results.insert(0, 'completion_time', (datetime.now() - init).total_seconds() * 1000)
        return results

    def one_method_only(self, method: str) -> pd.DataFrame:
        """
        Select best binning only for the selected method
        :param method:str,
        :return: pd.DataFrame with computational experiment results
        """
        # Make a copy of partitioned data
        xt, xv, yt, yv = self._xt.copy(), self._xv.copy(), self._yt.copy(), self._yv.copy()
        init = datetime.now()
        # Select best binning for each method, every method is present
        iv_values = self.iv_values.copy()
        iv_values = iv_values.loc[iv_values['method'] == method].reset_index(drop=True)
        iv_values.sort_values(by=['root_feature', 'iv'],
                              ascending=[1, 0]).reset_index(
            drop=True)
        iv_values = iv_values.groupby(['method', 'root_feature']).first()
        features = list(iv_values['feature'])
        # Get WoE encoding
        woe_encoding_map = asyncio.run(self.__get_woe_encoding(features))
        # WoE Encoding
        for f, woe_map in woe_encoding_map:
            xt[f].replace(woe_map, inplace=True)
            xv[f].replace(woe_map, inplace=True)
            xt[f] = np.where(xt[f].map(type) == str, 0, xt[f])
            xv[f] = np.where(xv[f].map(type) == str, 0, xv[f])

        # Logistic Regression (with l2 regularization) Parameter Estimation
        lr = LogisticRegression(penalty='l2', C=1.0)
        lr.fit(xt[features], yt[self.target])
        results = pd.concat([self.performance_metrics(xt[features], yt[self.target], lr, f'{method}_{self.name}_train'),
                             self.performance_metrics(xv[features], yv[self.target], lr,
                                                      f'{method}_{self.name}_validate')],
                            ignore_index=True)
        results.insert(0, 'completion_time', (datetime.now() - init).total_seconds() * 1000)
        return results


def computational_experimentation(file: str, n_bins: int, random_state: np.random.RandomState, iteration: int):
    """
    Performs computational experimentation on a given dataset
    :param file: str, file for performing computational experimentation
    :param n_bins: int, maximum number of discretization bins
    :param random_state: np.random.RandomState for reproducible results
    :param iteration: int, K-fold iteration index
    :return: pd.DataFrame with computational results
    """
    print(file, iteration)
    comb = DiscreteCombination(file, n_bins, random_state)
    results = pd.concat(map(comb.one_method_only, ['uni', 'qua', 'kme', 'gau']), ignore_index=True)
    results = pd.concat([results, comb.discrete_competitive_combination()], ignore_index=True)
    results = pd.concat([results, comb.discrete_exhaustive_combination()], ignore_index=True)
    return results.assign(i=iteration)


if __name__ == '__main__':
    init = datetime.now()
    warnings.filterwarnings('ignore')
    data_path = './data/clean'
    files_list = sorted(os.path.join(data_path, file) for file in os.listdir(data_path))
    MAX_KFOLDS = 30
    RANDOM_SEED = 2718281828
    MAX_BINS = 10
    random_state = np.random.RandomState(seed=RANDOM_SEED)

    results = reduce(lambda x, y: pd.concat([x, y], ignore_index=True),
                     map(lambda z: computational_experimentation(*z),
                         [(f, b, rs, i) for f in files_list for b in [MAX_BINS] for rs in [random_state] for i in
                          range(MAX_KFOLDS)]))
    results.to_excel('results.xlsx', index=False)
    print(rd(datetime.now(), init))
