import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

pd.options.mode.chained_assignment = None

class ModelInterface:
    """
    Root level of Model Interface
    """
    def __init__(self):
        self.raw_file_name ="assessments.csv"
        self.clean_file_name ="df_cleaned.csv"

        self.parentPath = os.path.abspath("../") + "/"
        self.data_path = self.parentPath + "data/"
        self.model_path = self.parentPath + "model/"
        self.from_year = 2000
        self.predict_years = 5
        self.df = None
        self.df_train = None
        self.df_test = None
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_predicted = None
        self.model_name = None

    def load_model(self, file_name="model"):
        file_ptr = open(self.model_path + file_name, 'rb')
        if os.path.exists(self.model_path) is True:
            self.model = pickle.load(file_ptr)
        file_ptr.close()

    def save_model(self, file_name="model"):
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)
        file_ptr = open(self.model_path + file_name, 'wb')
        pickle.dump(self.model, file_ptr)
        file_ptr.close()

    def preprocessing(self):
        self.df = pd.read_csv(self.data_path + self.raw_file_name)

        ## convert the previous sales into rows
        df_curr_sale = self.df[self.df.columns[~self.df.columns.isin(['PREVSALEDATE', 'PREVSALEPRICE', 'PREVSALEPRICE2', 'PREVSALEDATE2'])]]
        df_prev_sale = self.df[self.df.columns[~self.df.columns.isin(['SALEDATE', 'SALEPRICE', 'PREVSALEPRICE2', 'PREVSALEDATE2'])]]
        df_prev_sale_2 = self.df[self.df.columns[~self.df.columns.isin(['SALEDATE', 'SALEPRICE', 'PREVSALEPRICE', 'PREVSALEDATE'])]]

        df_prev_sale.rename(columns={"PREVSALEDATE": "SALEDATE", "PREVSALEPRICE": "SALEPRICE"}, errors="raise",
                            inplace=True)
        df_prev_sale_2.rename(columns={"PREVSALEDATE2": "SALEDATE", "PREVSALEPRICE2": "SALEPRICE"}, errors="raise",
                              inplace=True)

        ## combine current sale price and previous sale prices
        self.df = pd.concat([df_curr_sale, df_prev_sale, df_prev_sale_2]).reset_index(drop=True)

        ##
        self._filter()

        ##
        self._transformation()

        # print(df)

        ## save data as cleaned dataframe
        self.df.to_csv(self.data_path + self.clean_file_name, index=False)

        # ax = sn.lineplot(x['years_to_sale'], y / max(y), ci=80)
        # ax = sn.lineplot(x2['years_to_sale'], y2 / max(y2), ci=80)
        # ax = sn.lineplot(np.log(x['years_to_sale']), y / max(y), ci=80)
        # ax = sn.lineplot(np.log(x2['years_to_sale']), y2 / max(y2), ci=80)

    def _filter(self):

        # filters out noises:
        # only taxable properties
        # sale price is larger than $100
        # not empty in style, stories, yearbuild, bedrooms,
        filter_mask = (self.df['TAXCODE'] == 'T') & \
                      (~self.df['TAXCODE'].isna()) & \
                      (self.df['SALEPRICE'] > 20000) & \
                      (~self.df['SALEPRICE'].isna()) & \
                      (self.df['BEDROOMS'] > 0) & \
                      (~self.df['BEDROOMS'].isna()) & \
                      (self.df['FINISHEDLIVINGAREA'] > 0) & \
                      (~self.df['FINISHEDLIVINGAREA'].isna()) & \
                      (~self.df['PROPERTYZIP'].isna()) & \
                      (~self.df['SCHOOLCODE'].isna()) & \
                      (~self.df['STYLEDESC'].isna()) & \
                      (~self.df['STORIES'].isna()) & \
                      (~self.df['YEARBLT'].isna()) & \
                      (~self.df['FULLBATHS'].isna()) & \
                      (~self.df['BEDROOMS'].isna())

        self.df = self.df[filter_mask].reset_index(drop=True)

        ## mannully select input raw data
        self.df = self.df[[
            'PROPERTYZIP',
            'SCHOOLCODE',
            # 'NEIGHCODE',
            'SALEPRICE',
            'SALEDATE',
            # 'PREVSALEPRICE',
            # 'PREVSALEDATE',
            # 'PREVSALEPRICE2',
            # 'PREVSALEDATE2',
            # 'COUNTYBUILDING',
            # 'COUNTYLAND',
            # 'LOCALBUILDING',
            # 'LOCALLAND',
            # 'FAIRMARKETBUILDING',
            # 'FAIRMARKETLAND',
            'STYLEDESC',
            'STORIES',
            'YEARBLT',
            # 'EXTERIORFINISH',
            'ROOF',
            'BASEMENT',
            'GRADE',
            'CONDITION',
            'CDU',
            'BEDROOMS',
            'FULLBATHS',
            'HALFBATHS',
            # 'HEATINGCOOLING',
            'FIREPLACES',
            'BSMTGARAGE',
            'FINISHEDLIVINGAREA',
            'LOTAREA'
        ]]

    def _transformation(self):

        # how many years since built
        self.df['years_built'] = int(datetime.today().year) - self.df['YEARBLT'].apply(int)

        # get sale year and month
        self.df['SALEDATE'] = pd.to_datetime(self.df['SALEDATE'], format="%m-%d-%Y", errors='coerce')
        self.df = self.df[~self.df['SALEDATE'].isna()].reset_index(drop=True)
        tmp = pd.DatetimeIndex(self.df['SALEDATE'])
        self.df['sale_year'] = tmp.year
        self.df['sale_month'] = tmp.month
        self.df['sale_day'] = tmp.day
        self.df['years_to_sale'] = int(datetime.today().year) - tmp.year
        self.df['sale_quarter'] = tmp.quarter

        ### use recent 10 years of data
        self.df = self.df[self.df['sale_year'] >= self.from_year].reset_index(drop=True)  ##1970, 2000

        ## fill na
        self.df['CONDITION'] = self.df['CONDITION'].fillna(9)
        self.df['HALFBATHS'] = self.df['HALFBATHS'].fillna(0)
        self.df['BASEMENT'] = self.df['BASEMENT'].fillna(1)
        self.df['ROOF'] = self.df['ROOF'].fillna(7)
        self.df['FIREPLACES'] = self.df['FIREPLACES'].fillna(0)
        self.df['BSMTGARAGE'] = self.df['BSMTGARAGE'].fillna(0)
        self.df['LOTAREA'] = self.df['LOTAREA'].fillna(0)
        self.df['STORIES'] = self.df['STORIES'].fillna(1)
        self.df['GRADE'] = self.df['GRADE'].fillna('Unclear')
        self.df['CDU'] = self.df['CDU'].fillna('Unclear')

        ## data type conversion

        # convert some numerical data to categorical ones
        self.df['PROPERTYZIP'] = self.df['PROPERTYZIP'].apply(int).apply(str)
        self.df['SCHOOLCODE'] = self.df['SCHOOLCODE'].apply(int).apply(str)
        # self.df['NEIGHCODE'] = self.df['NEIGHCODE'].apply(str)
        self.df['STYLEDESC'] = self.df['STYLEDESC'].apply(str)
        # self.df['EXTERIORFINISH'] = self.df['EXTERIORFINISH'].apply(int).apply(str)
        # self.df['sale_quarter'] = self.df['sale_quarter'].apply(str)

        # convert some string data to  numerical ones
        self.df['STORIES'] = self.df['STORIES'].apply(int)
        self.df['CONDITION'] = self.df['CONDITION'].apply(int)
        self.df['BEDROOMS'] = self.df['BEDROOMS'].apply(int)
        self.df['FULLBATHS'] = self.df['FULLBATHS'].apply(int)
        self.df['HALFBATHS'] = self.df['HALFBATHS'].apply(int)
        self.df['BASEMENT'] = self.df['BASEMENT'].apply(int)
        self.df['ROOF'] = self.df['ROOF'].apply(int)
        self.df['FIREPLACES'] = self.df['FIREPLACES'].apply(int)
        self.df['BSMTGARAGE'] = self.df['BSMTGARAGE'].apply(int)
        self.df['LOTAREA'] = self.df['LOTAREA'].apply(int)

        # get sale price per square feet
        self.df['sale_price_per_sf'] = self.df['SALEPRICE'] / self.df['FINISHEDLIVINGAREA']

        ## adjust the order of some numbers based on plots
        print(self.df[['sale_quarter', 'sale_price_per_sf']].groupby("sale_quarter").mean())
        df_mean = self.df[['sale_quarter', 'sale_price_per_sf']].groupby("sale_quarter").mean()
        new_index = df_mean.index.to_list()
        df_mean.sort_values(by=['sale_price_per_sf'], inplace=True)
        old_index = df_mean.index.to_list()
        self.df['sale_quarter'] = self.df['sale_quarter'].replace(old_index, new_index)

        # adjust the order for roof
        print(self.df[['ROOF', 'sale_price_per_sf']].groupby("ROOF").mean())
        df_mean = self.df[['ROOF', 'sale_price_per_sf']].groupby("ROOF").mean()
        new_index = df_mean.index.to_list()
        df_mean.sort_values(by=['sale_price_per_sf'], inplace=True)
        old_index = df_mean.index.to_list()
        self.df['ROOF'] = self.df['ROOF'].replace(old_index, new_index)

        # keep copy of original sale month as sale_month_raw
        self.df['sale_month_raw'] = self.df['sale_month']

        # adjust the order for sale month
        print(self.df[['sale_month', 'sale_price_per_sf']].groupby("sale_month").mean())
        df_mean = self.df[['sale_month', 'sale_price_per_sf']].groupby("sale_month").mean()
        new_index = df_mean.index.to_list()
        df_mean.sort_values(by=['sale_price_per_sf'], inplace=True)
        old_index = df_mean.index.to_list()
        self.df['sale_month'] = self.df['sale_month'].replace(old_index, new_index)

        # convert some categorical data into numerical ones
        self.df['GRADE'] = self.df['GRADE'].replace({
                                           'XX+': 1, 'XX': 2, 'XX-': 3,
                                           'X+': 4, 'X': 5, 'X-': 6,
                                           'A+': 7, 'A': 8, 'A-': 9,
                                           'B+': 10, 'B': 11, 'B-': 12,
                                           'C+': 13, 'C': 14, 'C-': 15,
                                           'D+': 16, 'D': 17, 'D-': 18,
                                           'E+': 19, 'E': 20, 'E-': 21,
                                           'Unclear': 22
                                           })

        self.df['CDU'] = self.df['CDU'].replace({
                                       'EX': 1, 'VG': 2, 'GD': 3, 'AV': 4,
                                       'FR': 5, 'PR': 6, 'VP': 7, 'UN': 8,
                                       'Unclear': 9
                                       })

        # convert bath rooms
        self.df['bath'] = self.df['FULLBATHS'] + 0.5 * self.df['HALFBATHS']

        # ## outliner removal
        # df['z_score_sale_price'] = abs(df['SALEPRICE'] - np.mean(df['SALEPRICE']))/np.std(df['SALEPRICE'])
        # df = df[df['z_score_sale_price'] < 3].reset_index(drop=True)

        ## log transformation
        self.df['CDU'] = np.log(self.df['CDU'])
        self.df['CONDITION'] = np.log(self.df['CONDITION'])
        self.df['GRADE'] = np.log(self.df['GRADE'])
        mask_tmp = (self.df['LOTAREA'] > 0)
        self.df.loc[mask_tmp, 'LOTAREA'] = np.log(self.df.loc[mask_tmp, 'LOTAREA'])

        ## remove raw features which have been preprocessing
        self.df = self.df[self.df.columns[
            ~self.df.columns.isin(['YEARBLT', 'SALEDATE', 'FULLBATHS', 'HALFBATHS', 'z_score_sale_price'])]]

        self.df = self.df.sort_values(by=['STYLEDESC', 'SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month_raw', 'sale_day'],
                            ascending=True).reset_index(drop=True)
