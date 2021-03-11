import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import seaborn as sn
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None


class ModelInterface:
    """
    Root level of Model Interface
    """
    def __init__(self):
        self.raw_file_name ="assessments.csv"
        self.clean_file_name ="df_cleaned.csv"
        self.response_col = "sale_price_per_sf"

        self.parentPath = os.path.abspath("../") + "/"
        self.data_path = self.parentPath + "data/"
        self.model_path = self.parentPath + "model/"
        self.figure_path = self.parentPath + "figure/"
        self.from_year = 2000
        self.predict_years = 5
        self.df = None
        self.df_plt = None
        self.df_train = None
        self.df_test = None
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_predicted = None
        self.model_name = None

        self.df_out = None

        if os.path.exists(self.data_path) is False:
            os.mkdir(self.data_path)
        if os.path.exists(self.figure_path) is False:
            os.mkdir(self.figure_path)
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)

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

        ## filter
        self._filter()

        ## transform features
        self._transformation()

        ## save data as cleaned dataframe
        self.df.to_csv(self.data_path + self.clean_file_name, index=False)

    """
    a manual filter to select data set
    """
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

        ## input raw data
        self.df = self.df[[
            'PROPERTYZIP',
            'SCHOOLCODE',
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
            'ROOF',
            'BASEMENT',
            'GRADE',
            'CONDITION',
            'CDU',
            'BEDROOMS',
            'FULLBATHS',
            'HALFBATHS',
            'FIREPLACES',
            'BSMTGARAGE',
            'FINISHEDLIVINGAREA',
            'LOTAREA'
        ]]


    """
    Redo the feature encoding after sorting by response
    A converting categorical data to numerical data step
    """
    def redo_encoding(self, feature_name='sale_quarter', response_name='sale_price_per_sf'):

        ## print(self.df[[feature_name, response_name]].groupby(feature_name).mean())
        df_mean = self.df[[feature_name, response_name]].groupby(feature_name).mean()
        new_index = df_mean.index.to_list()
        df_mean.sort_values(by=[response_name], inplace=True)
        old_index = df_mean.index.to_list()
        self.df[feature_name] = self.df[feature_name].replace(old_index, new_index)
        ## print(self.df[[feature_name, response_name]].groupby(feature_name).mean())


    """
    transform the data inclluding nan filing, data convertion, encoding, log transformation, etc
    """
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
        # self.df['EXTERIORFINISH'] = self.df['EXTERIORFINISH'].apply(int)

        ## price per square feet
        self.df['sale_price_per_sf'] = self.df['SALEPRICE'] / self.df['FINISHEDLIVINGAREA']

        # # plot
        # self.plot_categorical_to_numerical("CONDITION")

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
                                       'UN': 0, 'EX': 1, 'VG': 2, 'GD': 3, 'AV': 4,
                                       'FR': 5, 'PR': 6, 'VP': 7, 'Unclear': 8
                                       })

        ## adjust the order of some numbers based on plots
        self.redo_encoding('sale_quarter', self.response_col)

        self.redo_encoding('ROOF', self.response_col)

        # keep copy of original sale month as sale_month_raw
        self.df['sale_month_raw'] = self.df['sale_month']
        self.redo_encoding('sale_month', self.response_col)

        # adjust the order for CDU
        self.redo_encoding('CDU', self.response_col)

        # adjust the order for CONDITION
        self.redo_encoding('CONDITION', self.response_col)

        # adjust the order for GRADE
        self.redo_encoding('GRADE', self.response_col)

        # convert bath rooms
        self.df['bath'] = self.df['FULLBATHS'] + 0.5 * self.df['HALFBATHS']
        # adjust the order for bath
        self.redo_encoding('bath', self.response_col)

        # ## normalize for feature importance
        # self.df['sale_quarter'] = self.df['sale_quarter'] / max(self.df['sale_quarter'])
        # self.df['sale_month'] = self.df['sale_month'] / max(self.df['sale_month'])
        # self.df['CDU'] = self.df['CDU'] / max(self.df['CDU'])
        # self.df['GRADE'] = self.df['GRADE'] / max(self.df['GRADE'])
        # self.df['bath'] = self.df['bath'] / max(self.df['bath'])
        # self.df['years_to_sale'] = self.df['years_to_sale'] / max(self.df['years_to_sale'])
        # self.df['years_built'] = self.df['years_built'] / max(self.df['years_built'])
        # self.df['STORIES'] = self.df['STORIES']/max( self.df['STORIES'])
        # self.df['CONDITION'] = self.df['CONDITION']/max( self.df['CONDITION'])
        # self.df['BEDROOMS'] = self.df['BEDROOMS']/max( self.df['BEDROOMS'])
        # self.df['BASEMENT'] = self.df['BASEMENT']/max( self.df['BASEMENT'])
        # self.df['ROOF'] = self.df['ROOF']/max( self.df['ROOF'])
        # self.df['FIREPLACES'] = self.df['FIREPLACES']/max( self.df['FIREPLACES'])
        # self.df['BSMTGARAGE'] = self.df['BSMTGARAGE']/max( self.df['BSMTGARAGE'])
        # self.df['LOTAREA'] = self.df['LOTAREA']/max( self.df['LOTAREA'])

        # ## outliner removal -- will do it later for each style
        # df['z_score_sale_price'] = abs(df['SALEPRICE'] - np.mean(df['SALEPRICE']))/np.std(df['SALEPRICE'])
        # df = df[df['z_score_sale_price'] < 3].reset_index(drop=True)

        # ## feature log transformation, does not work well
        # self.df['CDU'] = np.log(self.df['CDU'])
        # self.df['CONDITION'] = np.log(self.df['CONDITION'])
        # self.df['GRADE'] = np.log(self.df['GRADE'])
        # mask_tmp = (self.df['LOTAREA'] > 0)
        # self.df.loc[mask_tmp, 'LOTAREA'] = np.log(self.df.loc[mask_tmp, 'LOTAREA'])

        ## remove raw features which have been preprocessing
        self.df = self.df[self.df.columns[
            ~self.df.columns.isin(['YEARBLT', 'SALEDATE', 'FULLBATHS', 'HALFBATHS', 'z_score_sale_price'])]]

        self.df = self.df.sort_values(by=['STYLEDESC', 'SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month_raw', 'sale_day'],
                            ascending=True).reset_index(drop=True)


    def plot_compare_response(self, x_name='GRADE'):

        self.df['price'] = self.df['SALEPRICE'] / max(self.df['SALEPRICE'])
        self.df['log(price)'] = np.log(self.df['SALEPRICE']) / max(np.log(self.df['SALEPRICE']))
        self.df['price per square foot'] = self.df['sale_price_per_sf'] / max(self.df['sale_price_per_sf'])

        fig1 = plt.figure(figsize=(6, 6))

        # adjust the order for x_name feature
        self.redo_encoding(x_name, "price")
        ax = fig1.add_subplot(3, 1, 1)
        sn.lineplot(x=self.df[x_name], y=self.df['price'], ci=80, markers=True,
                    err_style="bars", legend='full', ax=ax, label=str('price'))

        # adjust the order for x_name feature
        self.redo_encoding(x_name, "log(price)")
        ax = fig1.add_subplot(3, 1, 2)
        sn.lineplot(x=self.df[x_name], y=self.df['log(price)'], ci=80, markers=True,
                    err_style="bars", legend='full', ax=ax, label=str('log(price)'))

        # # adjust the order for x_name feature
        self.redo_encoding(x_name, "price per square foot")
        ax = fig1.add_subplot(3, 1, 3)
        sn.lineplot(x=self.df[x_name], y=self.df['price per square foot'], ci=80, markers=True,
                    err_style="bars", legend='full', ax=ax, label=str('price per square foot'))

    def plot_categorical_to_numerical(self, x_name='GRADE'):

        fig1 = plt.figure(figsize=(6, 6))

        # before adjusting the order for x_name feature
        ax = fig1.add_subplot(2, 1, 1)
        sn.lineplot(x=self.df[x_name], y=self.df[self.response_col], ci=80, markers=True,
                    err_style="bars", legend='full', ax=ax, label='categorical data')

        ## adjust the order of some numbers based on plots
        self.redo_encoding(x_name, self.response_col)

        ## plot after adjusting order
        ax = fig1.add_subplot(2, 1, 2)
        sn.lineplot(x=self.df[x_name], y=self.df[self.response_col], ci=80, markers=True,
                    err_style="bars", legend='full', ax=ax, label='numerical data')


    def get_performance_metric(self):
        df_all = None
        for file in os.listdir(self.data_path):
            if "df_out_" in file:
                df = pd.read_csv(self.data_path + file)
                df_all = pd.concat([df_all, df], axis=0)

        if df_all is not None:
            ##rename model names
            df_all.loc[df_all["model"] == "quality_model", "model"] = "quality"
            df_all.loc[df_all["model"] == "lasso_single_model", "model"] = "lasso one"
            df_all.loc[df_all["model"] == "rfg_single_model", "model"] = "rf one"
            df_all.loc[df_all["model"] == "xgb_single_model", "model"] = "xfb one"
            df_all.loc[df_all["model"] == "lasso_style_model", "model"] = "lasso style"
            df_all.loc[df_all["model"] == "rf_regression_style_model", "model"] = "rf style"
            df_all.loc[df_all["model"] == "xgb_style_model", "model"] = "xgb style"

            df_all['error_rate'] = abs(df_all['y'] - df_all['y_hat'])/df_all['y']

            df_error_rate = df_all.groupby(["model"])["error_rate"].median().reset_index()

            df_error_rate_per_style = df_all.groupby(["style", "model"])["error_rate"].median().reset_index()

            fig1 = plt.figure(figsize=(6, 6))
            fig1.add_subplot(1, 1, 1)
            ax = sn.barplot(x="model", y="error_rate", data=df_error_rate)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.get_figure().savefig(self.figure_path + "all_model_error.png")

            fig1 = plt.figure(figsize=(6, 6))
            fig1.add_subplot(1, 1, 1)
            ax = sn.barplot(x="style", y="error_rate", hue="model", data=df_error_rate_per_style)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.get_figure().savefig(self.figure_path + "style_model.png")


if __name__ =="__main__":
    model_interface = ModelInterface()

    model_interface.get_performance_metric()
