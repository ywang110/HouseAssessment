import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sn
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pickle

parentPath = os.path.abspath("../")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

def data_preprocessing():
    df = pd.read_csv(parentPath + "/data/assessments.csv")

    # create the unique address for each house
    df['address'] = df['PROPERTYHOUSENUM'].apply(str) + ' ' + df['PROPERTYADDRESS'].apply(str) + ' ' + df[
        'PROPERTYUNIT'].apply(str) + ' ' + df['PROPERTYZIP'].apply(str)

    # the percentage of properties which are unique in the data: 86.57 %
    # which means there is only less than 15% of properties have more than 1 time transaction record
    # so we cannot find the trend of price based on aggregating of each single property across the time.
    unique_rate = len(df['address'].unique()) / len(df['address'])

    ## convert the previous sales into rows
    df_curr_sale = df[
        df.columns[~df.columns.isin(['PREVSALEDATE', 'PREVSALEPRICE', 'PREVSALEPRICE2', 'PREVSALEDATE2'])]]
    df_prev_sale = df[df.columns[~df.columns.isin(['SALEDATE', 'SALEPRICE', 'PREVSALEPRICE2', 'PREVSALEDATE2'])]]
    df_prev_sale_2 = df[df.columns[~df.columns.isin(['SALEDATE', 'SALEPRICE', 'PREVSALEPRICE', 'PREVSALEDATE'])]]

    df_prev_sale.rename(columns={"PREVSALEDATE": "SALEDATE", "PREVSALEPRICE": "SALEPRICE"}, errors="raise",
                        inplace=True)
    df_prev_sale_2.rename(columns={"PREVSALEDATE2": "SALEDATE", "PREVSALEPRICE2": "SALEPRICE"}, errors="raise",
                          inplace=True)

    ## combine current sale price and previous sale prices
    df = pd.concat([df_curr_sale, df_prev_sale, df_prev_sale_2]).reset_index(drop=True)

    # filters out noises:
    # only taxable properties
    # sale price is larger than $100
    # not empty in style, stories, yearbuild, bedrooms,
    filter_mask = (df['TAXCODE'] == 'T') & \
                  (~df['TAXCODE'].isna()) & \
                  (df['SALEPRICE'] > 20000) & \
                  (~df['SALEPRICE'].isna()) & \
                  (df['BEDROOMS'] > 0) & \
                  (~df['BEDROOMS'].isna()) & \
                  (df['FINISHEDLIVINGAREA'] > 0) & \
                  (~df['FINISHEDLIVINGAREA'].isna()) & \
                  (~df['PROPERTYZIP'].isna()) & \
                  (~df['SCHOOLCODE'].isna()) & \
                  (~df['STYLEDESC'].isna()) & \
                  (~df['STORIES'].isna()) & \
                  (~df['YEARBLT'].isna()) & \
                  (~df['FULLBATHS'].isna()) & \
                  (~df['BEDROOMS'].isna())

    df = df[filter_mask].reset_index(drop=True)

    # mannully selected input raw data
    df = df[[  # 'address',
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

    ## date preprocessing

    # how many years since built
    df['years_built'] = int(datetime.today().year) - df['YEARBLT'].apply(int)

    # get sale year and month
    df['SALEDATE'] = pd.to_datetime(df['SALEDATE'], format="%m-%d-%Y", errors='coerce')
    df = df[~df['SALEDATE'].isna()].reset_index(drop=True)
    tmp = pd.DatetimeIndex(df['SALEDATE'])
    df['sale_year'] = tmp.year
    df['sale_month'] = tmp.month
    df['sale_day'] = tmp.day
    df['years_to_sale'] = int(datetime.today().year) - tmp.year

    df['sale_quarter'] = tmp.quarter

    ### use recent 10 years of data
    df = df[df['sale_year'] >= 2000].reset_index(drop=True) ##1970, 2000

    ## fill na
    df['CONDITION'] = df['CONDITION'].fillna(9)
    df['HALFBATHS'] = df['HALFBATHS'].fillna(0)
    df['BASEMENT'] = df['BASEMENT'].fillna(1)
    df['ROOF'] = df['ROOF'].fillna(7)
    df['FIREPLACES'] = df['FIREPLACES'].fillna(0)
    df['BSMTGARAGE'] = df['BSMTGARAGE'].fillna(0)
    df['LOTAREA'] = df['LOTAREA'].fillna(0)
    df['STORIES'] = df['STORIES'].fillna(1)
    df['GRADE'] = df['GRADE'].fillna('Unclear')
    df['CDU'] = df['CDU'].fillna('Unclear')

    ## data type conversion

    # convert some numerical data to categorical ones
    df['PROPERTYZIP'] = df['PROPERTYZIP'].apply(int).apply(str)
    df['SCHOOLCODE'] = df['SCHOOLCODE'].apply(int).apply(str)
    # df['NEIGHCODE'] = df['NEIGHCODE'].apply(str)
    df['STYLEDESC'] = df['STYLEDESC'].apply(str)
    # df['EXTERIORFINISH'] = df['EXTERIORFINISH'].apply(int).apply(str)
    #df['sale_quarter'] = df['sale_quarter'].apply(str)

    # convert some string data to  numerical ones
    df['STORIES'] = df['STORIES'].apply(int)
    df['CONDITION'] = df['CONDITION'].apply(int)
    df['BEDROOMS'] = df['BEDROOMS'].apply(int)
    df['FULLBATHS'] = df['FULLBATHS'].apply(int)
    df['HALFBATHS'] = df['HALFBATHS'].apply(int)
    df['BASEMENT'] = df['BASEMENT'].apply(int)
    df['ROOF'] = df['ROOF'].apply(int)
    df['FIREPLACES'] = df['FIREPLACES'].apply(int)
    df['BSMTGARAGE'] = df['BSMTGARAGE'].apply(int)
    df['LOTAREA'] = df['LOTAREA'].apply(int)

    ## adjust some numbers based on plots

    # exchange 2 by 4, and 5 by 6 for roof
    mask_tmp = (df['ROOF'] == 2)
    mask_tmp_2 = (df['ROOF'] == 4)
    df.loc[mask_tmp, 'ROOF'] = 4
    df.loc[mask_tmp_2, 'ROOF'] = 2

    mask_tmp = (df['ROOF'] == 5)
    mask_tmp_2 = (df['ROOF'] == 6)
    df.loc[mask_tmp, 'ROOF'] = 6
    df.loc[mask_tmp_2, 'ROOF'] = 5

    mask_tmp = (df['sale_quarter'] == 2)
    mask_tmp_2 = (df['sale_quarter'] == 3)
    df.loc[mask_tmp, 'sale_quarter'] = 3
    df.loc[mask_tmp_2, 'sale_quarter'] = 2

    mask_tmp = (df['sale_quarter'] == 4)
    mask_tmp_2 = (df['sale_quarter'] == 2)
    df.loc[mask_tmp, 'sale_quarter'] = 2
    df.loc[mask_tmp_2, 'sale_quarter'] = 4

    # convert some categorical data into numerical ones
    df['GRADE'] = df['GRADE'].replace({'XX+': 1, 'XX': 2, 'XX-': 3,
                                       'X+': 4, 'X': 5, 'X-': 6,
                                       'A+': 7, 'A': 8, 'A-': 9,
                                       'B+': 10, 'B': 11, 'B-': 12,
                                       'C+': 13, 'C': 14, 'C-': 15,
                                       'D+': 16, 'D': 17, 'D-': 18,
                                       'E+': 19, 'E': 20, 'E-': 21,
                                       'Unclear': 22
                                       })

    df['CDU'] = df['CDU'].replace({'EX': 1, 'VG': 2, 'GD': 3, 'AV': 4,
                                   'FR': 5, 'PR': 6, 'VP': 7, 'UN': 8,
                                   'Unclear': 8
                                   })

    # convert bath rooms
    df['bath'] = df['FULLBATHS'] + 0.5 * df['HALFBATHS']

    # get sale price per square feet
    df['sale_price_per_sf'] = df['SALEPRICE'] / df['FINISHEDLIVINGAREA']

    # ## outliner removal
    # df['z_score_sale_price'] = abs(df['SALEPRICE'] - np.mean(df['SALEPRICE']))/np.std(df['SALEPRICE'])
    # df = df[df['z_score_sale_price'] < 3].reset_index(drop=True)

    ## log transformation
    df['CDU'] = np.log(df['CDU'])
    df['CONDITION'] = np.log(df['CONDITION'])
    df['GRADE'] = np.log(df['GRADE'])
    mask_tmp = (df['LOTAREA'] > 0)
    df.loc[mask_tmp, 'LOTAREA'] = np.log(df.loc[mask_tmp, 'LOTAREA'])

    ## remove raw features which have been preprocessing
    df = df[df.columns[
        ~df.columns.isin(['YEARBLT', 'SALEDATE', 'FULLBATHS', 'HALFBATHS', 'z_score_sale_price'])]]

    df = df.sort_values(by=['STYLEDESC', 'SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month', 'sale_day'], ascending=True).reset_index(drop=True)

    y = df['SALEPRICE']
    x = df.drop(columns=['SALEPRICE', 'sale_price_per_sf'])

    y2 = df['sale_price_per_sf']
    x2 = df.drop(columns=['SALEPRICE', 'sale_price_per_sf'])

    print(df)

    df.to_csv(parentPath + "/data/df_cleaned.csv", index=False)

    # ax = sn.lineplot(x['years_to_sale'], y / max(y), ci=80)
    # ax = sn.lineplot(x2['years_to_sale'], y2 / max(y2), ci=80)
    # ax = sn.lineplot(np.log(x['years_to_sale']), y / max(y), ci=80)
    # ax = sn.lineplot(np.log(x2['years_to_sale']), y2 / max(y2), ci=80)

def knn_model():

    ##load model from trained lasso model
    model_file = parentPath + "/data/model"
    outfile = open(model_file, 'rb')
    model = pickle.load(outfile)
    outfile.close()

    df = pd.read_csv(parentPath + "/data/df_cleaned.csv")

    df = df.sort_values(by=['STYLEDESC', 'SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month', 'sale_day'],
                        ascending=True).reset_index(drop=True)

    x_col_removal = ['PROPERTYZIP', 'SCHOOLCODE', 'STYLEDESC', 'SALEPRICE', 'sale_price_per_sf', 'sale_year', 'sale_month', 'sale_day']
    y_col = ['PROPERTYZIP', 'SCHOOLCODE', 'sale_year', 'sale_month', 'sale_day', 'SALEPRICE', 'sale_price_per_sf']
    all_styles = pd.unique(df['STYLEDESC'])

    def knn_rolling(df):
        df = np.flipud(df)
        df = list(df)
        if len(df) >= 5:
            df.remove(max(df))
            df.remove(min(df))
        return np.median(df)

    ## for each test data, find it is most recent at most 10 transactions in the past, which come from the same style, zip and school code
    for style in all_styles:
        ## the first 15 features are quality score features
        model_style = model[style]
        quality_score_coeff = model_style.coef_[:15]

        ## calculate quality score
        df_style = df[df['STYLEDESC'] == style]
        df_style_x = df_style[df_style.columns[~df_style.columns.isin(x_col_removal)]]
        df_style_y = df_style[y_col]

        # dot product with coefficients and quality features to get quality score
        df_style_y.loc[:, 'quality_score'] = df_style_x.dot(quality_score_coeff)
        # market score is the y - quality score
        df_style_y.loc[:, 'market_score'] = df_style_y.loc[:, 'sale_price_per_sf'] - df_style_y.loc[:, 'quality_score']

        # use market score of previous transacted properities
        df_style_y['market_score_lag'] = df_style_y['market_score'].shift(periods=1, fill_value=0)

        # use median value of 10 NN market score after removing max and min in NN
        df_style_y.sort_values(by=['SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month', 'sale_day'], inplace=True)
        d = df_style_y.groupby(['SCHOOLCODE', 'PROPERTYZIP'])['market_score_lag']
        s = d.rolling(window=10, center=False, min_periods=0).apply(lambda x: knn_rolling(x), raw=True).reset_index()
        df_style_y.loc[:, 'knn_10_market_score'] = pd.DataFrame(data=s['market_score_lag'].to_numpy(), index=s['level_2'].to_numpy(), columns=['knn_10_market_score'])
        # reset index starting from 0
        df_style_x = df_style_x.reset_index(drop=True)
        df_style_y = df_style_y[df_style_y.columns[~df_style_y.columns.isin(['market_score_lag'])]].reset_index(drop=True)
        # drop the first 2 rows due to the rolling empty fields
        df_style_x = df_style_x.loc[2:, :].reset_index(drop=True)
        df_style_y = df_style_y.loc[2:, :].reset_index(drop=True)

        ## use median value of 10 NN market score to estimate the y
        df_style_y.loc[:, 'y_hat'] = df_style_y.loc[:, 'quality_score'] + df_style_y.loc[:, 'knn_10_market_score']

        ## combine x and y
        df_style_combined = pd.concat([df_style_x, df_style_y], axis=1)

        ## count only testing data set
        df_style_testing = df_style_combined[df_style_combined['years_to_sale']<=5]
        df_style_testing.loc[:, 'error'] = abs(df_style_testing.loc[:, 'y_hat'] - df_style_testing.loc[:, 'sale_price_per_sf'])/df_style_testing.loc[:, 'sale_price_per_sf']
        df_style_testing.sort_values(by=['error'], inplace=True)
        print(style, 'cnt: ', df_style_testing.shape[0], ' KNN median abs error: ', np.median(df_style_testing['error'].to_numpy()))


def regression_model():
    df = pd.read_csv(parentPath + "/data/df_cleaned.csv")

    # ## create dummy variables for categorical data
    #dummy_variables = ['PROPERTYZIP', 'SCHOOLCODE', 'sale_quarter']
    dummy_variables = ['PROPERTYZIP', 'SCHOOLCODE']
    dummy_zip = pd.get_dummies(df['PROPERTYZIP'], prefix='zip')
    dummy_school = pd.get_dummies(df['SCHOOLCODE'], prefix='school')
    #dummy_quarter = pd.get_dummies(df['sale_quarter'], prefix='quarter')
    # remove the first one of the dummy columns to avoid colinearity
    dummy_zip = dummy_zip[dummy_zip.columns[~dummy_zip.columns.isin(['zip_15003'])]]
    dummy_school = dummy_school[dummy_school.columns[~dummy_school.columns.isin(['school_1'])]]
    #dummy_quarter = dummy_quarter[dummy_quarter.columns[~dummy_quarter.columns.isin(['quarter_1'])]]
    # create final data
    df = df[df.columns[~df.columns.isin(dummy_variables)]]
    #df = pd.concat([df, dummy_zip, dummy_school, dummy_quarter], axis=1)
    df = pd.concat([df, dummy_zip, dummy_school], axis=1)

    ## split data into training and testing
    ## train the model 5 years before, and test the model for the recent 5 years
    df_train = df[df['years_to_sale'] > 5]
    df_test = df[df['years_to_sale'] <= 5]

    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    y_predicted = {}
    model = {}

    x_col_removal = ['STYLEDESC', 'SALEPRICE', 'sale_price_per_sf', 'sale_year', 'sale_month', 'sale_day']
    y_col = 'sale_price_per_sf'
    ## training one model for each STYLEDESC
    all_styles = pd.unique(df['STYLEDESC'])
    for style in all_styles:
        style_df_train = df_train[df_train['STYLEDESC'] == style]
        style_df_test = df_test[df_test['STYLEDESC'] == style]

        if style_df_train is None or style_df_test is None or style_df_train.shape[0] == 0 or style_df_test.shape[0] == 0:
            continue

        ## for training data, remove the top and bottom 5% noise
        qt_95 = style_df_train['sale_price_per_sf'].quantile(0.95)
        qt_5 = style_df_train['sale_price_per_sf'].quantile(0.05)
        style_df_train = style_df_train[(style_df_train['sale_price_per_sf'] < qt_95) & (style_df_train['sale_price_per_sf'] > qt_5)]

        if style_df_train is None or style_df_test is None or style_df_train.shape[0] == 0 or style_df_test.shape[0] == 0:
            continue

        ##style_df_train[style_df_train.columns[~style_df_train.columns.isin(x_col_removal)]].to_csv("example_train.csv")

        x_train[style] = style_df_train[style_df_train.columns[~style_df_train.columns.isin(x_col_removal)]].to_numpy()
        y_train[style] = style_df_train[y_col].to_numpy()
        x_test[style] = style_df_test[style_df_test.columns[~style_df_test.columns.isin(x_col_removal)]].to_numpy()
        y_test[style] = style_df_test[y_col].to_numpy()

        # model[style] = linear_model.LinearRegression(normalize=True)
        # model[style].fit(x_train[style], y_train[style])
        # y_predicted[style] = model[style].predict(x_test[style])
        # print(style, 'cnt: ', y_test[style].shape[0], ' regression median abs error: ', np.median(abs(y_test[style] - y_predicted[style])/y_test[style]))

        model[style] = RandomForestRegressor(max_depth=2, random_state=0)
        model[style].fit(x_train[style], y_train[style])
        y_predicted[style] = model[style].predict(x_test[style])
        print(style, 'cnt: ', y_test[style].shape[0], ' RF median abs error: ',
              np.median(abs(y_test[style] - y_predicted[style]) / y_test[style]))

        model[style] = xgb.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
        model[style].fit(x_train[style], y_train[style])
        y_predicted[style] = model[style].predict(x_test[style])
        print(style, 'cnt: ', y_test[style].shape[0], ' xgboost median abs error: ',
              np.median(abs(y_test[style] - y_predicted[style]) / y_test[style]))

        model[style] = Lasso(alpha=0.001, normalize=True, max_iter=5000)
        model[style].fit(x_train[style], y_train[style])
        y_predicted[style] = model[style].predict(x_test[style])
        print(style, 'cnt: ', y_test[style].shape[0], ' lasso median abs error: ', np.median(abs(y_test[style] - y_predicted[style]) / y_test[style]))

    model_file = parentPath + "/data/model"
    outfile = open(model_file, 'wb')
    pickle.dump(model, outfile)
    outfile.close()
    #print(df)


if __name__ =="__main__":

    # data_preprocessing()
    #
    # regression_model()

    knn_model()