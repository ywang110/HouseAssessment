from ModelInterface import ModelInterface
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np


class PricingModel(ModelInterface):

    def __init__(self):
        super().__init__()

        self.x_train = {}
        self.y_train = {}
        self.x_test = {}
        self.y_test = {}
        self.y_predicted = {}
        self.model = {}
        self.all_styles = None

    def feature_engineering(self):

        ## load cleaned dataframe
        self.df = pd.read_csv(self.data_path + self.clean_file_name)

        self.all_styles = pd.unique(self.df['STYLEDESC'])

        self.df = self.df.sort_values(by=['STYLEDESC', 'SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month_raw', 'sale_day'],
                                      ascending=True).reset_index(drop=True)

        # ## create dummy variables for categorical data
        # dummy_variables = ['PROPERTYZIP', 'SCHOOLCODE', 'sale_quarter']
        dummy_variables = ['PROPERTYZIP', 'SCHOOLCODE']
        # dummy_quarter = pd.get_dummies(df['sale_quarter'], prefix='quarter')
        dummy_zip = pd.get_dummies(self.df['PROPERTYZIP'], prefix='zip')
        dummy_school = pd.get_dummies(self.df['SCHOOLCODE'], prefix='school')

        # remove the first one of the dummy columns to avoid colinearity
        # dummy_quarter = dummy_quarter[dummy_quarter.columns[~dummy_quarter.columns.isin(['quarter_1'])]]
        dummy_zip = dummy_zip[dummy_zip.columns[~dummy_zip.columns.isin(['zip_15003'])]]
        dummy_school = dummy_school[dummy_school.columns[~dummy_school.columns.isin(['school_1'])]]

        # remove categorical data after replacing by dummy variables
        self.df = self.df[self.df.columns[~self.df.columns.isin(dummy_variables)]]
        # create final data
        # df = pd.concat([df, dummy_quarter, dummy_zip, dummy_school], axis=1)
        self.df = pd.concat([self.df, dummy_zip, dummy_school], axis=1)

    def split_data(self):
        ## split data into training and testing
        ## train the model 5 years before, and test the model for the recent 5 years
        self.df_train = self.df[self.df['years_to_sale'] > self.predict_years]
        self.df_test = self.df[self.df['years_to_sale'] <= self.predict_years]

        # futher split data based on house style, train one model for each style
        self._split_data_per_style()

    def _split_data_per_style(self):

        x_col_removal = ['STYLEDESC', 'SALEPRICE', 'sale_price_per_sf', 'sale_year', 'sale_month_raw', 'sale_day']
        y_col = 'sale_price_per_sf'
        ## one model for each STYLEDESC
        for style in self.all_styles:
            style_df_train = self.df_train[self.df_train['STYLEDESC'] == style]
            style_df_test = self.df_test[self.df_test['STYLEDESC'] == style]

            if style_df_train is None or style_df_test is None or style_df_train.shape[0] == 0 or style_df_test.shape[0] == 0:
                continue

            ## for training data, remove the top and bottom 5% noise
            qt_95 = style_df_train['sale_price_per_sf'].quantile(0.95)
            qt_5 = style_df_train['sale_price_per_sf'].quantile(0.05)
            style_df_train = style_df_train[
                (style_df_train['sale_price_per_sf'] < qt_95) & (style_df_train['sale_price_per_sf'] > qt_5)]

            if style_df_train is None or style_df_test is None or style_df_train.shape[0] == 0 or style_df_test.shape[0] == 0:
                continue

            self.x_train[style] = style_df_train[style_df_train.columns[~style_df_train.columns.isin(x_col_removal)]].to_numpy()
            self.y_train[style] = style_df_train[y_col].to_numpy()
            self.x_test[style] = style_df_test[style_df_test.columns[~style_df_test.columns.isin(x_col_removal)]].to_numpy()
            self.y_test[style] = style_df_test[y_col].to_numpy()

    def train_rf_regression(self):
        # train one model for each style
        self.model_name = 'rf_regression_model'
        for style in self.all_styles:
            if style in self.model.keys():
                self.model[style] = RandomForestRegressor(max_depth=2, random_state=0)
                self.model[style].fit(self.x_train[style], self.y_train[style])
        self.save_model(self.model_name)

    def train_xgb(self):
        # train one model for each style
        self.model_name = 'xgb_model'
        for style in self.all_styles:
            if style in self.model.keys():
                self.model[style] = xgb.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
                self.model[style].fit(self.x_train[style], self.y_train[style])
        self.save_model(self.model_name)

    def train_lasso(self):
        # train one model for each style
        self.model_name = 'lasso_model'
        for style in self.all_styles:
            self.model[style] = Lasso(alpha=0.001, normalize=True, max_iter=5000)
            self.model[style].fit(self.x_train[style], self.y_train[style])
        self.save_model(self.model_name)

    def predict(self):
        self.load_model(self.model_name)
        for style in self.all_styles:
            self.y_predicted[style] = self.model[style].predict(self.x_test[style])
            print(style, 'cnt: ', self.y_test[style].shape[0], ' ', self.model_name,' median abs error: ', np.median(abs(self.y_test[style] - self.y_predicted[style]) / self.y_test[style]))


if __name__ =="__main__":
    pricing_model = PricingModel()

    pricing_model.preprocessing()
    pricing_model.feature_engineering()
    pricing_model.split_data()

    pricing_model.train_lasso()
    pricing_model.predict()

    pricing_model.train_rf_regression()
    pricing_model.predict()

    pricing_model.train_xgb()
    pricing_model.predict()

