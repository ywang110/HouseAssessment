import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from PricingModel import PricingModel


class SchoolRank(PricingModel):

    def __init__(self):
        super().__init__()

        self.x_train = None
        self.y_train = None
        self.x_test = {}
        self.y_test = {}
        self.y_predicted = {}
        self.model = None
        self.all_styles = None
        self.model_name = 'lasso_single_model'
        self.feature_columns = None
        self.rank = None

    def feature_engineering(self):

        ## load cleaned dataframe
        self.df = pd.read_csv(self.data_path + self.clean_file_name)

        self.all_styles = pd.unique(self.df['STYLEDESC'])

        self.df = self.df.sort_values(by=['SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month_raw', 'sale_day'],
                                      ascending=True).reset_index(drop=True)

        # ## create dummy variables for categorical data
        dummy_variables = ['SCHOOLCODE', 'PROPERTYZIP'] #, 'STYLEDESC']

        dummy_school = pd.get_dummies(self.df['SCHOOLCODE'], prefix='school')
        dummy_zip = pd.get_dummies(self.df['PROPERTYZIP'], prefix='zip')
        dummy_style = pd.get_dummies(self.df['STYLEDESC'], prefix='style')

        # remove the first one of the dummy columns to avoid colinearity
        # dummy_quarter = dummy_quarter[dummy_quarter.columns[~dummy_quarter.columns.isin(['quarter_1'])]]
        dummy_school = dummy_school[dummy_school.columns[~dummy_school.columns.isin(['school_1'])]]
        dummy_zip = dummy_zip[dummy_zip.columns[~dummy_zip.columns.isin(['zip_15003'])]]
        dummy_style = dummy_style[dummy_style.columns[~dummy_style.columns.isin(['style_BI-LEVEL'])]]

        # remove categorical data after replacing by dummy variables
        self.df = self.df[self.df.columns[~self.df.columns.isin(dummy_variables)]]
        # create final data
        self.df = pd.concat([self.df, dummy_school, dummy_zip, dummy_style], axis=1)

    def _split_data_per_style(self):

        x_col_removal = ['SALEPRICE', 'sale_price_per_sf', 'sale_year', 'sale_month_raw', 'sale_day'] + ['STYLEDESC']
        y_col = 'sale_price_per_sf'

        ## for training data, remove the top and bottom 5% noise
        qt_95 = self.df_train['sale_price_per_sf'].quantile(0.95)
        qt_5 = self.df_train['sale_price_per_sf'].quantile(0.05)
        self.df_train = self.df_train[(self.df_train['sale_price_per_sf'] < qt_95) & (self.df_train['sale_price_per_sf'] > qt_5)]

        ## use a single model for training
        self.x_train = self.df_train[self.df_train.columns[~self.df_train.columns.isin(x_col_removal)]].to_numpy()
        self.y_train = self.df_train[y_col].to_numpy()

        self.feature_columns = self.df_train[self.df_train.columns[~self.df_train.columns.isin(x_col_removal)]].columns

        ## for testing data, split by style
        for style in self.all_styles:
            ##style_df_train = self.df_train[self.df_train['STYLEDESC'] == style]
            style_df_test = self.df_test[self.df_test['STYLEDESC'] == style]

            if style_df_test is None or style_df_test.shape[0] == 0:
                continue
            self.x_test[style] = style_df_test[style_df_test.columns[~style_df_test.columns.isin(x_col_removal)]].to_numpy()
            self.y_test[style] = style_df_test[y_col].to_numpy()

    def train_lasso(self):
        # train a single model
        self.model = Lasso(alpha=0.001, normalize=True, max_iter=5000)
        self.model.fit(self.x_train, self.y_train)

        self.save_model(self.model_name)

    def predict(self):
        self.load_model(self.model_name)
        for style in self.all_styles:
            self.y_predicted[style] = self.model.predict(self.x_test[style])
            print(style, 'cnt: ', self.y_test[style].shape[0], ' ', self.model_name, ' median abs error: ',
                  np.median(abs(self.y_test[style] - self.y_predicted[style]) / self.y_test[style]))

    def school_rank(self):
        self.model = Lasso(alpha=0.001, normalize=True, max_iter=5000)
        self.model.fit(self.x_train, self.y_train)

        self.coeff = pd.DataFrame([self.feature_columns, self.model.coef_.tolist()]).transpose()

        school_coeff = self.coeff[self.coeff.loc[:,0].str.contains('school')]

        school_coeff.rename(columns={0: 'school_prefix', 1: 'school_quality_score'}, inplace=True)

        school_coeff = school_coeff.append(pd.Series({'school_prefix': 'school_1', 'school_quality_score': 0}), ignore_index=True)

        school_coeff['school_by_score'] = school_coeff['school_prefix'].str.replace('school_', '').astype(int)

        school_coeff = school_coeff.sort_values(by=['school_by_score']).reset_index(drop=True)

        school_coeff['school_by_score'] = school_coeff['school_by_score'].astype(str)

        school_coeff = school_coeff.drop(['school_prefix'], axis=1)

        ## rank based on school quality score
        school_coeff.sort_values(['school_quality_score'], ascending=False, inplace=True)

        school_coeff.reset_index(drop=True, inplace=True)

        ## school ranked based on mean sale_price_per_sf
        ## load cleaned dataframe
        df = pd.read_csv(self.data_path + self.clean_file_name)
        df_train = df[df['years_to_sale'] > self.predict_years]
        school_rank_by_price = df_train.groupby(['SCHOOLCODE'])['SCHOOLCODE', 'sale_price_per_sf'].mean()
        school_rank_by_price.reset_index(drop=True, inplace=True)

        school_rank_by_price.sort_values(['sale_price_per_sf'], ascending=False, inplace=True)

        school_rank_by_price.reset_index(drop=True, inplace=True)

        school_rank_by_price.rename(columns={'SCHOOLCODE': 'school_by_price'}, inplace=True)

        self.rank = pd.concat([school_coeff, school_rank_by_price], axis=1)

        self.rank['rank'] = self.rank.index.values.tolist()

        print(self.rank)


if __name__ =="__main__":
    school_rank = SchoolRank()

    # # school_rank.preprocessing()
    #
    school_rank.feature_engineering()
    school_rank.split_data()

    school_rank.train_lasso()
    # school_rank.predict()

    school_rank.school_rank()
