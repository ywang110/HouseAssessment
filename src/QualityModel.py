import pandas as pd
import numpy as np
from PricingModel import PricingModel


class QualityModel(PricingModel):

    def __init__(self):
        super().__init__()

        self.df_quality_model = {}
        self.df_test_quality_model = {}
        self.model_name = "lasso_model"

    def learn_quality(self, k=10):

        ##load model from trained lasso model
        self.load_model(self.model_name)

        ##load cleaned dataframe
        self.df = pd.read_csv(self.data_path + self.clean_file_name)

        self.df = self.df.sort_values(by=['STYLEDESC', 'SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month_raw', 'sale_day'],
                                      ascending=True).reset_index(drop=True)

        x_col_removal = ['PROPERTYZIP', 'SCHOOLCODE', 'STYLEDESC', 'SALEPRICE', 'sale_price_per_sf', 'sale_year',
                         'sale_month_raw', 'sale_day']
        y_col = ['PROPERTYZIP', 'SCHOOLCODE', 'sale_year', 'sale_month_raw', 'sale_day', 'SALEPRICE',
                 'sale_price_per_sf']

        self.all_styles = pd.unique(self.df['STYLEDESC'])

        def knn_rolling(df):
            df = np.flipud(df)
            df = list(df)
            if len(df) >= 5:
                df.remove(max(df))
                df.remove(min(df))
            return np.median(df)

        ## for each test data, find it is most recent at most 10 transactions in the past, which come from the same style, zip and school code
        for style in self.all_styles:
            ## the first 15 features are quality score features
            model_style = self.model[style]
            quality_score_coeff = model_style.coef_[:16]  # model_style.coef_[:17]

            ## calculate quality score
            df_style = self.df[self.df['STYLEDESC'] == style]
            df_style_x = df_style[df_style.columns[~df_style.columns.isin(x_col_removal)]]
            df_style_y = df_style[y_col]

            # dot product with coefficients and quality features to get quality score
            df_style_y.loc[:, 'quality_score'] = df_style_x.dot(quality_score_coeff)
            # market score is the y - quality score
            df_style_y.loc[:, 'market_score'] = df_style_y.loc[:, 'sale_price_per_sf'] - df_style_y.loc[:, 'quality_score']

            # use market score of previous transacted properties
            df_style_y['market_score_lag'] = df_style_y['market_score'].shift(periods=1, fill_value=0)

            # use median value of 10 NN market score after removing max and min in NN
            df_style_y.sort_values(by=['SCHOOLCODE', 'PROPERTYZIP', 'sale_year', 'sale_month_raw', 'sale_day'],
                                   inplace=True)
            d = df_style_y.groupby(['SCHOOLCODE', 'PROPERTYZIP'])['market_score_lag']
            s = d.rolling(window=k, center=False, min_periods=0).apply(lambda x: knn_rolling(x), raw=True).reset_index()
            df_style_y.loc[:, 'knn_10_market_score'] = pd.DataFrame(data=s['market_score_lag'].to_numpy(),
                                                                    index=s['level_2'].to_numpy(),
                                                                    columns=['knn_10_market_score'])
            # reset index starting from 0
            df_style_x = df_style_x.reset_index(drop=True)
            df_style_y = df_style_y[df_style_y.columns[~df_style_y.columns.isin(['market_score_lag'])]].reset_index(
                drop=True)
            # drop the first 2 rows due to the rolling empty fields
            df_style_x = df_style_x.loc[2:, :].reset_index(drop=True)
            df_style_y = df_style_y.loc[2:, :].reset_index(drop=True)

            ## use median value of 10 NN market score to estimate the y
            df_style_y.loc[:, 'y_hat'] = df_style_y.loc[:, 'quality_score'] + df_style_y.loc[:, 'knn_10_market_score']

            ## combine x and y
            self.df_quality_model[style] = pd.concat([df_style_x, df_style_y], axis=1)

    def predict(self):
        for style in self.all_styles:
            ## use only testing data set
            self.df_test_quality_model[style] = self.df_quality_model[style][self.df_quality_model[style]['years_to_sale'] <= self.predict_years]

            self.df_test_quality_model[style].loc[:, 'error'] = abs(self.df_test_quality_model[style].loc[:, 'y_hat'] - self.df_test_quality_model[style].loc[:, 'sale_price_per_sf']) / self.df_test_quality_model[style].loc[:,'sale_price_per_sf']
            self.df_test_quality_model[style].sort_values(by=['error'], inplace=True)
            print(style, 'cnt: ', self.df_test_quality_model[style].shape[0], ' Quality Model median abs error: ',
                  np.median(self.df_test_quality_model[style]['error'].to_numpy()))

if __name__ =="__main__":
    quality_model = QualityModel()

    # quality_model.preprocessing()
    # quality_model.feature_engineering()
    # quality_model.split_data()
    # quality_model.train_lasso()

    quality_model.learn_quality()
    quality_model.predict()