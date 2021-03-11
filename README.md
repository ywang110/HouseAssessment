# House Pricing Quality Score Model

- A model for house price prediction.
- Author: Yi Wang
- Date: Mar.10.2021

## Installation

- Create your owen virtual environment, run the following command:
```
pip install -r requirements.txt
```

Note: The code has only been tested on Python version 3.8.3 (released on May 13 2020).

## Download Input Data 

- The most recent updated 2021 data is used in the project. You may download data set `assessments.csv` from [here](https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/f2b8d575-e256-4718-94ad-1e12239ddb92/download/assessments.csv), and put the file under `src/data/` folder

## Run the code

- Run each algorithm: Create an instance from the following Classes and call run() method of each instance

### QualityModel

- QualityModel implements the project idea for calculating house quality score. It trains one model for each style and score the house price by KNN.
```
quality_model = QualityModel()
quality_model.run()
```

### StyleModel 

- As comparision algorithms, StyleModel implements 3 regressions (lasso, random forest, xgboost) by training one model for each style.
```
style_model = StyleModel()
style_model.run()
```

### SingleModel 

- As comparision algorithms, SingleModel implements 3 regressions (lasso, random forest, xgboost) by training one model for all styles.
```
single_model = SingleModel()
single_model.run())
```

## Check the results

- Model performance results are both 1) printed on console, and 2) saved as csv files starting by `df_out_` into the `src/data/` folder, e.g. `df_out_quality_model.csv` is the output results of quality socre model.

- Models are saved under `src/model/` folder

- Final performance plots are saved under `src/figure/`

- To see the visualized comparision figures: Create an instance from `ModelInterface` and call `get_performance_metric()`. Then look for figures under `src/figure/`. 
```
model_interface = ModelInterface()
model_interface.get_performance_metric()
```

## Contact me

- Please reach out to me directly by ywang110@gmail.com if you have any question to run the code.