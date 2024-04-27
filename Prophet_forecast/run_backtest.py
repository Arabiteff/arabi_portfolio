import os
import pandas as pd
from prophet import Prophet
from metrics import bias, forecast_accuracy, identity, exp_forecast_accuracy
from utils.bigquery_to_python import bigquery_to_python, python_to_bigquery_with_retries
import numpy as np
from datetime import datetime
import yaml
import itertools



##reading the data and preprocessing
df = pd.read_csv('.//data//abt_preparation_whs_scope.csv')
df['whsCode'] = df['whsCode'].astype(int)

data_type_mapping = {
    #'whsName': 'category',
    #'whsType': 'category',
    'whsCode': 'str',
    'Enseigne': 'category',
    'prepQun': 'int64',
    'prepWeek': 'int64',
    'prepYear': 'int64'
    # Add more columns and data types as needed
}


abt_forecast = df.astype(data_type_mapping)
abt_forecast_whs_scope = abt_forecast[['whsCode', 'Enseigne','prepWeek','prepYear','prepQun',"date"]]
abt_forecast_whs_scope = abt_forecast_whs_scope.sort_values(by='date')


# Convert the 'date' column to datetime if it's not already
abt_forecast_whs_scope['date'] = pd.to_datetime(abt_forecast_whs_scope['date'])


### split the data part
# Find the maximum date in the DataFrame
max_date = abt_forecast_whs_scope['date'].max()

# Calculate the cutoff date as 26 weeks before the maximum date
cutoff_date_test = max_date - pd.DateOffset(weeks=5)
test_df = abt_forecast_whs_scope[abt_forecast_whs_scope['date'] >= cutoff_date_test]
cutoff_date_train = test_df['date'].min() - pd.DateOffset(52)
# Now split the DataFrame into train and test sets
train_df = abt_forecast_whs_scope[abt_forecast_whs_scope['date'] <= cutoff_date_train]
validation_df=abt_forecast_whs_scope.query("date > @cutoff_date_train and date < @cutoff_date_test")

##preparing the data for the model
grouped = train_df.groupby('whsCode')
all_predictions = []
results = []

default_param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    # Add other default parameters as needed
}
#### reading params
with open('.//config//functional_config.yaml', 'r') as file:
    custom_param_grids = yaml.safe_load(file)
best_params_list = []

## i need to add a param to keep track on the tests am doing
## if i want to add only one whsCode i need to think about that 
## add the sort of dates and the freq 
##tuning and testing the model
for whsCode, group in grouped:
        print(f"treating whsCode: {whsCode}")

        best_FA = float('-inf')
        best_params = {}
        # Get the parameter grid for the current whsCode
        param_grid = custom_param_grids.get(str(whsCode), default_param_grid)
        # Generate all combinations of parameters
        keys, values = zip(*param_grid.items())    
        # Prepare data for Prophet
        prophet_train = group[['date', 'prepQun']]
        prophet_train.columns = ['ds', 'y']

        for param_combination in itertools.product(*values):
                params = {key: value for key, value in zip(keys, param_combination) if key not in ['fourier_order_weekly', 'fourier_order_yearly']}
                prophet = Prophet(**params)
                if 'fourier_order_weekly' in keys:
                    prophet.add_seasonality(name='weekly', period=7, fourier_order=dict(zip(keys, param_combination))['fourier_order_weekly'])
                if 'fourier_order_yearly' in keys:
                    prophet.add_seasonality(name='yearly', period=365.25, fourier_order=dict(zip(keys, param_combination))['fourier_order_yearly'])
                try:
                    prophet.fit(prophet_train)
                except ValueError as e:
                    print(f"Error with (whsCode: {whsCode}): {str(e)}")
                    continue

                # Prepare test data
                val_group = validation_df[(validation_df['whsCode'] == whsCode)]
                if val_group.empty:
                    print(f'here is the error  {whsCode}')
                    continue

                future = val_group[['date']]
                future.columns = ['ds']

                # Prophet prediction
                forecast = prophet.predict(future)
                prophet_pred = forecast['yhat']
                prophet_FA = forecast_accuracy(val_group['prepQun'].values, prophet_pred.values)

                # Update best parameters if needed
                if prophet_FA > best_FA:
                    best_FA = prophet_FA
                    best_params = params
        if best_params:
            best_params_list.append({'whsCode': whsCode, 'params': best_params})

        # Use best parameters to train Prophet
        prophet = Prophet(**best_params) if best_params else Prophet()
        try:
            prophet.fit(prophet_train)
        except ValueError as e :
            print(f"Error with (whsCode: {whsCode}): {str(e)}")
            continue

        # Prepare test data
        test_group = test_df[(test_df['whsCode'] == whsCode)]
        if test_group.empty:
            continue
        test_group = test_group.reset_index(drop=True)

        future = test_group[['date']]
        future.columns = ['ds']

        # Prophet prediction
        forecast = prophet.predict(future)
        prophet_pred = forecast['yhat']
        prophet_FA = forecast_accuracy(test_group['prepQun'].values, prophet_pred.values)
        test_group['Prophet_Predicted'] = prophet_pred
        all_predictions.append(test_group)
        results.append({
            'whsCode': whsCode,
            #'Enseigne': Enseigne,
            'Prophet_FA': prophet_FA,
            'Best_params': best_params
        })
               
results_df = pd.DataFrame(results)
results_df.to_csv('.//data//results//results_df.csv', index=False)
full_test_df_with_predictions = pd.concat(all_predictions)
full_test_df_with_predictions['FA'] = 1 - abs((full_test_df_with_predictions['Prophet_Predicted'] - full_test_df_with_predictions['prepQun']) / full_test_df_with_predictions['prepQun'])

train_df['Prophet_Predicted'] = np.nan
train_df['FA'] = np.nan




## i should add here a code that says if it's true we push the data to big query or no 
# # Concatenate df1 and df2
visual_backtest_data = pd.concat([train_df, full_test_df_with_predictions], ignore_index=True)

def determine_value_type(row):
    year = pd.to_datetime(row['date']).year
    if pd.notna(row['Prophet_Predicted']):
        return 'predicted ' + str(year)
    else:
        return 'histo ' + str(year)


visual_backtest_data['valueType'] = visual_backtest_data.apply(lambda row: determine_value_type(row), axis=1)
# Create the actualPredict column
visual_backtest_data['actualPredict'] = np.where(visual_backtest_data['Prophet_Predicted'].isna(), 
                                      visual_backtest_data['prepQun'], 
                                      visual_backtest_data['Prophet_Predicted'])

visual_backtest_data.to_csv('.//data//results//visual_backtest_data.csv', index=False)


forecast_BQ_table_schema = {
        "actualPredict": "FLOAT",
        "date": "DATE",
        "Enseigne": "STRING",
        "FA": "FLOAT",
        "prepQun": "FLOAT",
        "prepWeek": "INT64",
        "prepYear": "INT64",
        "Prophet_Predicted": "FLOAT",
        "valueType": "STRING",
        "whsCode":"STRING",
    }


forecast_tablename = 'gcp_dataSet'+ '.visual_backtest_data'
python_to_bigquery_with_retries(
        visual_backtest_data,
        'gcp_Prject',
        forecast_tablename,
        forecast_BQ_table_schema,
        "",
        # partition_field="forecastDate",
        nb_attempts=2,
    )
