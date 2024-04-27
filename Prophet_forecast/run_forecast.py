import pandas as pd
from prophet import Prophet
from datetime import timedelta,datetime
from utils.bigquery_to_python import bigquery_to_python, python_to_bigquery_with_retries
import numpy as np
import os

# Load your data
df = pd.read_csv('.//data//abt_preparation_whs_scope.csv')
results_df = pd.read_csv('.//data//results//results_df.csv')  # Load the results with best parameters
df['whsCode'] = df['whsCode'].astype(int)

# Ensure the 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Group the main DataFrame by 'whsCode'
grouped_df = df.groupby('whsCode')

predictions = {}

# Get the maximum date in your dataset
max_date = df['date'].max()

for whsCode, group in grouped_df:
    # Check if whsCode exists in the results_df
    if whsCode in results_df['whsCode'].values:
        best_params_row = results_df[results_df['whsCode'] == whsCode]
        if not best_params_row.empty:
            best_params = best_params_row['Best_params'].iloc[0]
            best_params = eval(best_params)  # Convert string representation of dictionary to actual dictionary

            # Prepare the data for Prophet
            prophet_df = group[['date', 'prepQun']]
            prophet_df.columns = ['ds', 'y']

            # Initialize and fit the Prophet model
            model = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'], 
                            seasonality_prior_scale=best_params['seasonality_prior_scale'])
            model.fit(prophet_df)

            # Make future predictions (5 weeks into the future, as an example)
            future = model.make_future_dataframe(periods=5, freq='W-MON')  # Use 'W' for weekly frequency
            forecast = model.predict(future)
            
            # Filter out only the forecasted (future) dates
            forecasted = forecast[forecast['ds'] > max_date]

            # Adjust the date for week calculation if needed
            # Example: if your weeks start on Sunday, subtract one day
            # forecasted['ds_adjusted'] = forecasted['ds'] - timedelta(days=1)

            # Add prepWeek and prepYear based on forecast dates
            forecasted['prepWeek'] = forecasted['ds'].dt.isocalendar().week
            forecasted['prepYear'] = forecasted['ds'].dt.year

            # Rename yhat to prepQun and add the whsCode as a column
            forecasted = forecasted.rename(columns={'yhat': 'prepQun'})
            forecasted['whsCode'] = whsCode
            forecasted['maximumDate'] = max_date  # Add the maximumDate column
            predictions[whsCode] = forecasted[['ds', 'prepQun', 'prepWeek', 'prepYear', 'whsCode', 'maximumDate']]
        else:
            print(f"No best parameters found for whsCode {whsCode}")
    else:
        print(f"whsCode {whsCode} not found in results")

# Concatenate all predictions into a single DataFrame
all_predictions = pd.DataFrame()
for whsCode, forecast_df in predictions.items():
    all_predictions = pd.concat([all_predictions, forecast_df], ignore_index=True)

#all_predictions.to_csv('.//data//predictions//predictions.csv', index=False)


file_path = './/data//predictions//predictions.csv'

if os.path.exists(file_path):
    # File exists, append without writing headers
    all_predictions.to_csv(file_path, mode='a', header=False, index=False)
else:
    # File does not exist, write with headers
    all_predictions.to_csv(file_path, mode='w', header=True, index=False)


df['Prophet_Predicted'] = np.nan
df['FA'] = np.nan
forecasted_data_full = pd.concat([df, all_predictions], ignore_index=True)
def determine_value_type(row):
    year = pd.to_datetime(row['date']).year
    if pd.isna(row['ds']):
        return 'histo ' + str(year)
    else:
        return 'predicted 2023'



forecasted_data_full['valueType'] = forecasted_data_full.apply(lambda row: determine_value_type(row), axis=1)
# Create the actualPredict column
forecasted_data_full['actualPredict'] = np.where(forecasted_data_full['Prophet_Predicted'].isna(), 
                                      forecasted_data_full['prepQun'], 
                                      forecasted_data_full['Prophet_Predicted'])

forecasted_data_full['date'] = np.where(forecasted_data_full['date'].isna(), forecasted_data_full['ds'], forecasted_data_full['date'])
forecasted_data_full=forecasted_data_full.drop(columns=['Unnamed: 0','whsName','whsType','Type','nCouples','serieType','maximumDate','ds'])

## i need a code here to say if i want to save the data on local or no because am using append

forecasted_data_full.to_csv('.//data//results//forecasted_data_full.csv', index=False)
# file_path = "G:\\Drive partag" + "\UE9" + "s\\FR-SIEGE-ECHANGE_SUPPLYCHAIN\\13. Direction Transverse\\PLANIFICATION ENTREPOT\\5 PROJETS\\Automatisation Arabi\\prophet_prev\\forecasted_data_full.csv"

try:
    # Try to save the DataFrame to the specified path
    forecasted_data_full.to_csv('G:\\Drive partagés\\FR-SIEGE-ECHANGE_SUPPLYCHAIN\\13. Direction Transverse\\PLANIFICATION ENTREPOT\\5 PROJETS\\Automatisation Arabi\\prophet_prev.csv')
except Exception as e:
    print(f"An error occurred: {e}")
    # If an error occurs, try saving it to an alternative path, for example, the current directory
    forecasted_data_full.to_csv(file_path)
    print("File was saved to the alternative path.")

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
        "whsCode":"INT64",
    }




forecast_tablename = 'stage_mta'+ '.planif_forecasted_data_full'
python_to_bigquery_with_retries(
        forecasted_data_full,
        'vg1np-apps-tdcsia-dev-ad',
        forecast_tablename,
        forecast_BQ_table_schema,
        "",
        # partition_field="forecastDate",
        nb_attempts=2,
    )



