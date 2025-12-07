import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
from gdf_US import create_county_US


#### FILE PATHS INPUT FILES ####
FILE_PATH_NC = '../data_inputs/carriers/water/cons_irr_2001_2010.nc'
FILE_PATH_EDGES = '../final_outputs/energy_system/set_edges.csv'
FILE_PATH_ANNUAL_WATER_DATA = '../data_inputs/carriers/water/41467_2024_44920_MOESM4_ESM.xlsx'
FILE_PATH_CONVERSION_FACTOR_IRRIGATION_SYS = '../final_outputs/technologies/conversion/irrigation_sys/conversion_factor_240918.csv'


### FILE PATHS OUTPUT FILES ####
INTERMEDIATE_PATH_WATER = '../intermediate_files/carriers/water'
FILE_PATH_DEMAND_WATER_MONTH_ROSA = f'{INTERMEDIATE_PATH_WATER}/demand_water_month_rosa.csv'
FILE_PATH_DEMAND_WATER_MONTH_ROSA_FILLED = f'{INTERMEDIATE_PATH_WATER}/demand_water_month_rosa_filled_missing_values.csv'
FILE_PATH_WATER_GW_SW_DRISCOLL = f'{INTERMEDIATE_PATH_WATER}/water_gw_sw_driscoll.csv'
FILE_PATH_DEMAND_CONSUMPTION_WATER_MONTH_DRISCOLL = f'{INTERMEDIATE_PATH_WATER}/demand_consumption_water_month_driscoll.csv'
FILE_PATH_FILTERED_NODES_P75 = '../final_outputs/energy_system/nodes_filtered_p75.csv'
FILE_PATH_DEMAND = f'../final_outputs/carriers/water/demand.csv'



LIST_MONTH = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_number_dict = {'jan_hourly': 1, 'feb_hourly': 2, 'mar_hourly': 3, 'apr_hourly': 4, 'may_hourly': 5,
                         'jun_hourly': 6, 'jul_hourly': 7, 'aug_hourly': 8, 'sep_hourly': 9, 'oct_hourly': 10,
                         'nov_hourly': 11, 'dec_hourly': 12}

# Process all months from NetCDF
def process_all_months_from_nc(file_path):
    # Open the NetCDF dataset
    ds = xr.open_dataset(file_path)

    # Extract lat, lon, and irr_cons (the data variable for all months)
    lat = ds['lat'].values
    lon = ds['lon'].values
    month_data_array = ds['irr_cons'].values  # Shape (time, lat, lon)

    # The shape is (time, lat, lon), we need to reorder to (lat, lon, time)
    month_data_array = np.transpose(month_data_array, (1, 2, 0))  # Now (lat, lon, time)

    return month_data_array, lon, lat


def reshape_data_to_lat_lon_v2000(month_data_array, lon, lat):
    # Create a meshgrid of lon and lat
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten the grids and the data array
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    df_month = pd.DataFrame({
        'lat': lat_flat,
        'lon': lon_flat
    })



    for i in range(12):
        # Flatten each month's data
        month = month_data_array[:, :, i]
        reshaped_pixel_data = month.flatten()

        # Add the flattened month data to the DataFrame
        df_month[LIST_MONTH[i]] = reshaped_pixel_data

    # Create a GeoDataFrame with points geometry
    geometry = [Point(xy) for xy in zip(df_month.lon, df_month.lat)]
    gdf_data = gpd.GeoDataFrame(df_month, geometry=geometry)

    # Define the CRS, assuming EPSG:4326 (WGS 84)
    gdf_data = gdf_data.set_crs(epsg=4326)

    return df_month, gdf_data


def allocate_to_nodes(data_gdf, gdf_allocation, node_column):
    ''' Allocate the median irrigation water data to US states.
    params: df: DataFrame containing the monthly median irrigation water data for different models in the US region for each longitutde and latitude
    return: demand_water_month_state: DataFrame containing the sum of monthly median irrigation water data for different models in each US state.
    '''
    # Ensure the US boundary GeoDataFrame is in the same CRS
    if gdf_allocation.crs != data_gdf.crs:
        gdf_allocation = gdf_allocation.to_crs(data_gdf.crs)

    month_gdf_filtered = gpd.sjoin(data_gdf, gdf_allocation, how="inner", predicate="within")


    data_filtered = month_gdf_filtered[[node_column,'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']]

    demand_water_month = data_filtered.groupby(node_column).sum().reset_index()

    # Convert data to m^3
    KM3_TO_M3 = 1e9  # conversion factor from km^3 to m^3
    demand_water_month.loc[:, 'jan':'dec'] = demand_water_month.loc[:, 'jan':'dec'] * KM3_TO_M3
    print("Unit changed to m^3")

    #add the total demand
    demand_water_month['total'] = demand_water_month.loc[:, 'jan':'dec'].sum(axis=1)
    demand_water_month.rename(columns={node_column: 'node'}, inplace=True)

    # Sort by node
    demand_water_month = demand_water_month.sort_values('node')

    save_path = os.path.join( FILE_PATH_DEMAND_WATER_MONTH_ROSA)

    print(f"\n Saving the demand_water_month data to {save_path}")
    demand_water_month.to_csv(save_path, index=False)
    return demand_water_month


def fill_missing_values(demand_water_month, us_counties, df_edge):
    '''
    Fill missing values in the demand_water_month DataFrame with the mean of neighboring counties.

    Parameters:
    us_counties (GeoDataFrame): GeoDataFrame with the county data
    demand_water_month (DataFrame): DataFrame with the water demand data allocated to the counties

    Returns:
    DataFrame: DataFrame with missing values filled with the mean of neighboring counties
    '''

    df_variation = pd.merge(us_counties['node'], demand_water_month, on='node', how='left')
    print(df_variation.head(2))

    # Add columns to track whether missing values were filled and the neighbors used for filling
    df_variation['filled_by_mean'] = False
    df_variation['neighbors_used'] = ''

    # Step 1: Find counties in df_variation with missing values
    missing_counties = df_variation[df_variation.isnull().any(axis=1)]['node'].tolist()

    # Step 2: Fill missing values with the mean of neighboring counties
    for county in missing_counties:
        # Find neighboring counties for this county from df_edge
        neighbors = df_edge[df_edge['node_from'] == county]['node_to'].tolist()

        # Filter the neighbors that have data in df_variation
        neighbors_with_data = df_variation[df_variation['node'].isin(neighbors)]

        # Exclude 'filled_by_mean' and 'neighbors_used' columns from the mean calculation
        # Also exclude 'node' because it's not part of the numeric data
        columns_to_exclude = ['node', 'filled_by_mean', 'neighbors_used']
        neighbors_with_data_cleaned = neighbors_with_data.drop(columns=columns_to_exclude)

        if not neighbors_with_data_cleaned.empty:
            # Calculate the mean of neighboring counties for each numeric column
            mean_values = neighbors_with_data_cleaned.mean()

            # Ensure that the number of mean values matches the number of columns to be updated
            columns_to_update = df_variation.columns.difference(columns_to_exclude)

            # Check if lengths of mean_values and the columns to update match
            if len(mean_values) == len(columns_to_update) :  # Adjusting for new columns
                # Update the missing county's row with the mean values
                df_variation.loc[df_variation['node'] == county, columns_to_update] = mean_values.values

                # Mark that the row was filled and store the neighbors used for filling
                df_variation.loc[df_variation['node'] == county, 'filled_by_mean'] = True
                df_variation.loc[df_variation['node'] == county, 'neighbors_used'] = ', '.join(neighbors_with_data['node'].tolist())
            else:
                print(f"Mismatch in length for {county}")
        else:
            print(f'No neighbors with data for {county}')

    return df_variation

def check_missing_data_driscoll_rosa(df_month_variation, df_driscoll_water):
    # Perform an outer merge to keep all GEOIDs from both DataFrames
    df_merged = pd.merge(df_month_variation, df_driscoll_water[['node', 'total (m3)']], on='node', how='outer', indicator=True)

    # Find GEOIDs missing only in df_driscoll_water
    missing_in_df_driscoll = df_merged[df_merged['_merge'] == 'left_only']['node']

    # Find GEOIDs missing only in df_month_variation
    missing_in_df_month_variation = df_merged[df_merged['_merge'] == 'right_only']['node']

    # Find GEOIDs missing in both DataFrames
    missing_in_both = df_merged[df_merged.isna().all(axis=1)]['node']

    # Print the results
    print(f"Number of GEOIDs missing only in df_driscoll_water: {len(missing_in_df_driscoll)}")


    print(f"\nNumber of GEOIDs missing only in df_month_variation: {len(missing_in_df_month_variation)}")


    print(f"\nNumber of GEOIDs missing in both DataFrames: {len(missing_in_both)}")
    return missing_in_df_driscoll, missing_in_df_month_variation, missing_in_both


def validate_water_calculation(df_water):
    """
    Validate the correctness of water calculations by comparing the calculated and original values for January.
    """
    jan_calculated = df_water['jan_hourly'].iloc[4] * 12 * 31
    jan_original = df_water['jan'].iloc[4]
    print(f'The calculated value of January (from hourly data): {jan_calculated:.2f} km^3')
    print(f'The original value of January (from monthly data): {jan_original:.2f} km^3')

def calculate_hourly_demand(df_water):
    """
    Calculate the hourly water demand from monthly water data, assuming electric pumps run 12 hours per day.
    """
    # Define constants
    HOURS_IRR_DAY = 12  # amount of hours of irrigation per day

    DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Calculate the total hours per month
    hours_per_month = [HOURS_IRR_DAY * days for days in DAYS_PER_MONTH]

    # Calculate hourly demand for each month
    df_hourly = df_water.loc[:, 'jan':'dec'].div(hours_per_month, axis=1)
    df_hourly.columns = [col + '_hourly' for col in df_hourly.columns]

    # Concatenate with original DataFrame
    df_hourly_water = pd.concat([df_water, df_hourly], axis=1)

    # Insert unit column
    df_hourly_water['unit'] = 'm3/h'

    # Validate calculations
    validate_water_calculation(df_hourly_water)

    today = date.today()
    today_formated = today.strftime("%y%m%d")

    path_hourly_water = f'../intermediate_files/carriers/water/demand_hourly_water_month_{today_formated}.csv'
    df_hourly_final = df_hourly_water[['node',  'jan_hourly', 'feb_hourly', 'mar_hourly', 'apr_hourly', 'may_hourly', 'jun_hourly', 'jul_hourly', 'aug_hourly', 'sep_hourly', 'oct_hourly', 'nov_hourly', 'dec_hourly', 'unit']]
    df_hourly_final.to_csv(path_hourly_water, index=False)
    print(f"Data saved to {path_hourly_water}")

    return df_hourly_final

def process_monthly_water_data(us_counties):

    # Usage:

    month_data_array, lon, lat = process_all_months_from_nc(FILE_PATH_NC)
    df_month, gdf_data = reshape_data_to_lat_lon_v2000(month_data_array, lon, lat)
    # Use us_counties to allocate the data to the GEOID

    demand_water_month = allocate_to_nodes(gdf_data, us_counties, node_column= 'node')

    ##### Fill missing values based on neighboring counties #####
    # Load the edge data (which contains neighboring county connections)
    df_edge = pd.read_csv(FILE_PATH_EDGES)

    df_variation = fill_missing_values(demand_water_month, us_counties, df_edge)

    df_variation.to_csv(FILE_PATH_DEMAND_WATER_MONTH_ROSA_FILLED, index=False)



def process_annual_water_data():
    ''' Process annual water data from Driscoll et al. 2024 and allocate to counties
    Save the data as csv file
    '''
    df_driscoll = pd.read_excel(FILE_PATH_ANNUAL_WATER_DATA, sheet_name='County emissions and water use')

    df_driscoll.rename(columns={'County FIPS':'node'}, inplace=True)

    # Assuming your dataframe is named df
    df_driscoll_water = df_driscoll.pivot(index='node', columns='Water Source', values='Water use (m3)')

    # Reset the index so 'GEOID' becomes a column again
    df_driscoll_water.reset_index(inplace=True)

    # Calculate the total water use for each county
    df_driscoll_water['total (m3)'] = df_driscoll_water[['ground','surface']].sum(axis=1)

    df_driscoll_water['gw_percentages'] = df_driscoll_water['ground'] / df_driscoll_water['total (m3)']
    df_driscoll_water['sw_percentages'] = df_driscoll_water['surface'] / df_driscoll_water['total (m3)']

    # Add the name of the unit behind the column name
    df_driscoll_water.columns = [col + ' (m3)' if col in ['ground', 'surface'] else col for col in df_driscoll_water.columns]

    #Store GEOID as string in new column 'node'
    df_driscoll_water['node'] = df_driscoll_water['node'].astype(str)

    # If only four digits, add a zero to the beginning
    df_driscoll_water['node'] = df_driscoll_water['node'].apply(lambda x: '0' + x if len(x) == 4 else x)

    # Add fisp in front of each node
    df_driscoll_water['node'] = 'fips' + df_driscoll_water['node']

    print(df_driscoll_water.head(2))

    # Sort by node
    df_driscoll_water = df_driscoll_water.sort_values(by='node')
    df_driscoll_water.to_csv(FILE_PATH_WATER_GW_SW_DRISCOLL, index=False)

def calculate_monthly_variation():
    
    # Create a new DataFrame with the 'node' column
    df_month_variation = pd.DataFrame()
    df_rosa_filled = pd.read_csv(FILE_PATH_DEMAND_WATER_MONTH_ROSA_FILLED)
    df_month_variation['node'] = df_rosa_filled['node']

    # Perform the division and assign it to the new dataframe
    df_month_variation[LIST_MONTH] = df_rosa_filled[LIST_MONTH].div(df_rosa_filled['total'], axis=0)
    return df_month_variation


def combine_annual_and_monthly_data():
    ''' Combine the annual water data from Driscoll et al. 2024 with the monthly variation from Rosa et al. 2024
    Save the data as csv file
    '''
    df_month_variation = calculate_monthly_variation()
    df_driscoll_water = pd.read_csv(FILE_PATH_WATER_GW_SW_DRISCOLL)
    df_efficiency_irr_sys = pd.read_csv(FILE_PATH_CONVERSION_FACTOR_IRRIGATION_SYS)

    df_driscoll_consumption = pd.merge(df_driscoll_water, df_efficiency_irr_sys, on='node', how='left')


    df_driscoll_consumption['total consumption (m3)'] = df_driscoll_consumption['total (m3)'] * df_driscoll_consumption['irrigation_water']

    # Merge the two DataFrames on the GEOID column (left join to keep all rows from df_month_variation)
    df_driscoll_month_variation = pd.merge(df_month_variation, df_driscoll_consumption[['node', 'total consumption (m3)']], on='node', how='left')
    # Drop the nan values
    df_driscoll_month_variation.dropna(subset=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec','total consumption (m3)'], inplace=True)

    # Perform the multiplication
    df_driscoll_month_variation[LIST_MONTH] = df_driscoll_month_variation[LIST_MONTH].multiply(df_driscoll_month_variation['total consumption (m3)'], axis=0)

    df_driscoll_month_variation['total-check'] = df_driscoll_month_variation[LIST_MONTH].sum(axis=1)
    df_driscoll_month_variation['diff'] = (df_driscoll_month_variation['total-check'] - df_driscoll_month_variation['total consumption (m3)'])/df_driscoll_month_variation['total consumption (m3)']

    # Count how many tmes the difference is larger than 0.1%
    print("Number of times the difference is larger than 0.1%:", df_driscoll_month_variation['diff'].abs().gt(0.001).sum())

    # Print the result
    print(df_driscoll_month_variation.head(2))

    # Save the DataFrame to a CSV file
    save_path = os.path.join(FILE_PATH_DEMAND_CONSUMPTION_WATER_MONTH_DRISCOLL)

    print(f"\n Saving the demand_water_month data to {save_path}")
    df_driscoll_month_variation.to_csv(save_path, index=False)

    # Print sum of df_driscoll_month_variation
    print(df_driscoll_month_variation[LIST_MONTH].sum().sum()/10**9)

    missing_in_df_driscoll, missing_in_df_month_variation, missing_in_both = check_missing_data_driscoll_rosa(df_month_variation, df_driscoll_water)

def filter_data_by_p75():
    ''' Filter the data by the 75th percentile of total water consumption
    Save the filtered nodes as csv file
    '''
    df_driscoll = pd.read_csv(FILE_PATH_DEMAND_CONSUMPTION_WATER_MONTH_DRISCOLL)
    df_driscoll['total cons (km3)'] = df_driscoll['total consumption (m3)'] / 10**9


    # Calculate the median
    median = df_driscoll['total cons (km3)'].median()
    p75 = df_driscoll['total cons (km3)'].quantile(0.75)

    # Filter that all values are larger than the median
    df_driscoll_filtered_p75 = df_driscoll[df_driscoll['total cons (km3)'] > p75]

    # Print the 75th percentile
    print(f'Counties are filtered for a total water withdrawal higher than: {p75.round(3)}km3')

    # Print the amount of datapoints
    print(f'The total amount of datapoints are: {df_driscoll["total cons (km3)"].count()}')
    print(f'The amount of filtered dataset are: {df_driscoll_filtered_p75["total cons (km3)"].count()}')

    # Print the sum of total water use
    print(f"The total of the filtered dataset is: {df_driscoll_filtered_p75['total cons (km3)'].sum().round(1)}km3")
    print(f"The total of the unfiltered dataset is: {df_driscoll['total cons (km3)'].sum().round(1)}km3")

    # Store the nodes of the filtered dataset
    df_driscoll_filtered_p75['node'].to_csv(FILE_PATH_FILTERED_NODES_P75, index=False)

def get_monthly_demand_data(df_am_hours, df_pm_hours, hourly_demand_df, month_str, month_num):
    # Filter df_am_hours for the current month and select the 'time' column
    df_am_month = df_am_hours[df_am_hours['month'] == month_num][['time']]
    df_am_month.reset_index(drop=True, inplace=True)

    # Select the node and demand columns for the current month from hourly_demand_df
    month_hourly_demand_df = hourly_demand_df[['node', month_str]]

    # Convert month_hourly_demand_df to a NumPy array and transpose it
    month_hourly_demand_array = month_hourly_demand_df.values.T

    # Repeat the second row as many times as the length of df_am_month
    month_hourly_demand_array_repeat = np.repeat(month_hourly_demand_array[1:], len(df_am_month), axis=0)

    # Convert the numpy array back to a DataFrame and set the 'time' column
    df_am_demand = pd.DataFrame(month_hourly_demand_array_repeat, columns=month_hourly_demand_array[0])
    df_am_demand.insert(0, 'time', df_am_month['time'])

    # Filter df_pm_hours for the current month and select the 'time' column
    df_pm_month = df_pm_hours[df_pm_hours['month'] == month_num][['time']]
    df_pm_month.reset_index(drop=True, inplace=True)

    # Create a NumPy array of zeros with the same shape as month_hourly_demand_array_repeat
    month_empty_array = np.zeros((len(df_pm_month), month_hourly_demand_array_repeat.shape[1]))

    # Convert the numpy array to a DataFrame and set the 'time' column
    df_pm_demand = pd.DataFrame(month_empty_array, columns=month_hourly_demand_array[0])
    df_pm_demand.insert(0, 'time', df_pm_month['time'])

    # Concatenate df_am_demand and df_pm_demand to create the demand DataFrame for the month
    demand_month_df = pd.concat([df_am_demand, df_pm_demand], ignore_index=True, sort=False)

    # Sort demand_month_df by the 'time' column and reset the index
    demand_month_df = demand_month_df.sort_values('time').reset_index(drop=True)
    demand_month_df['month'] = month_num

    return demand_month_df

def create_demand_df(hourly_water_df, save_demand=True):
    filter_nodes_df = pd.read_csv(FILE_PATH_FILTERED_NODES_P75)
    filter_nodes = filter_nodes_df['node'].tolist()

    # Print the length of the hourly_water_df
    print(f"Length of the hourly_water_df: {len(hourly_water_df)}")

    # Filter the DataFrame for the nodes in the filter_nodes list
    hourly_water_df = hourly_water_df[hourly_water_df['node'].isin(filter_nodes)]

    #Print the length of the hourly_water_df
    print(f"Length of the hourly_water_df: {len(hourly_water_df)}")

    # Create a date range for the entire year, with hourly frequency
    date_range = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='h')

    # Create the DataFrame
    df = pd.DataFrame({'time': range(len(date_range)),
                    'hours': date_range.hour,
                    'month': date_range.month})

    # Filter the DataFrame for hours from 8 am to 8 pm
    df_8am_to_8pm = df[(df['hours'] >= 8) & (df['hours'] <= 19)][['time', 'month']]


    # Filter the DataFrame for hours from 8 pm to 8 am
    df_8pm_to_8am = df[(df['hours'] < 8) | (df['hours'] > 19)][['time', 'month']]

    demand_df = pd.DataFrame(columns=['time'])
    for month_str, month_num in month_number_dict.items():
        demand_month_df = get_monthly_demand_data(df_8am_to_8pm, df_8pm_to_8am, hourly_water_df, month_str, month_num)
        demand_df = pd.concat([demand_df, demand_month_df], ignore_index=True, sort=False)
    demand_df.sort_values('time', inplace=True)
    
    if save_demand:

        demand_df.to_csv(FILE_PATH_DEMAND, index=False)
        print(f"Data saved to {FILE_PATH_DEMAND}")
    return demand_df

def main():
    us_counties = create_county_US()
    #### Process monthly water data from NetCDF and allocate to counties ####
    process_monthly_water_data(us_counties)
    #### Process annual water data from Driscoll et al. 2024 and allocate to counties ####
    process_annual_water_data()

    #### Combine annual and monthly data ####
    combine_annual_and_monthly_data()
    
    #### Filter data by 75th percentile ####
    filter_data_by_p75()

    #### Create hourly demand data ####
    df_driscoll_month_variation = pd.read_csv(FILE_PATH_DEMAND_CONSUMPTION_WATER_MONTH_DRISCOLL)
    print(f'Total water consumption check (calculated from jan - dec data): {(df_driscoll_month_variation["total-check"].sum()/10**9).round(1)}')
    print(f'Total water consumption: {(df_driscoll_month_variation["total consumption (m3)"].sum()/10**9).round(1)}')

    df_hourly_water = calculate_hourly_demand(df_driscoll_month_variation)

    df_demand_water = create_demand_df(df_hourly_water)
    print(f'Check the total of the generated dataframe: {(df_demand_water.sum().sum()/10**9).round(1)}km3')