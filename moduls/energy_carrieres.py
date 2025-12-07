'''
Module for processing energy carrier data such as electricity prices, diesel prices, and carbon intensity.
Date: 2025-12-07
Author: Jara Späte
'''

import os
import pandas as pd
from state_mapping import mapping, get_state_mappings
from gdf_US import create_county_US

regions_to_states = {
    'New England': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
    'Central Atlantic': ['DE', 'DC', 'MD', 'NJ', 'NY', 'PA'],
    'Lower Atlantic': ['FL', 'GA', 'NC', 'SC', 'VA', 'WV'],
    'Midwest': ['IL', 'IN', 'IA', 'KS', 'KY', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'OK', 'SD', 'TN', 'WI'],
    'Gulf Coast': ['AL', 'AR', 'LA', 'MS', 'NM', 'TX'],
    'Rocky Mountains': ['CO', 'ID', 'MT', 'UT', 'WY'],
    'West Coast exc California': ['AK', 'AZ', 'HI', 'NV', 'OR', 'WA'],
    'California': ['CA']
}

#### FILE PATHS
# Prepare file paths for saving the results
PATH_EL_OUTPUT = '../final_outputs/carriers/electricity'
PATH_DIESEL_OUTPUT = '../final_outputs/carriers/diesel'
PATH_NODES = '../final_outputs/energy_system/nodes_filtered_p75.csv'

PATH_DIESEL_PRICES = '../data_inputs/carriers/diesel/Weekly_On-Highway_Diesel_Fuel_Prices_20240720.csv'

PATH_EL_PRICES = '../data_inputs/carriers/electricity/price_import_history_eia.csv'

PATH_CARBON_INTENSITY = '../data_inputs/carriers/electricity/statistic_id1133295_power-sector-carbon-intensity-in-the-us-2022-by-state.xlsx'


def process_electricity_prices(df_price_el, us_counties, nodes_filtered):
    # Map state names to abbreviations and calculate the price in dollars/kWh
    df_electricity = mapping(df_price_el, 'Name', 'full_to_abbr')
    df_electricity['price_import'] = df_electricity['price electricity [ct./kWh]'] / 100

    # Rename mapped column for clarity
    df_electricity.rename(columns={'Name_mapped': 'State'}, inplace=True)
    print(df_electricity.head())

    # Group by 'State' and calculate the average, 95th percentile, and 5th percentile for price_import
    state_price_stats = df_electricity.groupby('State').agg(
        avg_price_import=('price_import', 'mean'),
        percentile_95_price_import=('price_import', lambda x: x.quantile(0.95)),
        percentile_5_price_import=('price_import', lambda x: x.quantile(0.05))
    ).reset_index()

    # Merge state-level price data with county-level GEOID information
    df_electricity_county = pd.merge(us_counties[['STUSPS', 'node']], state_price_stats,
                                     left_on='STUSPS', right_on='State', how='left')
    # Filter the DataFrame to include only the nodes in the filtered list
    df_electricity_county = df_electricity_county[df_electricity_county['node'].isin(nodes_filtered)]

    # Print all the NaN values in the 'avg_price_import' column
    print(f"Number of NaN values in 'avg_price_import': {df_electricity_county['avg_price_import'].isna().sum()}")

    # Sort by GEOID and rename the column for consistency
    df_electricity_county.sort_values('node', inplace=True)
    print(df_electricity_county.head())




    # Filenames for the output CSV files
    filename_avg = f'price_import.csv'
    filename_max = f'price_import_max.csv'
    filename_min = f'price_import_min.csv'

    # Save the average price_import data
    df_avg = df_electricity_county[['node', 'avg_price_import']].rename(columns={'avg_price_import': 'price_import'})
    df_avg.to_csv(os.path.join(PATH_EL_OUTPUT, filename_avg), index=False)

    # Save the 95th percentile price_import data
    df_max = df_electricity_county[['node', 'percentile_95_price_import']].rename(columns={'percentile_95_price_import': 'price_import'})
    df_max.to_csv(os.path.join(PATH_EL_OUTPUT, filename_max), index=False)
    # Save the 5th percentile price_import data
    df_min = df_electricity_county[['node', 'percentile_5_price_import']].rename(columns={'percentile_5_price_import': 'price_import'})
    df_min.to_csv(os.path.join(PATH_EL_OUTPUT, filename_min), index=False)

def process_carbon_intensity(data, us_counties, nodes_filtered):
    """
    Processes carbon intensity data by merging it with county data and saving the result to a CSV file.

    Parameters:
    data (DataFrame): Input DataFrame containing carbon intensity information for US states.
    us_counties (DataFrame): DataFrame containing US county information including nodes and state abbreviations.

    Returns:
    DataFrame: A DataFrame with county-level carbon intensity data.
    """
    # Merge carbon intensity data with county information
    df_carbon_intensity = pd.merge(
        data[['State', 'carbon intensity us grid [kg of CO2/MWh]']],
        us_counties[['node', 'STUSPS']],
        left_on='State', right_on='STUSPS', how='right'
    )

    # Rename the carbon intensity column for clarity
    df_carbon_intensity.rename(
        columns={'carbon intensity us grid [kg of CO2/MWh]': 'carbon_intensity_carrier_import'}, inplace=True
    )

    # Round carbon intensity values to two decimal places
    df_carbon_intensity['carbon_intensity_carrier_import'] = df_carbon_intensity['carbon_intensity_carrier_import'].round(2)

    # Filter the DataFrame to include only the nodes in the filtered list
    df_carbon_intensity = df_carbon_intensity[df_carbon_intensity['node'].isin(nodes_filtered)]

    # Print the amount of nan values in the 'carbon_intensity_carrier_import' column
    print(f"Number of NaN values in 'carbon_intensity_carrier_import': {df_carbon_intensity['carbon_intensity_carrier_import'].isna().sum()}")

    # Generate a filename with the current date
    filename = f'carbon_intensity_carrier_import.csv'

    # Save the processed data to a CSV file
    save_path = os.path.join(PATH_EL_OUTPUT, filename)
    df_carbon_intensity[['node', 'carbon_intensity_carrier_import']].to_csv(save_path, index=False)

    # Log the output location
    print(f"Saving the carbon intensity data to {save_path}")

    return df_carbon_intensity

def process_diesel_price(df_price_diesel, us_counties, nodes_filtered):
    # Group by Year and Region to calculate the average Diesel Price
    yearly_region_price = df_price_diesel[['Year', 'Region', 'Diesel Price']].groupby(['Year', 'Region']).mean()
    yearly_region_price.reset_index(inplace=True)

    gallons_to_megajoules = 144.945
    megajoules_to_kwh   = 1/3.6
    gallons_to_kwh = gallons_to_megajoules * megajoules_to_kwh
    yearly_region_price['Diesel Price'] = yearly_region_price['Diesel Price'] / gallons_to_kwh

    # Calculate mean and percentiles for each Region
    region_percentiles = yearly_region_price.groupby('Region').agg(
        mean_price=('Diesel Price', lambda x: round(x.mean(), 4)),
        percentile_95=('Diesel Price', lambda x: round(x.quantile(0.95), 4)),
        percentile_5=('Diesel Price', lambda x: round(x.quantile(0.05), 4))
    ).reset_index()

    # Create a list to store the state-level data
    state_data = []

    # Loop through each region and its associated states
    for region, states in regions_to_states.items():
        # Get the calculated percentiles for the current region
        region_data = region_percentiles[region_percentiles['Region'] == region].iloc[0]

        # Loop through each state in the current region and append its data
        for state in states:
            state_data.append({
                'State': state,
                'Region': region,
                'mean_price': region_data['mean_price'],
                'percentile_95': region_data['percentile_95'],
                'percentile_5': region_data['percentile_5']
            })

    # Convert the list of state data into a DataFrame
    state_percentiles_df = pd.DataFrame(state_data)

    # Merge the state-level percentiles with the county data
    df_county_price = pd.merge(state_percentiles_df, us_counties[['node', 'STUSPS']],
                               left_on='State', right_on='STUSPS', how='right')


    # Sort the DataFrame by GEOID
    df_county_price.sort_values('node', inplace=True)

    # Filter the DataFrame to include only the nodes in the filtered list
    df_county_price = df_county_price[df_county_price['node'].isin(nodes_filtered)]

    # Print the number of nan values in the 'mean_price' column
    print(f"Number of NaN values in 'mean_price': {df_county_price['mean_price'].isna().sum()}")

    # Print the first few rows for inspection
    print(df_county_price.head())

    # Define the save path for the output files
    

    # Get today's date and format it as needed


    # Define filenames for the output files
    filename_avg = f'price_import.csv'
    filename_max = f'price_import_max.csv'
    filename_min = f'price_import_min.csv'
    df_import_avg = df_county_price[['node', 'mean_price']]
    df_import_avg.rename(columns={'mean_price': 'price_import'}, inplace=True)
    df_import_max = df_county_price[['node', 'percentile_95']]
    df_import_max.rename(columns={'percentile_95': 'price_import'}, inplace=True)
    df_import_min = df_county_price[['node', 'percentile_5']]
    df_import_min.rename(columns={'percentile_5': 'price_import'}, inplace=True)

    # Save the mean, 95th, and 5th percentiles to separate CSV files
    df_import_avg[['node', 'price_import']].to_csv(os.path.join(PATH_DIESEL_OUTPUT, filename_avg), index=False)
    df_import_max[['node', 'price_import']].to_csv(os.path.join(PATH_DIESEL_OUTPUT, filename_max), index=False)
    df_import_min[['node', 'price_import']].to_csv(os.path.join(PATH_DIESEL_OUTPUT, filename_min), index=False)
    print(f"Data saved to {PATH_DIESEL_OUTPUT}")

def main():
    us_counties = create_county_US()
    nodes_df = pd.read_csv(PATH_NODES)
    nodes_filtered = nodes_df['node'].tolist()

    #### PROCESS DIESEL PRICES
    df_price_diesel = pd.read_csv(PATH_DIESEL_PRICES)
    process_diesel_price(df_price_diesel, us_counties, nodes_filtered)

    #### PROCESS ELECTRICITY PRICES
    df_price_el = pd.read_csv(PATH_EL_PRICES)
    process_electricity_prices(df_price_el, us_counties, nodes_filtered)

    #### PROCESS CARBON INTENSITY
    data = pd.read_excel(PATH_CARBON_INTENSITY, sheet_name='data')
    df_carbon_intensity = process_carbon_intensity(data, us_counties, nodes_filtered)