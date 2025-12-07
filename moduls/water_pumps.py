import pandas as pd
import os
from state_mapping import mapping, get_state_mappings
import json
import numpy as np
from gdf_US import create_county_US

### FILE PATHS ###
output_folder_el = '../final_outputs/carriers/electricity'
output_folder_diesel = '../final_outputs/carriers/diesel'
PATH_WATER_GW_SW_DRISCOLL = '../intermediate_files/carriers/water/water_gw_sw_driscoll_240918.csv'
PATH_NODES = '../final_outputs/energy_system/nodes_filtered_p75.csv'
PATH_ENERGY_SOURCE = '../intermediate_files/technologies/water_pumps/energy_source_pumps_240918.csv'
PATH_CONVERSION_DIESEL = '../final_outputs/technologies/conversion/diesel_WP/conversion_factor_240918.csv'
PATH_CONVERSION_EL = '../final_outputs/technologies/conversion/el_WP/conversion_factor_240918.csv'
PATH_IRRIGATION_AREA = '../intermediate_files/carriers/water/irrigation_irrigated_area_county.csv'
PATH_WELL_DEPTH = '../intermediate_files/technologies/water_pumps/well_depth_gw.csv'

# Load JSON file
PARAMETER_RELATIV_PATH = 'parameters_conversion.json'
with open(PARAMETER_RELATIV_PATH, 'r') as file:
    PARAMETERS_FORMULA = json.load(file)

def process_irrigation_system_type_data(df_irr, us_counties):
    """
    Processes irrigation system type data by calculating the percentage of sprinkler, drip, and surface irrigation.
    Uses IC (irrigation crop) data when available, otherwise defaults to IR (general irrigation) data.
    The results are merged with county data and saved to a CSV file.

    Parameters:
    df_irr (DataFrame): DataFrame containing irrigation system data.
    us_counties (DataFrame): DataFrame containing county data with 'node' and 'GEOID' columns.

    Returns:
    DataFrame: A DataFrame with calculated irrigation system percentages and county data.
    """
    # Replace all '--' with nan
    df_irr = df_irr.replace('--', np.nan)

    # Calculate irrigation system percentages, using IC data when available, otherwise using IR data
    print('Replacing NaN in "IC (irrigation crop)" with general irrigation (IR) data where necessary.')
    df_irr['sprinkler_percentage'] = np.where(
        pd.isna(df_irr['IC-IrSpr']),
        df_irr['IR-IrSpr'] / df_irr['IR-IrTot'],
        df_irr['IC-IrSpr'] / df_irr['IC-IrTot']
    )
    df_irr['drip_percentage'] = np.where(
        pd.isna(df_irr['IC-IrMic']),
        df_irr['IR-IrMic'] / df_irr['IR-IrTot'],
        df_irr['IC-IrMic'] / df_irr['IC-IrTot']
    )
    df_irr['surface_percentage'] = np.where(
        pd.isna(df_irr['IC-IrSur']),
        df_irr['IR-IrSur'] / df_irr['IR-IrTot'],
        df_irr['IC-IrSur'] / df_irr['IC-IrTot']
    )

    # Round percentage values to 3 decimal places
    df_irr = df_irr.round(3)

    # Convert FIPS and GEOID columns to integers for merging
    df_irr['FIPS'] = df_irr['FIPS'].astype(int)
    us_counties['GEOID'] = us_counties['GEOID'].astype(int)

    # Merge irrigation data with county data based on FIPS and GEOID
    df_merged = pd.merge(df_irr, us_counties[['node', 'GEOID','STUSPS']], left_on='FIPS', right_on='GEOID', how='right')
    print(df_merged.head(2))

    # Fill the nan with the state average for each irrigation system
    df_merged['sprinkler_percentage'] = df_merged['sprinkler_percentage'].fillna(df_merged.groupby('STUSPS')['sprinkler_percentage'].transform('mean'))
    df_merged['drip_percentage'] = df_merged['drip_percentage'].fillna(df_merged.groupby('STUSPS')['drip_percentage'].transform('mean'))
    df_merged['surface_percentage'] = df_merged['surface_percentage'].fillna(df_merged.groupby('STUSPS')['surface_percentage'].transform('mean'))

    # Fill the remaining nan with the US average
    df_merged['sprinkler_percentage'] = df_merged['sprinkler_percentage'].fillna(df_merged['sprinkler_percentage'].mean())
    df_merged['drip_percentage'] = df_merged['drip_percentage'].fillna(df_merged['drip_percentage'].mean())
    df_merged['surface_percentage'] = df_merged['surface_percentage'].fillna(df_merged['surface_percentage'].mean())

    # Save the resulting DataFrame to a CSV file
    df_merged[['node', 'sprinkler_percentage', 'drip_percentage', 'surface_percentage']].to_csv(
        PATH_IRRIGATION_AREA, index=False
    )

    # Log file saving
    print(f'Irrigation data saved to {save_path}')

    return df_merged

def allocate_well_depth_to_state(input_folder, n_states):
    '''This function reads the data from the input folder and calculates the average well depth for each state. It also calculates the amount of NaN values and the total amount of values for each state.
    :param input_folder: The folder where the input data is stored.
    :return: A dataframe containing the average well depth, the amount of NaN values and the total amount of values for each state.
    '''

    #get the list of all the states in the US
    us_states = get_state_mappings('full_to_abbr')
    #filter only the keys of the dictionary
    us_states = list(set(us_states.keys()))

    #change a spece to a underscore in us_states:
    us_states = [state.replace(' ', '_') for state in us_states]
    columns_keep = ['State', 'FIPS', 'USGS Water Use Category', 'well_depth (m)']

    #create an empty dataframe to store the results
    df_depth_full = pd.DataFrame(columns=columns_keep)

    df_depth_short = pd.DataFrame(columns=['State', 'Average well depth', 'Number of NaN values', 'Total number of values'])

    #iterate over all the states
    for state in us_states:
        n_states += 1
        print(f'Processing state: {state}')
        #import the data for the state
        filename = 'USGWD_' + state + '.csv'
        file_path = os.path.join(input_folder, filename)
        df_state =  pd.read_csv(file_path)

        # Delet all rows with NaN values in the column 'Well Depth (Feet)'
        df_state = df_state.dropna(subset=['Well Depth (Feet)'])

        # Filter the dataframe to only include the irrigation category and the subcategories 'Irrigation-Crop (IR-C)' and 'Irrigation-Unknown (IR-U)'
        water_use_category = ['Irrigation (IR)', 'Irrigation-Crop (IR-C)', 'Irrigation-Unknown (IR-U)','Unknown']
        df_state = df_state[df_state['USGS Water Use Category'].isin(water_use_category)]

        # Filter only the active wells or if the status is unknown
        status = ['Active', 'Unknown']
        df_state = df_state[df_state['Status'].isin(status)]


        df_state['well_depth (m)'] = df_state['Well Depth (Feet)'] * 0.3048

        # Print the number of NaN values
        print(f"Number of NaN values: {df_state['well_depth (m)'].isna().sum()}")
        # Print the numer of zeros
        print(f"Number of zeros: {df_state['well_depth (m)'].eq(0).sum()}")
        # Print the average number of values excluding zeros and nan values
        print(f"Average well depth: {df_state['well_depth (m)'].replace(0, np.nan).mean()}")

        # Create a new dataframe with the average well depth, the number of NaN values and the total number of values using pd.concate
        # Create a list to collect data
        data_depth_short = []

        # Collect the necessary information for each state
        data_depth_short.append({
            'State': state,
            'Average well depth': df_state['well_depth (m)'].replace(0, np.nan).mean(),
            'Number of NaN values': df_state['well_depth (m)'].isna().sum(),
            'Total number of values': df_state['well_depth (m)'].notna().sum()  # count non-NaN values
        })

        # Convert list of dictionaries into a DataFrame and concatenate with the existing one
        df_depth_short = pd.concat([df_depth_short, pd.DataFrame(data_depth_short)], ignore_index=True)

        # Concatenate df_depth_info with new_data
        df_depth_full = pd.concat([df_depth_full, df_state[columns_keep]], ignore_index=True)

    return df_depth_full, df_depth_short


def get_data_on_pumps(file_path, filename):
    '''
    Reads the data on the percentage of pumps per carrier in each state.

    :param INPUT_PATH: Path to the dataset on the number of pumps in each state and each carrier.
    :return: Tuple containing:
                - DataFrame with percentage of pumps per carrier in each state.
                - DataFrame with state names and percentages of diesel and electric pumps only.
    '''

    # Import the dataset on the number of pumps in each state and each carrier
    data_df_pumps = pd.read_excel(os.path.join(file_path, filename), sheet_name='data')

    # Map state names to abbreviations
    data_df_pumps_mapped = mapping(data_df_pumps, 'state', 'full_to_abbr')

    data_df_pumps_mapped.rename(columns={'state_mapped': 'state'}, inplace=True)
    # Change all NaN values to zeros
    data_df_pumps_mapped = data_df_pumps_mapped.fillna(0)

    # Calculate the total number of pumps in each state
    data_df_pumps_mapped['total'] = data_df_pumps_mapped['diesel'] + data_df_pumps_mapped['electric'] + data_df_pumps_mapped['natural_gas'] + data_df_pumps_mapped['LP_gas']

    print("Lorenzo said to count natural gas to electric and LP gas to diesel")
    data_df_pumps_mapped['diesel_percentage'] = (data_df_pumps_mapped['diesel']+ data_df_pumps_mapped['LP_gas']) / data_df_pumps_mapped['total']
    data_df_pumps_mapped['electric_percentage'] = (data_df_pumps_mapped['electric'] + data_df_pumps_mapped['natural_gas'] )/ data_df_pumps_mapped['total']

    # Select relevant columns for output DataFrame
    relevant_columns = ['state', 'diesel_percentage', 'electric_percentage']
    data_df_pumps_percentage = data_df_pumps_mapped[relevant_columns]

    return data_df_pumps_mapped, data_df_pumps_percentage




def calculate_pressure(df):
    '''This function calculates the pressure in the well. It uses the formula p = rho * g * h, where p is the pressure, rho is the density of the fluid, g is the acceleration due to gravity and h is the height of the fluid above the point where the pressure is being calculated. The density of the fluid is assumed to be 1000 kg/m^3 and the acceleration due to gravity is assumed to be 9.81 m/s^2. The height of the fluid above the point where the pressure is being calculated is assumed to be the well depth.
    :param df: The dataframe containing the data.
    :return: The dataframe with the pressure in the well calculated.
    '''
    rho = 1000 #kg/m^3
    g = 9.81 #m/s^2
    h = df['well_depth (m)']
    pa_to_bar = 1e-5
    df['well_pressure [bar]'] = rho * g * h * pa_to_bar
    return df


def clean_data_gw(df_gw_depth, us_counties):

    df_gw_depth = df_gw_depth[['FIPS', 'well_depth (m)','USGS Water Use Category']].copy()

    df_gw_depth2 = pd.merge(df_gw_depth, us_counties[['GEOID','STUSPS','node']], left_on=['FIPS'], right_on=['GEOID'], how='right')


    # Group by county (GEOID) and state (STUSPS) and calculate the average well depth
    df_gw_county = df_gw_depth2.groupby(['node', 'STUSPS']).agg({'well_depth (m)': 'mean'}).reset_index()

    # Change all negative values to NaN
    df_gw_county['well_depth (m)'] = df_gw_county['well_depth (m)'].apply(lambda x: np.nan if x < 0 else x)

    # Initialize the 'NaN values' column with empty strings
    df_gw_county['NaN values'] = ''

    # Fill NaN values with the average well depth of the state and mark how it was filled
    df_gw_county['NaN values'] = np.where(
        df_gw_county['well_depth (m)'].isna(),
        'Filled with state average',
        df_gw_county['NaN values']
    )
    df_gw_county['well_depth (m)'] = df_gw_county['well_depth (m)'].fillna(
        df_gw_county.groupby('STUSPS')['well_depth (m)'].transform('mean')
    )

    # Fill the remaining NaN values with the US average and update the description
    df_gw_county['NaN values'] = np.where(
        df_gw_county['well_depth (m)'].isna(),
        'Filled with US average',
        df_gw_county['NaN values']
    )
    df_gw_county['well_depth (m)'] = df_gw_county['well_depth (m)'].fillna(
        df_gw_county['well_depth (m)'].mean()
    )

    df_gw_county = calculate_pressure(df_gw_county)

    # Display the DataFrame to see the results
    print(df_gw_county[['node', 'STUSPS', 'well_depth (m)', 'NaN values','well_pressure [bar]']])

    df_gw_county.to_csv(PATH_WELL_DEPTH, index=False)

    print(f"Data saved to {PATH_WELL_DEPTH}")

    return df_gw_county

def apply_uncertainty(df, eff_diesel, eff_el, nodes_filtered):
    """
    Apply uncertainty to conversion factor columns in the DataFrame.

    :param df: DataFrame containing conversion factor columns.
    :param rate: Uncertainty rate to apply (e.g., 1.1 for high, 0.9 for low).
    :param suffix: Suffix to append to the new columns (e.g., '_high', '_low').
    :return: DataFrame with new columns for conversion factors with applied uncertainty.
    """
    pa_to_bar = 1e-5
    g = 9.81 #m/s^2
    rho = 1000 #kg/m^3
    # efficiencey are inversely proportional to the conversion factor
    PARAMETERS_FORMULA['eff_el'] = eff_el
    PARAMETERS_FORMULA['eff_diesel'] = eff_diesel


    df['operating_pressure'] = (df['surface_percentage'] * PARAMETERS_FORMULA['H']['surface']['value'] +
                                df['drip_percentage'] * PARAMETERS_FORMULA['H']['drip']['value'] +
                                df['sprinkler_percentage'] * PARAMETERS_FORMULA['H']['sprinkler']['value'])
    ##### HIGH VALUES CAlCULATION #####
    # Calculate pressure head
    df['pressure_head'] = (df['well_pressure [bar]'] * df['gw_percentages'] +
                           df['operating_pressure'] * df['gw_percentages'] +
                           PARAMETERS_FORMULA['f_losses']['value'])/(g*pa_to_bar*rho)

    # Calculate conversion factor for electric pumps
    df['conversion_factor_el'] = df['pressure_head'] / (PARAMETERS_FORMULA['eff_el'] * PARAMETERS_FORMULA['numeric_factor'])

    # Calculate conversion factor for diesel pumps
    df['conversion_factor_diesel'] = df['pressure_head'] / (PARAMETERS_FORMULA['eff_diesel'] * PARAMETERS_FORMULA['numeric_factor'])

     # Round to 4 decimal places
    df[['conversion_factor_el','conversion_factor_diesel']] = df[['conversion_factor_el','conversion_factor_diesel']].round(4)

    # Rename the columns to include the suffix

    df['unit_conversion_factor'] = 'kWh/m3'

    # Filter the DataFrame for the nodes in the filter_nodes list
    df = df[df['node'].isin(nodes_filtered)]
    return df

def calculate_conversion_factor_uncertainty(df, nodes_filtered):
    """
    Calculate the conversion factor for electric and diesel pumps based on given parameters.

    :param df: DataFrame containing data on operating pressure, well pressure, and percentages of irrigation systems.
    :param PARAMETERS_FORMULA: Dictionary containing formula parameters for calculations.
    :return: DataFrame with added columns for conversion factors.
    Values to have high and low uncertainty rates. 
    - 'well_pressure [bar]'
    - 'gw_percentages'
    - 'eff_el
    - 'eff_diesel'"""

    electrical_range =  [0.75, 0.9, 0.8]
    diesel_range = [0.15, 0.5, 0.3]
    df_uncertainty = df.copy()
    # Calculate operating pressure per state

    df_uncertainty_high = apply_uncertainty(df_uncertainty, diesel_range[1], electrical_range[1], nodes_filtered)
    df_uncertainty_low = apply_uncertainty(df_uncertainty, diesel_range[0], electrical_range[0], nodes_filtered)
    df_uncertainty_normal = apply_uncertainty(df_uncertainty, diesel_range[2], electrical_range[2], nodes_filtered)



    # Print the amount of NaN values
    print(f"Number of NaN values in the conversion factor for electric pumps: {df['conversion_factor_el'].isna().sum()}")

    print('the unit of the conversion factor is kWh/m3')

    return df_uncertainty_high, df_uncertainty_low, df_uncertainty_normal

def calculate_conversion_factor(df, nodes_filtered):
    """
    Calculate the conversion factor for electric and diesel pumps based on given parameters.

    :param df: DataFrame containing data on operating pressure, well pressure, and percentages of irrigation systems.
    :param PARAMETERS_FORMULA: Dictionary containing formula parameters for calculations.
    :return: DataFrame with added columns for conversion factors.
    """
    pa_to_bar = 1e-5
    g = 9.81 #m/s^2
    rho = 1000 #kg/m^3

    # Calculate operating pressure per state
    df['operating_pressure'] = (df['surface_percentage'] * PARAMETERS_FORMULA['H']['surface']['value'] +
                                df['drip_percentage'] * PARAMETERS_FORMULA['H']['drip']['value'] +
                                df['sprinkler_percentage'] * PARAMETERS_FORMULA['H']['sprinkler']['value'])

    # Calculate pressure head
    df['pressure_head'] = (df['well_pressure [bar]'] * df['gw_percentages'] +
                           df['operating_pressure'] * df['gw_percentages'] +
                           PARAMETERS_FORMULA['f_losses']['value'])/(g*pa_to_bar*rho)

    # Calculate conversion factor for electric pumps
    df['conversion_factor_el'] = df['pressure_head'] / (PARAMETERS_FORMULA['eff_el'] * PARAMETERS_FORMULA['numeric_factor'])

    # Calculate conversion factor for diesel pumps
    df['conversion_factor_diesel'] = df['pressure_head'] / (PARAMETERS_FORMULA['eff_diesel'] * PARAMETERS_FORMULA['numeric_factor'])

     # Round to 4 decimal places
    df[['conversion_factor_el','conversion_factor_diesel']] = df[['conversion_factor_el','conversion_factor_diesel']].round(4)

    df['unit_conversion_factor'] = 'kWh/m3'

    # Filter the DataFrame for the nodes in the filter_nodes list
    df = df[df['node'].isin(nodes_filtered)]

    # Print the amount of NaN values
    print(f"Number of NaN values in the conversion factor for electric pumps: {df['conversion_factor_el'].isna().sum()}")

    print('the unit of the conversion factor is kWh/m3')


    return df

def store_conversion_factors(df_conversion_factor, path_output, suffix= None):
    """
    Store the conversion factors in a CSV file in the correct folder.

    :param df_conversion_factor: DataFrame containing conversion factors.
    """
    # Define the filename

    if suffix:
        filename = f'conversion_factor_{suffix}.csv'
    else:
        filename = f'conversion_factor.csv'

    # Define folder paths for diesel and electric pumps
    FOLDER_DIESEL = os.path.join(path_output, 'diesel_WP')
    FOLDER_EL = os.path.join(path_output, 'el_WP')

    # Create folder for diesel pumps if it does not exist
    os.makedirs(FOLDER_DIESEL, exist_ok=True)

    # Create folder for electric pumps if it does not exist
    os.makedirs(FOLDER_EL, exist_ok=True)

    # Filter NaN values and select relevant columns for diesel pumps
    df_conversion_factor_diesel = df_conversion_factor[['node', 'conversion_factor_diesel']]

    # Rename columns and save to CSV for diesel pumps
    df_conversion_factor_diesel.rename(columns={'conversion_factor_diesel': 'diesel'}, inplace=True)
    df_conversion_factor_diesel['blue_water'] = 1
    df_conversion_factor_diesel.to_csv(os.path.join(FOLDER_DIESEL, filename), index=False)
    print(f"Data saved to {os.path.join(FOLDER_DIESEL, filename)}")

    # Filter NaN values and select relevant columns for electric pumps
    df_conversion_factor_el = df_conversion_factor[['node', 'conversion_factor_el']]

    # Rename columns and save to CSV for electric pumps
    df_conversion_factor_el.rename(columns={'conversion_factor_el': 'electricity'}, inplace=True)
    df_conversion_factor_el['blue_water'] = 1
    df_conversion_factor_el.to_csv(os.path.join(FOLDER_EL, filename), index=False)
    print(f"Data saved to {os.path.join(FOLDER_EL, filename)}")

def availability_water_import():
    # Load the necessary data
    df_water = pd.read_csv(PATH_WATER_GW_SW_DRISCOLL)
    df_nodes = pd.read_csv(PATH_NODES)
    df_energy_source = pd.read_csv(PATH_ENERGY_SOURCE)
    df_conversion_diesel = pd.read_csv(PATH_CONVERSION_DIESEL)
    df_conversion_el = pd.read_csv(PATH_CONVERSION_EL)
    # Filter df_water for the nodes present in df_nodes
    nodes_filtered = df_nodes['node'].tolist()
    df_water_filtered = df_water[df_water['node'].isin(nodes_filtered)]

    # Rename columns for better clarity
    df_conversion_el.rename(columns={'electricity': 'cf_electricity'}, inplace=True)
    df_conversion_diesel.rename(columns={'diesel': 'cf_diesel'}, inplace=True)

    # Merge water data with conversion factors for electricity and diesel
    df_availability = pd.merge(df_water_filtered, df_conversion_el, on='node', how='left')
    df_availability = pd.merge(df_availability, df_conversion_diesel, on='node', how='left')

    # Keep only the necessary columns
    columns_to_keep = ['node', 'total (m3)', 'cf_diesel', 'cf_electricity']
    df_availability = df_availability[columns_to_keep]

    # Merge the energy source data
    df_availability = pd.merge(df_availability, df_energy_source[['node', 'diesel_percentage', 'electric_percentage']], on='node', how='left')

    # Calculate the availability of electric and diesel pumps
    df_availability['availability_el'] = df_availability['total (m3)'] * df_availability['electric_percentage'] * df_availability['cf_electricity']
    df_availability['availability_diesel'] = df_availability['total (m3)'] * df_availability['diesel_percentage'] * df_availability['cf_diesel']

    # Filter for nodes_filtered
    df_availability = df_availability[df_availability['node'].isin(nodes_filtered)]

    # Print the amount of nan values in availability_el
    print(f"The amount of nan values in availability_el is: {df_availability['availability_el'].isna().sum()}")

    # Process availability for electricity
    df_availability_el = df_availability[['node', 'availability_el']].set_index('node').T
    df_availability_el.insert(0, 'year', 2023)  # Add 'year' column with the value 2023
    df_availability_el.reset_index(drop=True, inplace=True)

    # Process availability for diesel
    df_availability_diesel = df_availability[['node', 'availability_diesel']].set_index('node').T
    df_availability_diesel.insert(0, 'year', 2023)  # Add 'year' column with the value 2023
    df_availability_diesel.reset_index(drop=True, inplace=True)

    # Define paths for saving outputs

    filename = f'availability.csv'

    # Ensure the output directories exist (optional, if needed)
    os.makedirs(output_folder_el, exist_ok=True)
    os.makedirs(output_folder_diesel, exist_ok=True)

    # Save the processed availability data to CSV files
    df_availability_el.to_csv(os.path.join(output_folder_el, filename), index=False)
    df_availability_diesel.to_csv(os.path.join(output_folder_diesel, filename), index=False)

    # Print sample outputs for verification
    print("Electricity availability (first 2 rows):")
    print(df_availability_el.head(2))

    print("Diesel availability (first 2 rows):")
    print(df_availability_diesel.head(2))

def merge_and_store_conversion_factors(df_irrigation, df_groundwater, df_gw_sw, nodes_filtered):
    """
    Merges irrigation, groundwater, and groundwater-surface water data, calculates conversion factors,
    and stores the results in CSV files for water pump technologies.

    Parameters:
    df_irrigation (DataFrame): DataFrame containing irrigation data with node information.
    df_groundwater (DataFrame): DataFrame containing groundwater data including well pressure.
    df_gw_sw (DataFrame): DataFrame containing groundwater and surface water percentages by node.

    Returns:
    None
    """

    # Merge irrigation, groundwater, and gw/sw percentage dataframes on 'node'
    df_pump_data = pd.merge(df_irrigation, df_groundwater[['node', 'well_pressure [bar]']], on='node', how='right')
    df_pump_data = pd.merge(df_pump_data, df_gw_sw[['node', 'gw_percentages', 'sw_percentages']], on='node', how='left')

    # Reposition the 'node' column to the first position
    df_pump_data = df_pump_data[['node'] + [col for col in df_pump_data.columns if col != 'node']]

    print(df_pump_data.head(1))

    # Calculate the conversion factor using the merged data
    df_conversion_factors = calculate_conversion_factor(df_pump_data, nodes_filtered)
    df_cf_high, df_cf_low, df_cf_normal = calculate_conversion_factor_uncertainty(df_pump_data, nodes_filtered)
    print(df_cf_high.head(1))
    print(df_cf_low.head(1))

    # Generate a filename with the current date and save the conversion factor data
    save_path = '../intermediate_files/technologies/water_pumps'

    filename = f'conversion_factor_pumps.csv'
    df_conversion_factors.to_csv(os.path.join(save_path, filename), index=False)

    # Store the conversion factors in the final output path
    output_path = '../final_outputs/technologies/conversion/'
    store_conversion_factors(df_conversion_factors, output_path)
    store_conversion_factors(df_cf_high, output_path, suffix='_high')
    store_conversion_factors(df_cf_low, output_path, suffix='_low')
    store_conversion_factors(df_cf_normal, output_path, suffix='_normal')


    print(df_conversion_factors.head(1))


def process_capacity_existing_wp(df_hourly_water, df_energy_source, df_efficiency_irr_sys):

    df_nodes = pd.read_csv(PATH_NODES)
    nodes_filtered = df_nodes['node'].tolist()
    # Drop the unit column
    df_hourly_water.drop(columns=['unit'], inplace=True)


    # Find the maximum of each row and assigning it to each node
    # List of month columns
    months_columns = ['jan_hourly', 'feb_hourly', 'mar_hourly', 'apr_hourly',
                    'may_hourly', 'jun_hourly', 'jul_hourly', 'aug_hourly',
                    'sep_hourly', 'oct_hourly', 'nov_hourly', 'dec_hourly']

    # Calculate the maximum for each row (node) across all months
    df_hourly_water['max_monthly_value'] = df_hourly_water[months_columns].max(axis=1)

    df_capacities_WP = df_hourly_water[['node', 'max_monthly_value']]


    df_capacities_WP = pd.merge(df_capacities_WP, df_energy_source[['node','diesel_percentage', 'electric_percentage']], on='node', how='left')

    df_capacities_WP = pd.merge(df_capacities_WP, df_efficiency_irr_sys[['node', 'irr_sys_efficiency']], on='node', how='left')

    # Calculate the capacity for each energy source
    df_capacities_WP['capacity_el'] = (
        df_capacities_WP['max_monthly_value'] * df_capacities_WP['electric_percentage']
        /df_capacities_WP['irr_sys_efficiency'])
    df_capacities_WP['capacity_diesel'] = (
        df_capacities_WP['max_monthly_value'] * df_capacities_WP['diesel_percentage']
        /df_capacities_WP['irr_sys_efficiency'])

    df_capacities_WP['unit'] = 'watervolumen/hour'
    df_capacities_WP['year_construction'] = 2022

    # Filter the DataFrame for the nodes in the filter_nodes list
    df_capacities_WP = df_capacities_WP[df_capacities_WP['node'].isin(nodes_filtered)]

    # Print all nan values in the 'capacity_el' column
    print(f"Number of NaN values in 'capacity_el': {df_capacities_WP['capacity_el'].isna().sum()}")

    df_capacities_el = df_capacities_WP[['node', 'capacity_el', 'unit', 'year_construction']]
    df_capacities_el.rename(columns={'capacity_el':'capacity_existing'}, inplace=True)

    df_capacities_diesel = df_capacities_WP[['node', 'capacity_diesel', 'unit', 'year_construction']]
    df_capacities_diesel.rename(columns={'capacity_diesel':'capacity_existing'}, inplace=True)


    # Define the save path
    save_path_diesel = '../final_outputs/technologies/conversion/diesel_WP'
    save_path_electric = '../final_outputs/technologies/conversion/el_WP'

    filename = f'capacities_WP.csv'

    # Save the data to a csv file
    df_capacities_el[['node', 'capacity_existing', 'unit', 'year_construction']].to_csv(os.path.join(save_path_electric, filename), index=False)
    df_capacities_diesel[['node', 'capacity_existing', 'unit', 'year_construction']].to_csv(os.path.join(save_path_diesel, filename), index=False)


def filter_data():
    df_nodes = pd.read_csv(PATH_NODES)
    nodes_filtered = df_nodes['node'].tolist()
    df_gw_sw_unfiltered = pd.read_csv('../intermediate_files/carriers/water/water_gw_sw_driscoll_240918.csv')
    df_irr_unfiltered = pd.read_csv('../intermediate_files/carriers/water/irrigation_irrigated_area_county_240918.csv')
    df_energy_source_unfiltered = pd.read_csv('../intermediate_files/technologies/water_pumps/energy_source_pumps_240918.csv')
    df_groundwater_depth_unfiltered = pd.read_csv('../intermediate_files/technologies/water_pumps/well_depth_gw_240918.csv')
    # Filter the dataframes for the nodes in nodes_filtered
    df_gw_sw_filered = df_gw_sw_unfiltered[df_gw_sw_unfiltered['node'].isin(nodes_filtered)]
    df_irr_filtered = df_irr_unfiltered[df_irr_unfiltered['node'].isin(nodes_filtered)]
    df_energy_source_filtered = df_energy_source_unfiltered[df_energy_source_unfiltered['node'].isin(nodes_filtered)]
    df_groundwater_depth_filtered = df_groundwater_depth_unfiltered[df_groundwater_depth_unfiltered['node'].isin(nodes_filtered)]


    # Store all the filtered dataframes
    df_gw_sw_filered.to_csv(f'../intermediate_files/carriers/water/water_gw_sw_driscoll_filtered_p75.csv', index=False)
    df_irr_filtered.to_csv(f'../intermediate_files/carriers/water/irrigation_irrigated_area_county_filtered_p75.csv', index=False)
    df_energy_source_filtered.to_csv(f'../intermediate_files/technologies/water_pumps/energy_source_pumps_filtered_p75.csv', index=False)
    df_groundwater_depth_filtered.to_csv(f'../intermediate_files/technologies/water_pumps/well_depth_gw_filtered_p75.csv', index=False)

def main():
    ##### PREPROCESSING DATA ON ENERGY SOURCES OF WATER PUMPS #####
    us_counties = create_county_US()
    filename='number_pumps_us_states.xlsx'
    file_path = '../data_inputs/technologies/conversion/water_pump'
    data_df_pumps_mapped, data_df_pumps_percentage = get_data_on_pumps(file_path, filename)
    print(data_df_pumps_mapped.head())
    df_energy_source = pd.merge(data_df_pumps_mapped, us_counties[['STUSPS','node']], left_on='state', right_on='STUSPS', how='right')


    filename = f'energy_source_pumps.csv'
    save_path = '../intermediate_files/technologies/water_pumps'
    df_energy_source.to_csv(os.path.join(save_path, filename), index=False)

    # Load the data from driscoll which was modified
    df_driscoll_water = pd.read_csv('../intermediate_files/carriers/water/water_gw_sw_driscoll_240917.csv')
    # Calculate the percentage of surface and ground water
    df_driscoll_water['sw_percentage'] = df_driscoll_water['surface (m3)'] / df_driscoll_water['total (m3)']
    df_driscoll_water['gw_percentage'] = df_driscoll_water['ground (m3)'] / df_driscoll_water['total (m3)']
    print(df_driscoll_water.head(2))

    filename='irrigation_irrigated_area_county.xlsx'
    file_path = '../data_inputs/technologies/conversion/water_pump'
    df_irr = pd.read_excel(os.path.join(file_path, filename),sheet_name='data')
    df_irr_processed = process_irrigation_system_type_data(df_irr, us_counties)

    # Load the data from the files
    path_groundwater = '../data_inputs/technologies/conversion/water_pump/USGWD-Tabular'
    n_states = 0
    df_gw_depth, df_gw_depth_short = allocate_well_depth_to_state(path_groundwater, n_states)

    #Fill the missing values, convert from feet to m and claculate it averge depth in bar for each county
    df_gw_county = clean_data_gw(df_gw_depth, us_counties)


    ##### CALCULATE CONVERSION FACTORS FOR WATER PUMPS #####
    df_irr = pd.read_csv('../intermediate_files/carriers/water/irrigation_irrigated_area_county_240918.csv')
    print(df_irr.columns)
    df_groundwater = pd.read_csv('../intermediate_files/technologies/water_pumps/well_depth_gw_240918.csv')
    print(df_groundwater.columns)
    df_gw_sw = pd.read_csv('../intermediate_files/carriers/water/water_gw_sw_driscoll_240918.csv')
    print(df_gw_sw.columns)
    output_folder = '../final_outputs/carriers/water'
    # Read the CSV file
    filter_nodes_df = pd.read_csv('../final_outputs/energy_system/nodes_filtered_p75.csv')

    # Extract the 'node' column and convert it into a list
    filter_nodes = filter_nodes_df['node'].tolist()

    merge_and_store_conversion_factors(df_irr, df_groundwater, df_gw_sw, filter_nodes)

    ##### CAlLCULATE EXISTING CAPACITY OF WATER PUMPS #####
    # Load the data of the different energy sources
    df_hourly_water = pd.read_csv('../intermediate_files/carriers/water/demand_hourly_water_month_240918.csv')

    # Load the hourly water demand for 0each month
    df_energy_source = pd.read_csv('../intermediate_files/technologies/water_pumps/energy_source_pumps_240918.csv')

    df_efficiency_irr_sys = pd.read_csv('../final_outputs/technologies/conversion/irrigation_sys/conversion_factor_all_240921.csv')
    df_efficiency_irr_sys.rename(columns={'irrigation_water':'irr_sys_efficiency'}, inplace=True)

    # Process the capacity of the existing water pumps
    process_capacity_existing_wp(df_hourly_water, df_energy_source, df_efficiency_irr_sys)

