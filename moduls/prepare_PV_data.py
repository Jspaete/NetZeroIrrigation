import pandas as pd
import os
from gdf_US import create_county_US

def calculate_monthly_hourly_mean(df_cf):
    """
    Calculate the mean capacity factor (CF) for each hour of each month.
    Parameters:
        df_cf (pd.DataFrame): DataFrame containing capacity factor data with 'Date' and 'Hour' columns.
    Returns:
        pd.DataFrame: DataFrame with the mean CF for each hour of each month, merged with the original time data.
    """
    # Convert 'Date' to datetime format and extract the month
    df_cf['Date'] = pd.to_datetime(df_cf['Date'])
    df_cf.insert(2, 'Month', df_cf['Date'].dt.month)

    # Drop the 'Date' column as it's no longer needed
    df_cf.drop(columns=['Date'], inplace=True)

    # Make a copy of the DataFrame for calculating mean values
    df_cf_copy = df_cf.copy()

    # Add a 'Time' column representing hours from 0 to 8759
    df_cf.insert(0, 'Time', range(8760))

    # Calculate the mean CF for each hour of each month
    df_cf_mean = df_cf_copy.groupby(['Month', 'Hour']).mean().reset_index()

    # Prepare a DataFrame with original 'Time', 'Hour', and 'Month' columns
    df_time_mapping = df_cf[['Hour', 'Month', 'Time']].copy()

    # Merge the mean CF values with the original time mapping
    df_cf_result = pd.merge(df_time_mapping, df_cf_mean, on=['Month', 'Hour'], how='left')

    return df_cf_result

def load_shifted_cf_for_geoids(path_dpv_data, geoids_df):
    """
    Load and time-shift capacity factor (CF) data for a list of GEOIDs.

    Parameters:
        path_dpv_data (str): Path to the directory containing the CSV files for each GEOID.
        geoids_df (pd.DataFrame): DataFrame containing GEOIDs and their associated time shift hours.

    Returns:
        pd.DataFrame: DataFrame containing the date, hour, and time-shifted CF for each GEOID.
    """
    # Initialize an empty DataFrame to store the final results
    df_final_cfs = pd.DataFrame()
    missing_geoids = []

    for geoid, node in geoids_df[['GEOID', 'node']].values.tolist():

        # Adjust filename if GEOID starts with '0'
        filename = geoid[1:] if geoid.startswith('0') else geoid
        file_path = os.path.join(path_dpv_data, f'{filename}.csv')

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f'File for GEOID {geoid} does not exist')
            missing_geoids.append(geoid)
            continue

        # Load the CSV file
        df = pd.read_csv(file_path)
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])

        # Filter the data for the year 2019
        df_2019 = df[df['Unnamed: 0'].dt.year == 2019][['Unnamed: 0', 'CF']]

        # Add 'Date' and 'Hour' columns
        df_2019['Date'] = df_2019['Unnamed: 0'].dt.date
        df_2019['Hour'] = df_2019['Unnamed: 0'].dt.strftime('%H:%M')

        # Get the time shift for the current GEOID
        time_shift_hours = geoids_df.loc[geoids_df['GEOID'] == geoid, 'time_shift_hours'].values[0]
        cf_values = df_2019['CF'].values

        # Apply the time shift to CF values
        if time_shift_hours >= 0:
            shifted_cf_values = np.concatenate([np.zeros(time_shift_hours), cf_values[:-time_shift_hours]])
        else:
            shifted_cf_values = np.concatenate([cf_values[abs(time_shift_hours):], np.zeros(abs(time_shift_hours))])

        print(f'GEOID {geoid} has a time shift of {abs(time_shift_hours)} hours')

        # Add the shifted CF values for this GEOID to the DataFrame
        df_2019[node] = shifted_cf_values

        # Merge with the final DataFrame
        if df_final_cfs.empty:
            df_final_cfs = df_2019[['Date', 'Hour', node]]
        else:
            df_final_cfs = pd.concat([df_final_cfs, df_2019[[node]]], axis=1)


    return df_final_cfs, missing_geoids

def add_time_shift_to_gdf(geo_df):
    """
    Adds a time shift column in hours to the GeoDataFrame based on the centroid of each polygon's timezone.
    """
    # Initialize the timezone finder
    timezone_finder = TimezoneFinder()

    # Apply the time shift calculation directly to each geometry in the GeoDataFrame
    def calculate_time_shift(geometry):
        centroid = geometry.centroid
        timezone_name = timezone_finder.timezone_at(lng=centroid.x, lat=centroid.y)

        if timezone_name:
            timezone = pytz.timezone(timezone_name)
            utc_offset_seconds = timezone.utcoffset(datetime.now()).total_seconds()
            return int(utc_offset_seconds / 3600)
        return None

    geo_df['time_shift_hours'] = geo_df['geometry'].apply(calculate_time_shift)

    return geo_df

def load_shifted_cf_for_geoids(path_dpv_data, geoids_df):
    """
    Load and time-shift capacity factor (CF) data for a list of GEOIDs.

    Parameters:
        path_dpv_data (str): Path to the directory containing the CSV files for each GEOID.
        geoids_df (pd.DataFrame): DataFrame containing GEOIDs and their associated time shift hours.

    Returns:
        pd.DataFrame: DataFrame containing the date, hour, and time-shifted CF for each GEOID.
    """
    # Initialize an empty DataFrame to store the final results
    df_final_cfs = pd.DataFrame()
    missing_geoids = []

    for geoid, node in geoids_df[['GEOID', 'node']].values.tolist():

        # Adjust filename if GEOID starts with '0'
        filename = geoid[1:] if geoid.startswith('0') else geoid
        file_path = os.path.join(path_dpv_data, f'{filename}.csv')

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f'File for GEOID {geoid} does not exist')
            missing_geoids.append(geoid)
            continue

        # Load the CSV file
        df = pd.read_csv(file_path)
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])

        # Filter the data for the year 2019
        df_2019 = df[df['Unnamed: 0'].dt.year == 2019][['Unnamed: 0', 'CF']]

        # Add 'Date' and 'Hour' columns
        df_2019['Date'] = df_2019['Unnamed: 0'].dt.date
        df_2019['Hour'] = df_2019['Unnamed: 0'].dt.strftime('%H:%M')

        # Get the time shift for the current GEOID
        time_shift_hours = geoids_df.loc[geoids_df['GEOID'] == geoid, 'time_shift_hours'].values[0]
        cf_values = df_2019['CF'].values

        # Apply the time shift to CF values
        if time_shift_hours >= 0:
            shifted_cf_values = np.concatenate([np.zeros(time_shift_hours), cf_values[:-time_shift_hours]])
        else:
            shifted_cf_values = np.concatenate([cf_values[abs(time_shift_hours):], np.zeros(abs(time_shift_hours))])

        print(f'GEOID {geoid} has a time shift of {abs(time_shift_hours)} hours')

        # Add the shifted CF values for this GEOID to the DataFrame
        df_2019[node] = shifted_cf_values

        # Merge with the final DataFrame
        if df_final_cfs.empty:
            df_final_cfs = df_2019[['Date', 'Hour', node]]
        else:
            df_final_cfs = pd.concat([df_final_cfs, df_2019[[node]]], axis=1)


    return df_final_cfs, missing_geoids

def calculate_monthly_hourly_mean(df_cf):
    """
    Calculate the mean capacity factor (CF) for each hour of each month.
    Parameters:
        df_cf (pd.DataFrame): DataFrame containing capacity factor data with 'Date' and 'Hour' columns.
    Returns:
        pd.DataFrame: DataFrame with the mean CF for each hour of each month, merged with the original time data.
    """
    # Convert 'Date' to datetime format and extract the month
    df_cf['Date'] = pd.to_datetime(df_cf['Date'])
    df_cf.insert(2, 'Month', df_cf['Date'].dt.month)

    # Drop the 'Date' column as it's no longer needed
    df_cf.drop(columns=['Date'], inplace=True)

    # Make a copy of the DataFrame for calculating mean values
    df_cf_copy = df_cf.copy()

    # Add a 'Time' column representing hours from 0 to 8759
    df_cf.insert(0, 'Time', range(8760))

    # Calculate the mean CF for each hour of each month
    df_cf_mean = df_cf_copy.groupby(['Month', 'Hour']).mean().reset_index()

    # Prepare a DataFrame with original 'Time', 'Hour', and 'Month' columns
    df_time_mapping = df_cf[['Hour', 'Month', 'Time']].copy()

    # Merge the mean CF values with the original time mapping
    df_cf_result = pd.merge(df_time_mapping, df_cf_mean, on=['Month', 'Hour'], how='left')

    return df_cf_result



def fill_missing_geoids_with_state_mean(df_store, geoids_filtered, us_counties, shifted_cf_plants_WA_OR_mean2):
    """
    Fills missing GEOIDs in df_store with the mean capacity factor (CF) of their respective states,
    or with the state-level mean from shifted_cf_plants_WA_OR_mean2 for WA and OR.

    Parameters:
        df_store (pd.DataFrame): DataFrame containing CF data with GEOIDs as columns.
        geoids_filtered (list): List of GEOIDs to check.
        us_counties (pd.DataFrame): DataFrame containing 'GEOID' and 'STUSPS' (state codes) for each county.
        shifted_cf_plants_WA_OR_mean2 (pd.DataFrame): DataFrame containing mean CF values for WA and OR.

    Returns:
        pd.DataFrame: Updated df_store with missing GEOIDs filled with the state mean CF values.
        list: List of states where no mean CF was found.
    """
    # List of GEOIDs currently in df_store (excluding 'Time')
    geoids_present = df_store.columns.tolist()
    geoids_present.remove('Time')

    # Identify missing GEOIDs not present in df_store
    missing_geoids = list(set(geoids_filtered) - set(geoids_present))

    # Transpose df_store for easier merging with us_counties
    df_state = df_store.T.iloc[1:]  # Skip the first row ('Time')

    # Merge with us_counties to get state codes (STUSPS) for each GEOID
    df_state = pd.merge(df_state, us_counties[['node', 'STUSPS']], left_index=True, right_on='node', how='left')

    # Calculate mean CF for each state (STUSPS)
    df_state_mean = df_state.drop(columns=['node']).groupby('STUSPS').mean()

    # Prepare a DataFrame to store new columns for missing GEOIDs
    new_geoids_data = {}

    missing_state = []

    # Add missing GEOIDs to df_store, filled with the state's mean CF values
    for missing_geoid in missing_geoids:
        # Get the state code for the missing GEOID
        state_code = us_counties.loc[us_counties['node'] == missing_geoid, 'STUSPS'].values[0]

        # Special handling for WA and OR using shifted_cf_plants_WA_OR_mean2
        if state_code == 'WA':
            state_mean_cf = shifted_cf_plants_WA_OR_mean2['WA']
            new_geoids_data[missing_geoid] = state_mean_cf.values
        elif state_code == 'OR':
            state_mean_cf = shifted_cf_plants_WA_OR_mean2['OR']
            new_geoids_data[missing_geoid] = state_mean_cf.values
        elif state_code in df_state_mean.index:
            # Retrieve the mean CF for the corresponding state
            state_mean_cf = df_state_mean.loc[state_code]
            new_geoids_data[missing_geoid] = state_mean_cf.values
        else:
            if state_code not in missing_state:
                missing_state.append(state_code)
            continue

    # Convert new_geoids_data to a DataFrame
    df_new_geoids = pd.DataFrame(new_geoids_data)

    # Concatenate the new GEOID columns to df_store
    df_store = pd.concat([df_store, df_new_geoids], axis=1)

    # Print states for which no mean CF was found
    print(f"No mean CF found for state(s): {missing_state}")

    return df_store, missing_state

def process_shifted_cf_plants(info_plants_df, cf_plants_df, us_counties, df_final_cfs):
    """
    Processes capacity factor (CF) data for WA and OR by applying time shifts and calculating mean values.

    Parameters:
        info_plants_df (pd.DataFrame): DataFrame containing information about plant codes and states.
        cf_plants_df (pd.DataFrame): DataFrame containing CF values for different plant codes.
        us_counties (pd.DataFrame): DataFrame containing county-specific information, including time shift hours.
        df_final_cfs (pd.DataFrame): DataFrame containing date and hour information.
        dp (module): Module containing custom functions, such as `calculate_monthly_hourly_mean`.

    Returns:
        pd.DataFrame: A DataFrame containing the mean CF values for WA and OR after applying time shifts.
    """
    missing_states = ['WA', 'OR']

    # Find the plant codes for each missing state
    missing_state_plants = info_plants_df[info_plants_df['state'].isin(missing_states)][['state', 'plant_code']]
    print(missing_state_plants.head(2))

    # Get plant codes for WA and OR, converting them to strings
    columns_cf_plants = {
        'WA': [str(plant) for plant in missing_state_plants[missing_state_plants['state'] == 'WA']['plant_code'].tolist()],
        'OR': [str(plant) for plant in missing_state_plants[missing_state_plants['state'] == 'OR']['plant_code'].tolist()]
    }

    # Filter cf_plants_df to get columns for WA and OR
    cf_plants = {state: cf_plants_df[columns_cf_plants[state]] for state in ['WA', 'OR']}

    # Get time shift hours for WA and OR
    time_shift_hours = {
        'WA': us_counties[us_counties['STUSPS'] == 'WA']['time_shift_hours'].values[0],
        'OR': us_counties[us_counties['STUSPS'] == 'OR']['time_shift_hours'].values[0]
    }
    print(f"Time shift for WA: {time_shift_hours['WA']}, OR: {time_shift_hours['OR']}")

    # Function to apply time shift
    def apply_time_shift(cf_values, shift_hours):
        if shift_hours >= 0:
            return np.concatenate([np.zeros(shift_hours), cf_values[:-shift_hours]])
        else:
            return np.concatenate([cf_values[abs(shift_hours):], np.zeros(abs(shift_hours))])

    # Apply time shift to each column for WA and OR
    shifted_cf_plants = {state: pd.DataFrame({col: apply_time_shift(cf_plants[state][col].values, time_shift_hours[state])
                                              for col in columns_cf_plants[state]})
                         for state in ['WA', 'OR']}

    # Combine the shifted CF values for WA and OR
    shifted_cf_plants_WA_OR = pd.concat([shifted_cf_plants['WA'], shifted_cf_plants['OR']], axis=1)

    # Reset the index of both DataFrames to ensure alignment
    shifted_cf_plants_WA_OR.reset_index(drop=True, inplace=True)
    df_final_cfs.reset_index(drop=True, inplace=True)

    # Insert 'Date' and 'Hour' columns from df_final_cfs
    shifted_cf_plants_WA_OR.insert(0, 'Date', df_final_cfs['Date'])
    shifted_cf_plants_WA_OR.insert(1, 'Hour', df_final_cfs['Hour'])

    # Calculate the monthly/hourly mean
    shifted_cf_plants_WA_OR_mean = calculate_monthly_hourly_mean(shifted_cf_plants_WA_OR.copy())

    # Create a DataFrame with the mean CF values for OR and WA
    shifted_cf_plants_WA_OR_mean2 = pd.DataFrame({
        'OR': shifted_cf_plants_WA_OR_mean[columns_cf_plants['OR']].mean(axis=1),
        'WA': shifted_cf_plants_WA_OR_mean[columns_cf_plants['WA']].mean(axis=1)
    })

    return shifted_cf_plants_WA_OR_mean2

def main():

    # Load the nodes of which we conduct the analysis
    nodes_df = pd.read_csv('../final_outputs/energy_system/nodes_filtered_p75.csv')
    geoids_filtered = nodes_df['node'].tolist()

    # Load the data from the files for the missing values in the state OR and WA
    info_plants_df = pd.read_csv('../data_inputs/technologies/conversion/PV/eia_solar_configs.csv')
    cf_plants_df = pd.read_csv('../data_inputs/technologies/conversion/PV/solar_gen_cf_2022.csv')

    # Add the time shift to the GeoDataFrame
    us_counties = add_time_shift_to_gdf(us_counties)


    geoids_df = us_counties[us_counties['node'].isin(geoids_filtered)][['node', 'GEOID', 'time_shift_hours']]
    path_dpv_data = '../data_inputs/technologies/conversion/PV/DPV by county'

    # Load and time-shift CF data for the selected GEOIDs
    df_final_cfs, missing_geoids = load_shifted_cf_for_geoids(path_dpv_data, geoids_df)

    df_final_cfs2 = df_final_cfs.copy()

    # Calculate the monthly hourly mean
    df_cf_fill = calculate_monthly_hourly_mean(df_final_cfs2)


    filename = f'cf_solar_PV_unfilled.csv'
    save_path = '../final_outputs/technologies/conversion/PV'
    df_cf_fill.to_csv(os.path.join(save_path,filename), index=False)

    # Load the data from the files for the missing values in the state OR and WA
    shifted_cf_plants_WA_OR_mean2 = process_shifted_cf_plants(info_plants_df, cf_plants_df, us_counties, df_final_cfs[['Date','Hour']])
    print(shifted_cf_plants_WA_OR_mean2.head(2))

    # Exclude the columns 'Hour' and 'Month' for saving to a CSV file
    df_store = df_cf_fill.drop(columns=['Hour', 'Month'])
    # Fill missing GEOIDs with the mean of the state
    df_store, missing_state = fill_missing_geoids_with_state_mean(df_store, geoids_filtered, us_counties, shifted_cf_plants_WA_OR_mean2)


    filename = f'cf_solar_PV.csv'
    save_path = '../final_outputs/technologies/conversion/PV'

    df_store.to_csv(os.path.join(save_path, filename), index=False)