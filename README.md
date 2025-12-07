Data Preprosessing 
## ZEN Garden Inputs
Datasets for running all scenarios in Zen-Garden in Zip file: zen_garden_inputs.zip

## Water Pump Conversion & Availability Module

### 1. What this module does
This module preprocesses U.S. county-level irrigation and groundwater data to compute conversion factors for electric and diesel water pumps. It derives irrigation system shares, groundwater well depth and pressure, pump energy source shares, and water availability. Based on these inputs, it calculates energy conversion factors (kWh/m³) and existing water pump capacities. All results are exported in Zen-Garden–compatible CSV formats.

### 2. How to use it
1. Ensure all required input files and folder paths defined at the top of the script exist.  
2. Place `parameters_conversion.json` in the same directory as the script.  
3. Run the script directly:
```bash
python water_pumps.py
```

## 3. Inputs and Outputs

**Inputs**
- Irrigation system data (Excel): irrigation_irrigated_area_county.xlsx  
- Groundwater well data (CSV per state): USGWD_*.csv  
- Groundwater/surface water abstraction: water_gw_sw_driscoll_*.csv  
- Pump energy source shares: number_pumps_us_states.xlsx  
- Filtered energy nodes: nodes_filtered_p75.csv  
- Hourly water demand: demand_hourly_water_month_*.csv  
- Irrigation system efficiency: conversion_factor_all_*.csv  
- Physical and technical parameters: parameters_conversion.json  

**Outputs**
- Irrigation system shares per node:
../intermediate_files/carriers/water/irrigation_irrigated_area_county*.csv
- Groundwater depth and pressure per node:
../intermediate_files/technologies/water_pumps/well_depth_gw*.csv
- Pump conversion factors (electric & diesel, incl. uncertainty):
../final_outputs/technologies/conversion/{el_WP,diesel_WP}/conversion_factor*.csv
- Existing pump capacities:
../final_outputs/technologies/conversion/{el_WP,diesel_WP}/capacities_WP.csv
- Water availability per carrier:
../final_outputs/carriers/{electricity,diesel}/availability.csv

## Irrigation Water Demand Processing Module (Rosa & Driscoll)

### 1. What this module does
This module processes gridded monthly irrigation water demand from NetCDF data and allocates it to U.S. counties. It combines this spatial distribution with annual county-level water use from Driscoll et al. (2024) to derive consistent monthly and hourly water demand. Missing county data are filled using spatial neighbors. The output is a filtered, hourly water demand time series for high-consumption counties (p75). All results are exported in Nexus-e–compatible CSV formats.

### 2. How to use it
1. Ensure all input files and paths defined at the top of the script are available.  
2. Make sure `create_county_US()` from `gdf_US` is accessible.  
3. Run the script directly:
```bash
python water_demand_data.py
```

### 3. Inputs and Outputs
**Inputs**
- Monthly gridded irrigation consumption (NetCDF):
cons_irr_2001_2010.nc  
- County adjacency (edges):
set_edges.csv  
- Annual county water use (Driscoll et al. 2024, Excel):
41467_2024_44920_MOESM4_ESM.xlsx  
- Irrigation system efficiency / conversion factors:
conversion_factor_240918.csv  
- U.S. county geometries with node IDs:
from create_county_US()

**Outputs**

- Monthly county water demand from Rosa (m³):
demand_water_month_rosa.csv  
- Monthly county demand with filled missing values:
demand_water_month_rosa_filled_missing_values.csv  
- Annual groundwater/surface water shares per county (Driscoll):
water_gw_sw_driscoll.csv  
- Monthly county water consumption (Driscoll × Rosa):
demand_consumption_water_month_driscoll.csv  
- Filtered high-demand nodes (p75):
nodes_filtered_p75.csv  
- Hourly water demand per county:
demand_hourly_water_month_YYMMDD.csv  
- Final hourly water demand time series for the model:
../final_outputs/carriers/water/demand.csv  


## Energy System Nodes & Edges Generator (US Counties)

### 1. What this module does
This module creates the spatial structure of the U.S. energy system by generating nodes and edges from county geometries. Each U.S. county is converted into a model node with latitude and longitude. Neighboring counties are identified via shared boundaries to build the system edges. The outputs define the full spatial topology required by the energy system model.

### 2. How to use it
1. Ensure `create_county_US()` from `gdf_US` is available and returns a county GeoDataFrame.
2. Run the script directly:
```bash
python create_system_parameters.py
```

### 3. Inputs and Outputs
Outputs
- Energy system nodes (counties with coordinates):
    - ../final_outputs/energy_system/set_nodes.csv
    - Columns: node, lat, lon
- Energy system edges (neighbor relations between counties):
    - ../final_outputs/energy_system/set_edges.csv
    - Columns: edge, node_from, node_to


## Energy Carrier Data Processing Module (Prices & Carbon Intensity)

### 1. What this module does
This module processes U.S. electricity prices, diesel prices, and power-sector carbon intensity and maps them from state or regional level to county-level energy system nodes. It computes mean, 5th, and 95th percentile import prices for both electricity and diesel. Diesel prices are converted from $/gallon to $/kWh. All outputs are filtered to the p75 node set used in the energy system model and exported as CSV files.

### 2. How to use it
1. Ensure all input files and paths defined at the top of the script exist.
2. Make sure `create_county_US()` and the state mapping utilities are available.
3. Run the script directly:
```bash
python process_energy_carriers.py
```
### 3. Inputs and Outputs
**Inputs**
- Diesel price time series (CSV):  
Weekly_On-Highway_Diesel_Fuel_Prices_20240720.csv
- Electricity price history by state (CSV):  
price_import_history_eia.csv
- Power-sector carbon intensity by state (Excel):
statistic_id1133295_power-sector-carbon-intensity-in-the-us-2022-by-state.xlsx
- Filtered model nodes:
nodes_filtered_p75.csv
- U.S. county geometries and state codes:
from create_county_US()

**Outputs**
- Electricity carrier (../final_outputs/carriers/electricity/):
    - price_import.csv (mean)  
    - price_import_max.csv (95th percentile)  
    - price_import_min.csv (5th percentile)  
    - carbon_intensity_carrier_import.csv
- Diesel carrier (../final_outputs/carriers/diesel/):
    - price_import.csv (mean)
    - price_import_max.csv (95th percentile)
    - price_import_min.csv (5th percentile)


## County-Level Solar PV Capacity Factor Processing Module

### 1. What this module does
This module loads county-level solar PV capacity factor (CF) time series, applies timezone-based time shifts, and computes monthly–hourly mean CF profiles. Missing county CFs are filled using state-level means, with a dedicated fallback for WA and OR based on plant-level data. The output is a complete, consistent CF dataset for all filtered model nodes. Results are saved in a format compatible with the energy system model.

### 2. How to use it
1. Ensure all required CSV input files and folder paths exist as defined in the script.
2. Make sure county geometries with `GEOID`, `node`, and time zones are available.
3. Run the script directly:
```bash
python process_solar_cf.py
```

### 3. Inputs and Outputs
**Inputs**
- Filtered model nodes:  
../final_outputs/energy_system/nodes_filtered_p75.csv
- County-level distributed PV CF time series (CSV per GEOID):  
../data_inputs/technologies/conversion/PV/DPV by county/*.csv
- Plant metadata for WA and OR:  
eia_solar_configs.csv
- Plant-level CF time series:  
solar_gen_cf_2022.csv
- U.S. county geometries with GEOID and state codes:  
from create_county_US()

**Outputs**
- Monthly–hourly mean CF before filling:
../final_outputs/technologies/conversion/PV/cf_solar_PV_unfilled.csv
- Final filled monthly–hourly CF per county:
../final_outputs/technologies/conversion/PV/cf_solar_PV.csv

## U.S. State & County Geometry Loader Module

### 1. What this module does
This module loads U.S. state and county shapefiles as GeoDataFrames for spatial analysis. It standardizes county identifiers into a `node` format based on FIPS codes. A utility function creates geographic bounding boxes. The outputs are ready-to-use geospatial datasets for energy system modeling and spatial allocation tasks.

## #2. How to use it
1. Ensure the U.S. state and county shapefiles exist at the paths defined in the script.
2. Import and call the required function:
```python
from gdf_US import create_county_US, create_state_US

us_counties = create_county_US()
us_states = create_state_US()
```
### 3. Inputs and Outputs
**Inputs**
- U.S. state GeoJSON:  
../data_inputs/shape-files/states/States_shapefile.geojson
- U.S. county shapefile:  
../data_inputs/shape-files/county/cb_2023_us_county_20m/cb_2023_us_county_20m.shp
