# Data Preprosessing 
The following code allows to process various data inputs to the format that is needed to run the optimization model Zen-Garden [1]. The final computed datasets can be found in Zen-Garden zen_garden_inputs.zip
For the optimization model Zen-Garden, please use their open access code directly.


## 1. Irrigation Water Demand Processing Module (Rosa & Driscoll)

### 1.1 What this module does
This module processes gridded monthly irrigation water demand from NetCDF data and allocates it to U.S. counties. It combines this spatial distribution with annual county-level water use from Driscoll et al. (2024) [1] to derive consistent monthly and hourly water demand. Missing county data are filled using spatial neighbors. The output is a filtered, hourly water demand time series for high-consumption counties (p75). All results are exported in Nexus-e–compatible CSV formats.

### 1.2. How to use it
1. Ensure all input files and paths defined at the top of the script are available.  
2. Make sure `create_county_US()` from `gdf_US` is accessible.  
3. Run the script directly:
```bash
python water_demand_data.py
```

### 1.3. Inputs and Outputs
**Inputs**
- Annual county water use (Driscoll et al. 2024, Excel):
41467_2024_44920_MOESM4_ESM.xlsx [2]
- Monthly gridded irrigation consumption (NetCDF):
cons_irr_2001_2010.nc [3]
- County adjacency (edges):
set_edges.csv
- Irrigation system efficiency / conversion factors:
conversion_factor_240918.csv  
- U.S. county geometries with node IDs:
from create_county_US() [4]

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


## 2 Energy System Nodes & Edges Generator (US Counties)

### 2.1. What this module does
This module creates the spatial structure of the U.S. energy system by generating nodes and edges from county geometries. Each U.S. county is converted into a model node with latitude and longitude. Neighboring counties are identified via shared boundaries to build the system edges. The outputs define the full spatial topology required by the energy system model.

### 2.2. How to use it
1. Ensure `create_county_US()` from `gdf_US` is available and returns a county GeoDataFrame.
2. Run the script directly:
```bash
python create_system_parameters.py
```

### 2.3. Inputs and Outputs
**Inputs**
- U.S. county geometries with node IDs:
from create_county_US() [4]

**Outputs**
- Energy system nodes (counties with coordinates):
    - ../final_outputs/energy_system/set_nodes.csv
    - Columns: node, lat, lon
- Energy system edges (neighbor relations between counties):
    - ../final_outputs/energy_system/set_edges.csv
    - Columns: edge, node_from, node_to


## 3 Water Pump Conversion & Availability Module

### 3.1. What this module does
This module preprocesses U.S. county-level irrigation and groundwater data to compute conversion factors for electric and diesel water pumps. It derives irrigation system shares, groundwater well depth and pressure, pump energy source shares, and water availability. Based on these inputs, it calculates energy conversion factors (kWh/m³) and existing water pump capacities. All results are exported in Zen-Garden–compatible CSV formats.

### 3.2. How to use it
1. Ensure all required input files and folder paths defined at the top of the script exist.  
2. Place `parameters_conversion.json` in the same directory as the script.  
3. Run the script directly:
```bash
python water_pumps.py
```

## 3.3. Inputs and Outputs

**Inputs**
- Irrigation system data (Excel): irrigation_irrigated_area_county.xlsx [2]
- Groundwater well data (CSV per state): USGWD_*.csv [5]
- Groundwater/surface water abstraction: water_gw_sw_driscoll_*.csv [2]
- Pump energy source shares: number_pumps_us_states.xlsx [6]
- Filtered energy nodes: nodes_filtered_p75.csv  
- Hourly water demand: demand_hourly_water_month_*.csv  
- Irrigation system efficiency: conversion_factor_all_*.csv  
- Physical and technical parameters: parameters_conversion.json [7]  

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



## 4 Energy Carrier Data Processing Module (Prices & Carbon Intensity)

### 4.1. What this module does
This module processes U.S. electricity prices, diesel prices, and power-sector carbon intensity and maps them from state or regional level to county-level energy system nodes. It computes mean, 5th, and 95th percentile import prices for both electricity and diesel. Diesel prices are converted from \$/gallon to \$/kWh. All outputs are filtered to the p75 node set used in the energy system model and exported as CSV files.

### 4.2. How to use it
1. Ensure all input files and paths defined at the top of the script exist.
2. Make sure `create_county_US()` and the state mapping utilities are available.
3. Run the script directly:
```bash
python process_energy_carriers.py
```
### 4.3. Inputs and Outputs
**Inputs**
- Diesel price time series (CSV):  
Weekly_On-Highway_Diesel_Fuel_Prices_20240720.csv [8]
- Electricity price history by state (CSV):  
price_import_history_eia.csv [9]
- Power-sector carbon intensity by state (Excel):
statistic_id1133295_power-sector-carbon-intensity-in-the-us-2022-by-state.xlsx [10]
- Filtered model nodes:
nodes_filtered_p75.csv
- U.S. county geometries and state codes:
from create_county_US() [4]

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


## 5 County-Level Solar PV Capacity Factor Processing Module

### 5.1. What this module does
This module loads county-level solar PV capacity factor (CF) time series, applies timezone-based time shifts, and computes monthly–hourly mean CF profiles. Missing county CFs are filled using state-level means, with a dedicated fallback for WA and OR based on plant-level data. The output is a complete, consistent CF dataset for all filtered model nodes. Results are saved in a format compatible with the energy system model.

### 5.2. How to use it
1. Ensure all required CSV input files and folder paths exist as defined in the script.
2. Make sure county geometries with `GEOID`, `node`, and time zones are available.
3. Run the script directly:
```bash
python process_solar_cf.py
```

### 5.3. Inputs and Outputs
**Inputs**
- Filtered model nodes:  
../final_outputs/energy_system/nodes_filtered_p75.csv
- County-level distributed PV CF time series (CSV per GEOID):  
../data_inputs/technologies/conversion/PV/DPV by county/*.csv
- Plant-level CF time series:  
solar_gen_cf_2022.csv [11]
- Plant metadata for WA and OR:  
eia_solar_configs.csv [12]
- U.S. county geometries with GEOID and state codes:  
from create_county_US()

**Outputs**
- Monthly–hourly mean CF before filling:
../final_outputs/technologies/conversion/PV/cf_solar_PV_unfilled.csv
- Final filled monthly–hourly CF per county:
../final_outputs/technologies/conversion/PV/cf_solar_PV.csv

## 5. Helper function: U.S. State & County Geometry Loader Module

### 5.1. What this module does
This module loads U.S. state and county shapefiles as GeoDataFrames for spatial analysis. It standardizes county identifiers into a `node` format based on FIPS codes. A utility function creates geographic bounding boxes. The outputs are ready-to-use geospatial datasets for energy system modeling and spatial allocation tasks.

### 5.2. How to use it
1. Ensure the U.S. state and county shapefiles exist at the paths defined in the script.
2. Import and call the required function:
```python
from gdf_US import create_county_US, create_state_US

us_counties = create_county_US()
us_states = create_state_US()
```
### 5.3. Inputs and Outputs
**Inputs**
- U.S. state GeoJSON:  
../data_inputs/shape-files/states/States_shapefile.geojson
- U.S. county shapefile:  
../data_inputs/shape-files/county/cb_2023_us_county_20m/cb_2023_us_county_20m.shp

## Reference
1.	Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco De Marco, Lukas Kunz, Lukas Schmidt-Engelbertz, Paolo Gabrielli, Giovanni Sansavini (2025). ZEN-garden: Optimizing energy transition pathways with user-oriented data handling. https://www.sciencedirect.com/science/article/pii/S2352711025000263
2. Driscoll, A.W., Conant, R.T., Marston, L.T., Choi, E. and Mueller, N.D., 2024. Greenhouse gas emissions from US irrigation pumping and implications for climate-smart irrigation policy. Nature Communications, 15(1), p.675.
3. Huang, Z., Hejazi, M., Li, X., Tang, Q., Vernon, C., Leng, G., Liu, Y., Döll, P., Eisner, S., Gerten, D. and Hanasaki, N., 2018. Reconstruction of global gridded monthly sectoral water withdrawals for 1971–2010 and analysis of their spatiotemporal patterns. Hydrology and Earth System Sciences, 22(4), pp.2117-2133.
4. data.gov. 2023 Cartographic Boundary File. Retrieved February 27, 2026, from https://catalog.data.gov/dataset/2023-cartographic-boundary-file-shp-county-and-equivalent-for-united-states-1-20000000/resource/9316cbcc-474e-46d8-8591-0baa01b65787?inner_span=True
5.	Lin, C.-Y., Miller, A., Waqar, M. & Marston, L. T. A database of groundwater wells in the United States. Sci Data 11, 335 (2024).
6.	S. Perdue und H. Hamer, 2018 Irrigation and Water Management Survey, United States Department of Agriculture, 2019. Available at: https://www.nass.usda.gov/Publications/AgCensus/2017/Online_Resources/Farm_and_Ranch_Irrigation_Survey/fris.pdf. Accessed on: November 2024. 
7.	Qin, J., Duan, W., Zou, S., Chen, Y., Huang, W. and Rosa, L., 2024. Global energy use and carbon emissions from irrigated agriculture. Nature Communications, 15(1), p.3084.
8. United States Departement of Agriculture (USDA), 2024a. Historical Diesel Fuel Prices. Available at: https://agtransport.usda.gov/Fuel/Historical-Diesel-Fuel-Prices/u2kh-s8ke. Accessed on: July 2024
9.	U.S. Energy Information Administration (EIA), State Electricity Profiles: 2013 to 2023. Available at: https://www.eia.gov/electricity/. Accessed on: November 2024.
10. U.S. Energy Information Administration (EIA), Carbon Dioxide Emissions Coefficients, 2023. Available at: https://www.eia.gov/environment/emissions/co2_vol_mass.php. Accessed on: April 2024.
11.	J. Seel, A. Mills, D. Millstein, W. Gorman und S. Jeong, Solar-to-Grid Public Data File for Utility-scale (UPV) and Distributed Photovoltaics (DPV) Generation, Capacity Credit, and Value for 2012-2020, United States, 2021. 
12.	Bracken, Cameron, Scott Underwood, Allison Campbell, Travis B Thurber, and Nathalie Voisin. “Hourly Wind and Solar Generation Profiles for Every EIA 2020 Plant in the CONUS.” Zenodo, May 5, 2023. https://doi.org/10.5281/zenodo.7901615.
13. data.gov. cb_2023_us_stat_20m. Retrieved February 27, 2026, from https://catalog.data.gov/dataset/2023-cartographic-boundary-file-shp-state-and-equivalent-entities-for-united-states-1-20000000/resource/b5ec957e-2ba3-4f3f-b0a3-c01973a50aec
