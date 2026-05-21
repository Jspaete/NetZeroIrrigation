# Data Preprocessing

Processes irrigation, pump, energy carrier, and solar PV data into inputs compatible with the optimization model Zen-Garden [1]. The final computed datasets can be found in `zen_garden_inputs.zip`. For the optimization model itself, use the open-access Zen-Garden code directly.

Full data for each figure is available in `inputs_outputs.xlsx`.

---

## 1. Setup & Prerequisites

Python 3.11 is required. Create an environment (conda or venv) and install dependencies:

```bash
# conda
conda create -n <your-env-name> python=3.11
conda activate <your-env-name>
pip install -r requirements.txt

# or venv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

All input data must be placed in `data_inputs/` before running. File locations are configured in `run_pipeline.toml` — see [Configuration](#3-configuration).

---

## 2. Quick Start

All pipeline configuration lives in a single file: `run_pipeline.toml`. Edit that file, then run from the repo root:

```bash
conda activate <your-env-name>   # or: source .venv/bin/activate
python run_pipeline.py
```

Individual steps can also be run by name or number:

```bash
python run_pipeline.py water_pumps          # run a single step by name
python run_pipeline.py 3 4                  # run steps 3 and 4 by number
```

---

## 3. Configuration

All configuration lives in `run_pipeline.toml`.

### `[run]` — step toggles

```toml
[run]
create_system_parameters = true   # Step 1: county nodes & neighbour edges
water_demand_data         = true   # Step 2: hourly water demand per county (p75 filtered)
water_pumps               = true   # Step 3: pump conversion factors & existing capacities
energy_carrieres          = true   # Step 4: electricity/diesel prices & carbon intensity
prepare_PV_data           = true   # Step 5: solar PV capacity factors per county
```

### `[paths]` — file locations

All input, intermediate, and output paths are defined here in three sections. Paths are relative to the `moduls/` directory.

```toml
[paths.input]
county_shapefile  = "../data_inputs/shape-files/county/..."
nc_irrigation     = "../data_inputs/carriers/water/cons_irr_2001_2010.nc"
# ... (see run_pipeline.toml for the full list)

[paths.intermediate]
water_month_rosa  = "../intermediate_files/carriers/water/demand_water_month_rosa.csv"
hourly_water      = "../intermediate_files/carriers/water/demand_hourly_water_month.csv"
# ...

[paths.output]
set_nodes         = "../final_outputs/energy_system/set_nodes.csv"
water_demand      = "../final_outputs/carriers/water/demand.csv"
# ...
```

---

## 4. Pipeline

Steps run in order. Steps 3–5 all depend on Step 2 completing successfully.

```
Step 1  create_system_parameters   U.S. county geometries
                                         ↓ spatial join
                                   set_nodes.csv  (county centroids)
                                   set_edges.csv  (neighbour pairs)

Step 2  water_demand_data          NetCDF irrigation grid + Driscoll annual data
                                         ↓ allocate → fill → p75 filter
                                   nodes_filtered_p75.csv
                                   demand_hourly_water_month.csv
                                   demand.csv

Step 3  water_pumps          ┐
Step 4  energy_carrieres     ├─── all read Step 2 outputs
Step 5  prepare_PV_data      ┘
```

---

## 5. Step Details

### Step 1 — Energy System Nodes & Edges (`create_system_parameters.py`)

#### What it does
Creates the spatial structure of the U.S. energy system by generating nodes and edges from county geometries. Each U.S. county becomes a model node with latitude and longitude. Neighbouring counties are identified via shared boundaries to build the system edges.

#### Inputs
- U.S. county geometries: `create_county_US()` from `gdf_US` [4]

#### Outputs
- Energy system nodes: `final_outputs/energy_system/set_nodes.csv` — columns: `node`, `lat`, `lon`
- Energy system edges: `final_outputs/energy_system/set_edges.csv` — columns: `edge`, `node_from`, `node_to`

---

### Step 2 — Irrigation Water Demand (`water_demand_data.py`)

#### What it does
Processes gridded monthly irrigation water demand from NetCDF data and allocates it to U.S. counties. Combines this spatial distribution with annual county-level water use from Driscoll et al. (2024) [2] to derive consistent monthly and hourly water demand. Missing county data are filled using spatial neighbours. The output is a filtered, hourly water demand time series for high-consumption counties (p75). All results are exported in Zen-Garden–compatible CSV formats.

#### Inputs
- Annual county water use (Driscoll et al. 2024, Excel): `41467_2024_44920_MOESM4_ESM.xlsx` [2]
- Monthly gridded irrigation consumption (NetCDF): `cons_irr_2001_2010.nc` [3]
- County adjacency (edges): `set_edges.csv` (Step 1 output)
- Irrigation system efficiency: `conversion_factor_240919.csv`
- U.S. county geometries: `create_county_US()` [4]

#### Outputs
- Monthly county water demand from Rosa (m³): `demand_water_month_rosa.csv`
- Monthly county demand with filled missing values: `demand_water_month_rosa_filled_missing_values.csv`
- Annual groundwater/surface water shares per county (Driscoll): `water_gw_sw_driscoll.csv`
- Monthly county water consumption (Driscoll × Rosa): `demand_consumption_water_month_driscoll.csv`
- Filtered high-demand nodes (p75): `nodes_filtered_p75.csv`
- Hourly water demand per county: `demand_hourly_water_month.csv`
- Final hourly water demand time series: `final_outputs/carriers/water/demand.csv`

---

### Step 3 — Water Pump Conversion & Availability (`water_pumps.py`)

#### What it does
Preprocesses U.S. county-level irrigation and groundwater data to compute conversion factors for electric and diesel water pumps. Derives irrigation system shares, groundwater well depth and pressure, pump energy source shares, and water availability. Calculates energy conversion factors (kWh/m³) and existing water pump capacities. All results are exported in Zen-Garden–compatible CSV formats.

#### Inputs
- Irrigation system type data (Excel): `irrigation_irrigated_area_county.xlsx` [2]
- Groundwater well data (CSV per state): `USGWD-Tabular/USGWD_*.csv` [5]
- Groundwater/surface water abstraction: `water_gw_sw_driscoll.csv` (Step 2 output)
- Pump energy source shares: `number_pumps_us_states.xlsx` [6]
- Filtered model nodes: `nodes_filtered_p75.csv` (Step 2 output)
- Hourly water demand: `demand_hourly_water_month.csv` (Step 2 output)
- Irrigation system efficiency: `conversion_factor_all_240921.csv`
- Physical and technical parameters: `parameters_conversion.json` [7]

#### Outputs
- Irrigation system shares per node: `intermediate_files/carriers/water/irrigation_irrigated_area_county.csv`
- Groundwater depth and pressure per node: `intermediate_files/technologies/water_pumps/well_depth_gw.csv`
- Pump conversion factors (electric & diesel, incl. uncertainty): `final_outputs/technologies/conversion/{el_WP,diesel_WP}/conversion_factor*.csv`
- Existing pump capacities: `final_outputs/technologies/conversion/{el_WP,diesel_WP}/capacities_WP.csv`
- Water availability per carrier: `final_outputs/carriers/{electricity,diesel}/availability.csv`

---

### Step 4 — Energy Carrier Prices & Carbon Intensity (`energy_carrieres.py`)

#### What it does
Processes U.S. electricity prices, diesel prices, and power-sector carbon intensity and maps them from state or regional level to county-level energy system nodes. Computes mean, 5th, and 95th percentile import prices for both carriers. Diesel prices are converted from \$/gallon to \$/kWh. All outputs are filtered to the p75 node set.

#### Inputs
- Diesel price time series (CSV): `Weekly_On-Highway_Diesel_Fuel_Prices_20240720.csv` [8]
- Electricity price history by state (CSV): `price_import_history_eia.csv` [9]
- Power-sector carbon intensity by state (Excel): `statistic_id1133295_power-sector-carbon-intensity-in-the-us-2022-by-state.xlsx` [10]
- Filtered model nodes: `nodes_filtered_p75.csv` (Step 2 output)
- U.S. county geometries and state codes: `create_county_US()` [4]

#### Outputs
- Electricity carrier (`final_outputs/carriers/electricity/`):
  - `price_import.csv` (mean)
  - `price_import_max.csv` (95th percentile)
  - `price_import_min.csv` (5th percentile)
  - `carbon_intensity_carrier_import.csv`
- Diesel carrier (`final_outputs/carriers/diesel/`):
  - `price_import.csv` (mean)
  - `price_import_max.csv` (95th percentile)
  - `price_import_min.csv` (5th percentile)

---

### Step 5 — Solar PV Capacity Factors (`prepare_PV_data.py`)

#### What it does
Loads county-level solar PV capacity factor (CF) time series, applies timezone-based time shifts, and computes monthly–hourly mean CF profiles. Missing county CFs are filled using state-level means, with a dedicated fallback for WA and OR based on plant-level data. The output is a complete CF dataset for all filtered model nodes.

#### Inputs
- Filtered model nodes: `nodes_filtered_p75.csv` (Step 2 output)
- County-level distributed PV CF time series (CSV per GEOID): `data_inputs/technologies/conversion/PV/DPV by county/*.csv` [11]
- Plant-level CF time series: `solar_gen_cf_2022.csv` [12]
- Plant metadata for WA and OR: `eia_solar_configs.csv` [12]
- U.S. county geometries with GEOID and state codes: `create_county_US()` [4]

#### Outputs
- Monthly–hourly mean CF before gap-filling: `final_outputs/technologies/conversion/PV/cf_solar_PV_unfilled.csv`
- Final filled monthly–hourly CF per county: `final_outputs/technologies/conversion/PV/cf_solar_PV.csv`

---

## 6. Helper Module: U.S. County & State Geometry Loader (`gdf_US.py`)

Loads U.S. state and county shapefiles as GeoDataFrames. Standardizes county identifiers into a `node` format based on FIPS codes. Used by all pipeline steps.

```python
from gdf_US import create_county_US, create_state_US

us_counties = create_county_US()
us_states = create_state_US()
```

**Inputs**
- U.S. state GeoJSON: `data_inputs/shape-files/states/States_shapefile.geojson` [13]
- U.S. county shapefile: `data_inputs/shape-files/county/cb_2023_us_county_20m/cb_2023_us_county_20m.shp` [4]

---

## References

1. Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco De Marco, Lukas Kunz, Lukas Schmidt-Engelbertz, Paolo Gabrielli, Giovanni Sansavini (2025). ZEN-garden: Optimizing energy transition pathways with user-oriented data handling. https://www.sciencedirect.com/science/article/pii/S2352711025000263
2. Driscoll, A.W., Conant, R.T., Marston, L.T., Choi, E. and Mueller, N.D., 2024. Greenhouse gas emissions from US irrigation pumping and implications for climate-smart irrigation policy. Nature Communications, 15(1), p.675.
3. Huang, Z., Hejazi, M., Li, X., Tang, Q., Vernon, C., Leng, G., Liu, Y., Döll, P., Eisner, S., Gerten, D. and Hanasaki, N., 2018. Reconstruction of global gridded monthly sectoral water withdrawals for 1971–2010 and analysis of their spatiotemporal patterns. Hydrology and Earth System Sciences, 22(4), pp.2117-2133.
4. data.gov. 2023 Cartographic Boundary File. Retrieved February 27, 2026, from https://catalog.data.gov/dataset/2023-cartographic-boundary-file-shp-county-and-equivalent-for-united-states-1-20000000/resource/9316cbcc-474e-46d8-8591-0baa01b65787?inner_span=True
5. Lin, C.-Y., Miller, A., Waqar, M. & Marston, L. T. A database of groundwater wells in the United States. Sci Data 11, 335 (2024).
6. S. Perdue und H. Hamer, 2018 Irrigation and Water Management Survey, United States Department of Agriculture, 2019. Available at: https://www.nass.usda.gov/Publications/AgCensus/2017/Online_Resources/Farm_and_Ranch_Irrigation_Survey/fris.pdf. Accessed on: November 2024.
7. Qin, J., Duan, W., Zou, S., Chen, Y., Huang, W. and Rosa, L., 2024. Global energy use and carbon emissions from irrigated agriculture. Nature Communications, 15(1), p.3084.
8. United States Department of Agriculture (USDA), 2024a. Historical Diesel Fuel Prices. Available at: https://agtransport.usda.gov/Fuel/Historical-Diesel-Fuel-Prices/u2kh-s8ke. Accessed on: July 2024.
9. U.S. Energy Information Administration (EIA), State Electricity Profiles: 2013 to 2023. Available at: https://www.eia.gov/electricity/. Accessed on: November 2024.
10. U.S. Energy Information Administration (EIA), Carbon Dioxide Emissions Coefficients, 2023. Available at: https://www.eia.gov/environment/emissions/co2_vol_mass.php. Accessed on: April 2024.
11. J. Seel, A. Mills, D. Millstein, W. Gorman und S. Jeong, Solar-to-Grid Public Data File for Utility-scale (UPV) and Distributed Photovoltaics (DPV) Generation, Capacity Credit, and Value for 2012-2020, United States, 2021.
12. Bracken, Cameron, Scott Underwood, Allison Campbell, Travis B Thurber, and Nathalie Voisin. "Hourly Wind and Solar Generation Profiles for Every EIA 2020 Plant in the CONUS." Zenodo, May 5, 2023. https://doi.org/10.5281/zenodo.7901615.
13. data.gov. cb_2023_us_stat_20m. Retrieved February 27, 2026, from https://catalog.data.gov/dataset/2023-cartographic-boundary-file-shp-state-and-equivalent-entities-for-united-states-1-20000000/resource/b5ec957e-2ba3-4f3f-b0a3-c01973a50aec
