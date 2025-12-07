
import geopandas as gpd
from shapely.geometry import Polygon
import os

def make_bbox(long0, lat0, long1, lat1):
    """
    Function to create a bounding box polygon.

    Args:
        long0 (float): Starting longitude.
        lat0 (float): Starting latitude.
        long1 (float): Ending longitude.
        lat1 (float): Ending latitude.

    Returns:
        Polygon: Bounding box polygon object.
    """
    return Polygon([[long0, lat0],
                    [long1, lat0],
                    [long1, lat1],
                    [long0, lat1]])

def create_state_US():
    """
    Function to create a geospatial dataset representing countries in Europe (Faraway islands of France).

    Returns:
        GeoDataFrame: Geospatial dataset representing European countries.
    """
    # Define the bounding box to exclude far away Islands of France
    bbox = make_bbox(-125, 0, -50, 90)
    bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox])

    # Read the geospatial dataset of European countries

    # Construct the path to the state shapefile
    state_shapefile_path = os.path.join(os.path.dirname(__file__), '../data_inputs/shape-files/states/States_shapefile.geojson')

    # Read the shapefile
    us_gdf = gpd.read_file(state_shapefile_path)

    return us_gdf

def create_county_US():
    """
    Function to create a geospatial dataset representing countries in Europe (Faraway islands of France).

    Returns:
        GeoDataFrame: Geospatial dataset representing European countries.
    """

    # Read the geospatial dataset of European countries

    # Construct the path to the state shapefile
    state_shapefile_path = '../data_inputs/shape-files/county/cb_2023_us_county_20m/cb_2023_us_county_20m.shp'

    # Read the shapefile
    us_counties = gpd.read_file(state_shapefile_path)

    #Store GEOID as string in new column 'node'
    us_counties['node'] = us_counties['GEOID'].astype(str)

    # If only four digits, add a zero to the beginning
    us_counties['node'] = us_counties['node'].apply(lambda x: '0' + x if len(x) == 4 else x)

    # Add fisp in front of each node
    us_counties['node'] = 'fips' + us_counties['node']

    # Sort by node
    us_counties = us_counties.sort_values('node')

    return us_counties