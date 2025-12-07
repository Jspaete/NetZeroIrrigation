'''create system parameters for energy system model
Date: 2025-12-07
Author: Jara Späte

- nodes are US counties with lat lon
- edges are neighboring counties
- outputs are csv files with the nodes and edges defining the system
'''

from gdf_US import create_county_US
import pandas as pd
import geopandas as gpd

def create_edges(gdf):
    '''creating file with edges'''
    # Identify neighboring regions by checking for shared boundaries
    if 'node' not in gdf.columns:
        gdf['node'] = gdf['GEOID']
    neighbors = gpd.sjoin(gdf, gdf, how="inner", predicate="touches")
    # check if node column is in the dataframe

    # Create edges between neighboring regions
    edges = []
    for idx, row in neighbors.iterrows():
        if row['node_left'] != row['node_right']:
            edge = {
                'edge': f"{row['node_left']}-{row['node_right']}",
                'node_from': row['node_left'],
                'node_to': row['node_right']
            }
            edges.append(edge)

    # Create a DataFrame from the list of edges
    edges_df = pd.DataFrame(edges)

    return edges_df


def main():
    '''main function to create system parameters'''

    ##### LOAD GIS DATA FOR US COUNTIES #####
    us_counties = create_county_US()
    # Focus on the main land area of the US exclude Alaska, Hawaii, and Puerto Rico
    us_counties = us_counties.cx[-125:-65, 24:50]
    # Extract the latitude and longitude from the centroid
    us_counties['lat'] = us_counties.centroid.y
    us_counties['lon'] = us_counties.centroid.x


    #### CREATE NODES FILE #####
    df_nodes = us_counties[['node','lat', 'lon']]
    df_nodes = df_nodes.sort_values(by='node')
    df_nodes.to_csv(f'../final_outputs/energy_system/set_nodes.csv', index=False)


    #### CREATE EDGES FILE #####
    df_edges = create_edges(us_counties)
    df_edges = df_edges.sort_values(by='node_from')
    df_edges.to_csv('../final_outputs/energy_system/set_edges.csv', index=False)