#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lunar Position Calculator for Earthquake Analysis

This script calculates the precise position of the moon relative to Earth locations
over a 20-year period to enable correlation analysis with earthquake data.

Key features:
1. Calculates sub-lunar points (Earth locations directly under the moon)
2. Tracks moon distance and identifies perigee/apogee events
3. Computes gravitational influence metrics for different Earth regions
4. Integrates with the 20-year earthquake dataset

This supports the project's goal of analyzing relationships between significant 
earthquakes and lunar gravitational influence over a comprehensive timeframe.
"""

import os
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, wgs84
import logging
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "RawData")
PROCESSED_DIR = os.path.join(DATA_DIR, "ProcessedData")
LUNAR_RAW_DIR = os.path.join(RAW_DATA_DIR, "LunarData")
LUNAR_PROCESSED_DIR = os.path.join(PROCESSED_DIR, "LunarData")
EARTHQUAKE_DIR = os.path.join(PROCESSED_DIR, "Earthquakes")
EARTHQUAKE_FILE = os.path.join(EARTHQUAKE_DIR, "comcat_cleaned.csv")

# Create directories if they don't exist
for directory in [LUNAR_PROCESSED_DIR, EARTHQUAKE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def calculate_lunar_positions(start_date=None, end_date=None, output_path=None):
    """
    Calculate detailed lunar positions relative to Earth over a time period.
    
    Computes daily positions using the Skyfield astronomy library,
    including distances, phases, and identifying perigee (closest) 
    and apogee (furthest) events.
    
    Args:
        start_date (str or datetime): Start date (defaults to 20 years ago)
        end_date (str or datetime): End date (defaults to today)
        output_path (str): Path to save the calculated lunar positions
        
    Returns:
        pd.DataFrame: DataFrame with daily lunar position data
    """
    try:
        logging.info("Loading ephemeris data for lunar calculations...")
        # Load the ephemeris data (planets and their positions)
        ts = load.timescale()
        
        # If start_date and end_date are strings, convert them to datetime objects
        if start_date is not None and isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date is not None and isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set default dates if not provided (20-year period)
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 20)
        if end_date is None:
            end_date = datetime.now()
            
        # Calculate the number of days between start and end dates
        delta_days = (end_date - start_date).days + 1
        
        logging.info(f"Calculating lunar positions for {delta_days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        lunar_positions = []
        current_date = start_date
        
        # Process each day in the range
        while current_date <= end_date:
            t = ts.utc(current_date.year, current_date.month, current_date.day)
            
            # Get position of moon relative to earth
            moon_position = load('de421.bsp')['earth'].at(t).observe(load('de421.bsp')['moon'])
            ra, dec, distance = moon_position.radec()
            
            # Calculate the sub-lunar point (lat/long on Earth directly under moon)
            # Convert geocentric position to geographic point
            geocentric = moon_position.apparent()
            lat, lon = wgs84.latlon_of(geocentric)
            
            lunar_positions.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'distance_km': distance.km,
                'declination': dec.degrees,  # North-south position (-28.5° to +28.5°)
                'right_ascension': ra.hours, # East-west position (0-24h)
                'sublunar_lat': lat.degrees,  # Latitude on Earth
                'sublunar_lon': lon.degrees,  # Longitude on Earth
                # Gravity index proportional to 1/r²
                'gravity_index': (384400 / distance.km) ** 2
            })
            
            current_date += timedelta(days=1)
            
        # Convert to DataFrame
        df = pd.DataFrame(lunar_positions)
        logging.info(f"Calculated lunar positions for {len(df)} days")
        
        if output_path:
            df.to_csv(output_path, index=False)
            logging.info(f"Lunar positions saved to {output_path}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating lunar positions: {str(e)}")
        return None

def identify_perigee_apogee_events(lunar_df, window_size=15):
    """
    Identify perigee (closest) and apogee (furthest) events in the lunar dataset.
    
    Args:
        lunar_df (pd.DataFrame): DataFrame with lunar position data
        window_size (int): Window size for identifying local minima/maxima
        
    Returns:
        pd.DataFrame: DataFrame with perigee/apogee events marked
    """
    logging.info("Identifying perigee and apogee events...")
    
    # Create columns for marking events
    lunar_df['is_perigee'] = False
    lunar_df['is_apogee'] = False
    
    # Find local minima (perigee) and maxima (apogee)
    for i in range(window_size, len(lunar_df) - window_size):
        center_date = lunar_df.iloc[i]['date']
        center_distance = lunar_df.iloc[i]['distance_km']
        
        # Check for local minimum (perigee)
        is_perigee = True
        for j in range(i - window_size, i + window_size + 1):
            if j != i and lunar_df.iloc[j]['distance_km'] < center_distance:
                is_perigee = False
                break
        
        # Check for local maximum (apogee)
        is_apogee = True
        for j in range(i - window_size, i + window_size + 1):
            if j != i and lunar_df.iloc[j]['distance_km'] > center_distance:
                is_apogee = False
                break
        
        if is_perigee:
            lunar_df.at[i, 'is_perigee'] = True
            logging.info(f"Perigee (closest approach) identified on {center_date}: {center_distance:.0f} km")
        
        if is_apogee:
            lunar_df.at[i, 'is_apogee'] = True
            logging.info(f"Apogee (furthest distance) identified on {center_date}: {center_distance:.0f} km")
    
    return lunar_df

def combine_with_distance_data(lunar_positions_df, distance_data_path):
    """
    Combine new position data with existing distance dataset if available.
    
    Args:
        lunar_positions_df (pd.DataFrame): DataFrame with new lunar position data
        distance_data_path (str): Path to existing distance data CSV
        
    Returns:
        pd.DataFrame: Combined DataFrame
    """
    if os.path.exists(distance_data_path):
        logging.info(f"Combining with existing distance data from {distance_data_path}")
        try:
            distance_df = pd.read_csv(distance_data_path)
            
            # Log the columns in each dataset for debugging
            logging.info(f"Columns in new lunar positions: {lunar_positions_df.columns.tolist()}")
            logging.info(f"Columns in existing distance data: {distance_df.columns.tolist()}")
            
            # Convert date columns to string format for merging
            distance_df['date'] = pd.to_datetime(distance_df['date']).dt.strftime('%Y-%m-%d')
            lunar_positions_df['date'] = pd.to_datetime(lunar_positions_df['date']).dt.strftime('%Y-%m-%d')
            
            # Merge the datasets
            combined_df = pd.merge(distance_df, lunar_positions_df, on='date', how='outer')
            logging.info(f"Merged dataset columns: {combined_df.columns.tolist()}")
            logging.info(f"Merged datasets with {len(combined_df)} rows")
            return combined_df
        
        except Exception as e:
            logging.error(f"Error combining datasets: {str(e)}")
    
    # If no existing data or error occurred, return just the new data
    return lunar_positions_df

def merge_with_earthquake_data(lunar_df, earthquake_path):
    """
    Merge lunar position data with earthquake data for correlation analysis.
    
    This merges based on date to allow analysis of relationships between
    lunar position/distance and earthquake occurrences.
    
    Args:
        lunar_df (pd.DataFrame): DataFrame with lunar position data
        earthquake_path (str): Path to earthquake CSV file
        
    Returns:
        pd.DataFrame: DataFrame with merged data
    """
    if not os.path.exists(earthquake_path):
        logging.error(f"Earthquake data file not found: {earthquake_path}")
        return None
    
    try:
        logging.info(f"Loading earthquake data from {earthquake_path}")
        eq_df = pd.read_csv(earthquake_path)
        
        # Ensure date format compatibility
        if 'time' in eq_df.columns:
            eq_df['date'] = pd.to_datetime(eq_df['time']).dt.strftime('%Y-%m-%d')
        else:
            logging.error("Earthquake data missing 'time' column")
            return None
        
        # Convert lunar dates if needed
        if isinstance(lunar_df['date'].iloc[0], str):
            pass  # Already string format
        else:
            lunar_df['date'] = pd.to_datetime(lunar_df['date']).dt.strftime('%Y-%m-%d')
        
        # Count earthquakes per day
        eq_counts = eq_df.groupby('date').size().reset_index(name='earthquake_count')
        
        # Merge with lunar data
        merged_df = pd.merge(lunar_df, eq_counts, on='date', how='left')
        
        # Fill missing earthquake counts with 0
        merged_df['earthquake_count'] = merged_df['earthquake_count'].fillna(0)
        
        logging.info(f"Merged data contains {len(merged_df)} rows")
        return merged_df
        
    except Exception as e:
        logging.error(f"Error merging earthquake data: {str(e)}")
        return None

def create_integrated_dataset(lunar_data_path, earthquake_data_path, tectonic_distance_path=None):
    """
    Create an integrated dataset combining lunar, earthquake, and tectonic plate data.
    
    Args:
        lunar_data_path (str): Path to processed lunar data
        earthquake_data_path (str): Path to cleaned earthquake data
        tectonic_distance_path (str, optional): Path to earthquake-tectonic distance data
        
    Returns:
        pd.DataFrame: Integrated dataset with all features
    """
    logging.info("Creating integrated dataset...")
    
    # Load lunar data
    logging.info(f"Loading lunar data from {lunar_data_path}")
    try:
        lunar_df = pd.read_csv(lunar_data_path)
        lunar_df['date'] = pd.to_datetime(lunar_df['date'])
        logging.info(f"Loaded lunar data with {len(lunar_df)} rows")
    except Exception as e:
        logging.error(f"Error loading lunar data: {str(e)}")
        return None
    
    # Load earthquake data
    logging.info(f"Loading earthquake data from {earthquake_data_path}")
    try:
        eq_df = pd.read_csv(earthquake_data_path)
        eq_df['time'] = pd.to_datetime(eq_df['time'])
        logging.info(f"Loaded earthquake data with {len(eq_df)} rows")
    except Exception as e:
        logging.error(f"Error loading earthquake data: {str(e)}")
        return None
    
    # Combine earthquake and lunar data
    logging.info("Combining earthquake and lunar data with 24 hour window...")
    combined_df = eq_df.copy()
    
    # Create date from time
    combined_df['date'] = combined_df['time'].dt.date
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Merge with lunar data on date
    combined_df = pd.merge(combined_df, lunar_df, on='date', how='left')
    logging.info(f"Combined data has {len(combined_df)} rows")
    
    # Calculate distances to sub-lunar points
    logging.info("Calculating distances between earthquake epicenters and sub-lunar points...")
    combined_df = calculate_sublunary_distances(combined_df)
    
    # Save intermediate combined file
    combined_output = os.path.join(PROCESSED_DIR, "earthquake_lunar_combined.csv")
    combined_df.to_csv(combined_output, index=False)
    logging.info(f"Saved combined earthquake-lunar data with {len(combined_df)} events to {combined_output}")
    
    # Add tectonic distance data if available
    if tectonic_distance_path and os.path.exists(tectonic_distance_path):
        logging.info(f"Loading tectonic distance data from {tectonic_distance_path}")
        try:
            tectonic_df = pd.read_csv(tectonic_distance_path)
            logging.info(f"Loaded tectonic distance data with {len(tectonic_df)} rows")
            
            # Check if it has the same number of rows as our combined data
            if len(tectonic_df) == len(eq_df):
                # If same length, we can just add the distance column(s)
                # Find the distance column
                distance_cols = [col for col in tectonic_df.columns if 'distance' in col.lower()]
                
                if distance_cols:
                    for col in distance_cols:
                        combined_df[col] = tectonic_df[col].values
                    logging.info(f"Added tectonic distance column(s): {distance_cols}")
                else:
                    logging.warning("No distance column found in tectonic data")
            else:
                # If different lengths, attempt to merge on identifiers
                logging.warning(f"Tectonic data length ({len(tectonic_df)}) differs from earthquake data ({len(eq_df)})")
                logging.info("Attempting to merge on common identifiers...")
                
                # Find common columns to merge on
                eq_cols = set(eq_df.columns)
                tect_cols = set(tectonic_df.columns)
                common_cols = list(eq_cols.intersection(tect_cols))
                
                if common_cols:
                    logging.info(f"Merging on common columns: {common_cols}")
                    combined_df = pd.merge(combined_df, tectonic_df, on=common_cols, how='left')
                else:
                    logging.error("No common columns found for merging with tectonic data")
        
        except Exception as e:
            logging.error(f"Error integrating tectonic data: {str(e)}")
    
    # Save the integrated dataset
    output_path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    combined_df.to_csv(output_path, index=False)
    logging.info(f"Saved integrated dataset with {len(combined_df)} events to {output_path}")
    logging.info(f"Successfully created integrated dataset with {len(combined_df)} records")
    
    return combined_df

def calculate_sublunary_distances(df):
    """
    Calculate the great-circle distances between earthquake epicenters and sub-lunar points.
    
    This function computes the spherical distance between each earthquake's location and
    the sub-lunar point (where the moon is directly overhead) on the same day.
    
    Args:
        df (pd.DataFrame): DataFrame containing integrated earthquake and lunar data
                          Must have 'latitude', 'longitude', 'sublunar_lat', and 'sublunar_lon' columns
    
    Returns:
        pd.DataFrame: DataFrame with added 'distance_to_sublunar_km' column
    """
    if not all(col in df.columns for col in ['latitude', 'longitude', 'sublunar_lat', 'sublunar_lon']):
        logging.error("Missing required columns for calculating sub-lunar distances")
        return df
    
    # Earth radius in kilometers
    earth_radius = 6371.0
    
    # Function to calculate great-circle distance between two points
    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point in degrees
            lat2, lon2: Latitude and longitude of second point in degrees
            
        Returns:
            float: Distance in kilometers
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = earth_radius * c
        
        return distance
    
    # Calculate distances
    df['distance_to_sublunar_km'] = df.apply(
        lambda row: haversine(
            row['latitude'], row['longitude'], 
            row['sublunar_lat'], row['sublunar_lon']
        ), axis=1
    )
    
    # Create categories for sub-lunar proximity
    distance_bins = [0, 1000, 2000, 3000, 5000, 10000, 20000]
    distance_labels = [
        'Very Near (0-1000 km)',
        'Near (1000-2000 km)',
        'Moderate (2000-3000 km)',
        'Distant (3000-5000 km)',
        'Far (5000-10000 km)',
        'Very Far (>10000 km)'
    ]
    
    df['sublunar_proximity'] = pd.cut(
        df['distance_to_sublunar_km'], 
        bins=distance_bins, 
        labels=distance_labels,
        right=False,
        include_lowest=True
    )
    
    # Handle values above the highest bin
    df.loc[df['distance_to_sublunar_km'] >= distance_bins[-1], 'sublunar_proximity'] = distance_labels[-1]
    
    logging.info(f"Added sub-lunar distance calculations to {len(df)} records")
    return df

def main():
    """Main function to run the lunar position calculation and integration"""
    try:
        # Calculate lunar positions
        lunar_output = os.path.join(LUNAR_PROCESSED_DIR, "lunar_positions_processed.csv")
        lunar_df = calculate_lunar_positions(output_path=lunar_output)
        
        if lunar_df is not None:
            # Identify perigee and apogee events
            lunar_df = identify_perigee_apogee_events(lunar_df)
            
            # Save to main processed directory for compatibility with other scripts
            compat_path = os.path.join(PROCESSED_DIR, "lunar_processed.csv")
            lunar_df.to_csv(compat_path, index=False)
            logging.info(f"Saved copy of lunar data to {compat_path}")
            
            # Create integrated dataset
            logging.info("Creating integrated dataset...")
            
            # Define paths
            lunar_data_path = compat_path
            earthquake_data_path = os.path.join(EARTHQUAKE_DIR, "comcat_cleaned.csv")
            tectonic_distance_path = os.path.join(EARTHQUAKE_DIR, "earthquake_tectonic_distance_with_wkt.csv")
            
            # Create integrated dataset
            create_integrated_dataset(lunar_data_path, earthquake_data_path, tectonic_distance_path)
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()