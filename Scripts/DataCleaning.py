#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Cleaning Script for Earthquake Prediction Project

This script provides functions to clean and preprocess the three core datasets used in 
earthquake prediction analysis:
1. Earthquake records from USGS (20-year historical dataset)
2. Tectonic plate boundaries
3. Lunar position data

The focus is on preparing data for exploring relationships between seismic activity,
plate tectonics, and lunar influences.
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check for dependencies
def check_dependency(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        logging.warning(f"{package_name} not found. Some functionality will be limited.")
        return False

class EarthquakeDataCleaner:
    """
    Data cleaning class for earthquake prediction project.
    Handles cleaning and processing of earthquake data, tectonic plates,
    and lunar positions.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize with path to data directory.
        
        Args:
            data_dir (str): Path to data directory containing raw data folders
        """
        if data_dir is None:
            # Default to project structure
            self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Data"
            )
        else:
            self.data_dir = data_dir
            
        self.raw_data_dir = os.path.join(self.data_dir, "RawData")
        self.processed_data_dir = os.path.join(self.data_dir, "ProcessedData")
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
            
        # Define subdirectories
        self.earthquake_dir = os.path.join(self.raw_data_dir, "Earthquakes")
        self.tectonic_dir = os.path.join(self.raw_data_dir, "TectonicPlates")
        self.lunar_dir = os.path.join(self.raw_data_dir, "LunarData")
        
        # Create processed data subdirectories
        self.processed_earthquake_dir = os.path.join(self.processed_data_dir, "Earthquakes")
        if not os.path.exists(self.processed_earthquake_dir):
            os.makedirs(self.processed_earthquake_dir)
            logging.info(f"Created directory: {self.processed_earthquake_dir}")
            
        # Data containers
        self.earthquake_data = None
        self.tectonic_data = None
        self.lunar_data = None
        self.comcat_data = None
        
        logging.info(f"Initialized data cleaner with data directory: {self.data_dir}")
    
    def load_earthquake_data(self, recent_only=False, min_magnitude=None):
        """
        Load and clean USGS earthquake data.
        
        Args:
            recent_only (bool): If True, only load the most recent data file
            min_magnitude (float): Minimum earthquake magnitude to include
            
        Returns:
            pandas.DataFrame: Cleaned earthquake data
        """
        logging.info("Loading earthquake data...")
        
        # Find earthquake data files
        eq_files = [f for f in os.listdir(self.earthquake_dir) 
                   if f.endswith('.csv') and f.startswith('usgs')]
        
        if not eq_files:
            logging.warning("No earthquake data files found.")
            return None
        
        # Sort by modification time to get most recent file first
        eq_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.earthquake_dir, x)), 
                     reverse=True)
        
        if recent_only and eq_files:
            eq_files = [eq_files[0]]  # Keep only the most recent file
            
        dfs = []
        for file in eq_files:
            file_path = os.path.join(self.earthquake_dir, file)
            logging.info(f"Reading {file}...")
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
                
        if not dfs:
            logging.warning("No earthquake data could be loaded.")
            return None
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Clean and standardize columns
        if 'time' in combined_df.columns:
            combined_df['time'] = pd.to_datetime(combined_df['time'])
            
        # Rename columns for consistency
        column_map = {
            'mag': 'magnitude',
            'place': 'location',
            'longitude': 'lon',
            'latitude': 'lat'
        }
        
        for old_col, new_col in column_map.items():
            if old_col in combined_df.columns and new_col not in combined_df.columns:
                combined_df[new_col] = combined_df[old_col]
                
        # Filter by magnitude if specified
        if min_magnitude is not None and 'magnitude' in combined_df.columns:
            combined_df = combined_df[combined_df['magnitude'] >= min_magnitude]
            
        # Remove duplicates
        if 'id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['id'])
        else:
            # If no ID column, use time, location, and magnitude to identify duplicates
            subset = [col for col in ['time', 'latitude', 'longitude', 'magnitude'] 
                     if col in combined_df.columns]
            if subset:
                combined_df = combined_df.drop_duplicates(subset=subset)
                
        # Handle missing values
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)
        
        # Store cleaned data
        self.earthquake_data = combined_df
        
        logging.info(f"Loaded earthquake data with {combined_df.shape[0]} events")
        return self.earthquake_data
    
    def load_tectonic_plate_data(self):
        """
        Load and clean tectonic plate boundary data.
        
        Returns:
            geopandas.GeoDataFrame: Cleaned tectonic plate boundary data
        """
        if not check_dependency('geopandas'):
            logging.warning("geopandas not found. Cannot process tectonic plate data.")
            return None
            
        logging.info("Loading tectonic plate data...")
        
        # Find GeoJSON files
        plate_files = [f for f in os.listdir(self.tectonic_dir) if f.endswith('.json')]
        
        if not plate_files:
            logging.warning("No tectonic plate data files found.")
            return None
            
        # Load boundaries file
        boundaries_file = next((f for f in plate_files if 'boundaries' in f), None)
        
        if not boundaries_file:
            logging.warning("No tectonic plate boundaries file found.")
            return None
            
        try:
            boundaries_path = os.path.join(self.tectonic_dir, boundaries_file)
            gdf = gpd.read_file(boundaries_path)
            
            # Check and repair geometries if needed
            if not all(gdf.is_valid):
                logging.info("Repairing invalid geometries...")
                gdf['geometry'] = gdf['geometry'].apply(lambda g: g.buffer(0) if not g.is_valid else g)
                
            # Ensure CRS is WGS84 (EPSG:4326)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs(epsg=4326)
                
            # Store cleaned data
            self.tectonic_data = gdf
            
            logging.info(f"Loaded tectonic data with {gdf.shape[0]} features")
            return self.tectonic_data
            
        except Exception as e:
            logging.error(f"Error loading tectonic plate data: {e}")
            return None
    
    def load_lunar_position_data(self):
        """
        Load and clean lunar position data.
        
        Returns:
            pandas.DataFrame: Cleaned lunar position data
        """
        logging.info("Loading lunar position data...")
        
        lunar_file = os.path.join(self.lunar_dir, "lunar_positions.csv")
        
        if not os.path.exists(lunar_file):
            logging.warning("No lunar position data file found.")
            return None
            
        try:
            df = pd.read_csv(lunar_file)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
            # Sort by date
            df = df.sort_values('date')
            
            # Handle missing values
            df = df.dropna()
            
            # Store cleaned data
            self.lunar_data = df
            
            logging.info(f"Loaded lunar data with {df.shape[0]} positions")
            return self.lunar_data
            
        except Exception as e:
            logging.error(f"Error loading lunar position data: {e}")
            return None
    
    def clean_comcat_data(self, min_magnitude=4.0, start_date=None, end_date=None):
        """
        Clean and combine ComCat earthquake data, which contains the full 20-year dataset.
        
        Args:
            min_magnitude (float): Minimum earthquake magnitude to include
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Cleaned ComCat earthquake data
        """
        logging.info("Cleaning ComCat earthquake data...")
        
        # Find ComCat data files (assuming they follow a pattern)
        comcat_files = [f for f in os.listdir(self.earthquake_dir) 
                       if f.endswith('.csv') and 'comcat' in f.lower()]
        
        if not comcat_files:
            logging.warning("No ComCat data files found.")
            return None
        
        # Sort files by date if they have date information in filename
        # This assumes a naming convention like comcat_YYYYMMDD.csv
        date_pattern = re.compile(r'(\d{8})')
        comcat_files.sort(key=lambda x: date_pattern.search(x).group(1) if date_pattern.search(x) else x)
        
        logging.info(f"Found {len(comcat_files)} ComCat data files: {comcat_files}")
        
        dfs = []
        for file in comcat_files:
            file_path = os.path.join(self.earthquake_dir, file)
            logging.info(f"Reading {file}...")
            try:
                df = pd.read_csv(file_path)
                logging.info(f"File {file} has {len(df)} rows and columns: {df.columns.tolist()}")
                
                # Check if 'time' column exists and is in the expected format
                if 'time' in df.columns:
                    # USGS data typically has timestamp in milliseconds
                    # Convert to datetime
                    try:
                        # First, check if we're dealing with milliseconds since epoch (appears as large integers)
                        if pd.api.types.is_numeric_dtype(df['time']) or df['time'].astype(str).str.match(r'^\d{13}$').all():
                            # Convert milliseconds since epoch to datetime
                            df['time'] = pd.to_datetime(df['time'].astype(float), unit='ms')
                            logging.info(f"Converted time column from milliseconds since epoch to datetime")
                        else:
                            # Try to convert to datetime if already in ISO format
                            df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    except Exception as e:
                        logging.error(f"Failed to convert time column: {e}")
                
                    # Log the date range
                    logging.info(f"Date range in file: {df['time'].min()} to {df['time'].max()}")
                
                # Check magnitude column
                if 'mag' in df.columns:
                    logging.info(f"Magnitude range: {df['mag'].min()} to {df['mag'].max()}")
                    logging.info(f"Earthquakes ≥ M{min_magnitude}: {len(df[df['mag'] >= min_magnitude])}")
                
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
        
        if not dfs:
            logging.warning("No ComCat data could be loaded.")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Combined data has {len(combined_df)} rows.")
        
        # Standard column names for the combined data
        if 'time' in combined_df.columns:
            # Ensure time is in datetime format
            try:
                # For large integer values (milliseconds since epoch)
                if pd.api.types.is_numeric_dtype(combined_df['time']) or combined_df['time'].astype(str).str.match(r'^\d{13}$').all():
                    combined_df['time'] = pd.to_datetime(combined_df['time'].astype(float), unit='ms')
                    logging.info("Converted numeric time to datetime (milliseconds since epoch)")
                else:
                    combined_df['time'] = pd.to_datetime(combined_df['time'])
                    logging.info("Converted string time to datetime")
            except Exception as e:
                logging.error(f"Error converting time to datetime: {e}")
                # Create a backup of the original time column
                combined_df['time_original'] = combined_df['time'].copy()
                # Try a different approach for large integer values
                try:
                    # Force conversion to float then to datetime with millisecond units
                    combined_df['time'] = pd.to_datetime(combined_df['time'].astype(float), unit='ms')
                    logging.info("Converted time using float conversion and milliseconds")
                except Exception as e:
                    logging.error(f"Failed alternative time conversion: {e}")
            
            # Log the date range after conversion
            logging.info(f"Date range after time conversion: {combined_df['time'].min()} to {combined_df['time'].max()}")
            
        # Filter by date range if provided
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Log counts before filtering
            original_count = len(combined_df)
            logging.info(f"Filtering by date range: {start_date} to {end_date}")
            
            # Filter by date range
            combined_df = combined_df[(combined_df['time'] >= start_date) & 
                                      (combined_df['time'] <= end_date)]
            
            logging.info(f"Filtered earthquakes from {start_date} to {end_date} (kept {len(combined_df)}/{original_count} events)")
            
        # Filter by magnitude
        if 'mag' in combined_df.columns and min_magnitude is not None:
            original_count = len(combined_df)
            
            # Check for NaN values in mag column
            nan_count = combined_df['mag'].isna().sum()
            if nan_count > 0:
                logging.warning(f"Found {nan_count} NaN values in magnitude column")
                
            # Filter by magnitude and handle NaN values
            combined_df = combined_df[combined_df['mag'] >= min_magnitude]
            logging.info(f"Filtered earthquakes ≥ M{min_magnitude} (kept {len(combined_df)}/{original_count} events)")
        
        # Standardize column names
        column_map = {
            'mag': 'magnitude',
            'place': 'location',
            'longitude': 'lon',
            'latitude': 'lat'
        }
        
        for old_col, new_col in column_map.items():
            if old_col in combined_df.columns and new_col not in combined_df.columns:
                combined_df[new_col] = combined_df[old_col]
        
        # Add geometry column for spatial analysis if lat/lon are available
        if all(col in combined_df.columns for col in ['lat', 'lon']):
            combined_df['geometry'] = combined_df.apply(
                lambda row: Point(row['lon'], row['lat']), axis=1)
        
        # Save to variable and disk
        if len(combined_df) > 0:
            self.comcat_data = combined_df
            
            # Save to processed data folder
            output_path = os.path.join(self.processed_earthquake_dir, "comcat_cleaned.csv")
            combined_df.to_csv(output_path, index=False)
            logging.info(f"Saved cleaned ComCat data with {len(combined_df)} events to {output_path}")
        
        return self.comcat_data

    def process_lunar_data(self, start_date=None, end_date=None):
        """
        Process lunar position data to calculate phases and other parameters.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Processed lunar data with phases and distances
        """
        logging.info("Processing lunar data...")
        
        # Find lunar data files
        lunar_files = [f for f in os.listdir(self.lunar_dir) 
                      if f.endswith('.csv') and 'lunar' in f.lower()]
        
        if not lunar_files:
            logging.warning("No lunar data files found.")
            return None
        
        # Use the most comprehensive file (should be the 20-year dataset)
        lunar_file = max(lunar_files, key=lambda x: os.path.getsize(os.path.join(self.lunar_dir, x)))
        lunar_path = os.path.join(self.lunar_dir, lunar_file)
        
        try:
            lunar_df = pd.read_csv(lunar_path)
            
            # Make sure date is in datetime format
            if 'date' in lunar_df.columns:
                lunar_df['date'] = pd.to_datetime(lunar_df['date'])
            
            # Filter by date range if provided
            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                lunar_df = lunar_df[(lunar_df['date'] >= start) & (lunar_df['date'] <= end)]
                logging.info(f"Filtered lunar data from {start} to {end}")
            
            # Calculate lunar phase (0-1 where 0=new moon, 0.5=full moon)
            if 'illumination' in lunar_df.columns:
                # Already have illumination data
                lunar_df['phase'] = lunar_df['illumination']
            elif all(col in lunar_df.columns for col in ['sun_angle', 'moon_angle']):
                # Calculate phase from angles
                lunar_df['phase'] = lunar_df.apply(
                    lambda row: (1 + np.cos(np.radians(row['sun_angle'] - row['moon_angle']))) / 2,
                    axis=1
                )
            
            # Calculate simplified phase names
            def get_phase_name(phase):
                if phase < 0.05:
                    return 'New Moon'
                elif phase < 0.25:
                    return 'Waxing Crescent'
                elif phase < 0.30:
                    return 'First Quarter'
                elif phase < 0.45:
                    return 'Waxing Gibbous'
                elif phase < 0.55:
                    return 'Full Moon'
                elif phase < 0.70:
                    return 'Waning Gibbous'
                elif phase < 0.80:
                    return 'Last Quarter'
                elif phase < 0.95:
                    return 'Waning Crescent'
                else:
                    return 'New Moon'
            
            # Add phase name column if phase data is available
            if 'phase' in lunar_df.columns:
                lunar_df['phase_name'] = lunar_df['phase'].apply(get_phase_name)
            
            # Calculate distance categories if distance data is available
            if 'distance_km' in lunar_df.columns:
                # Calculate apogee/perigee thresholds (nearest/farthest 10%)
                distance_threshold_low = lunar_df['distance_km'].quantile(0.1)
                distance_threshold_high = lunar_df['distance_km'].quantile(0.9)
                
                def get_distance_category(distance):
                    if distance <= distance_threshold_low:
                        return 'Perigee (Near)'
                    elif distance >= distance_threshold_high:
                        return 'Apogee (Far)'
                    else:
                        return 'Average'
                
                lunar_df['distance_category'] = lunar_df['distance_km'].apply(get_distance_category)
            
            # Save processed data to variable and disk
            self.lunar_data = lunar_df
            
            output_path = os.path.join(self.processed_data_dir, "lunar_processed.csv")
            lunar_df.to_csv(output_path, index=False)
            logging.info(f"Saved processed lunar data with {len(lunar_df)} days to {output_path}")
            
            return self.lunar_data
            
        except Exception as e:
            logging.error(f"Error processing lunar data: {e}")
            return None

    def calculate_tectonic_distance(self):
        """
        Calculate the distance of each earthquake to the nearest tectonic plate boundary.
        
        Returns:
            pandas.DataFrame: Earthquake data with added tectonic distance columns
        """
        logging.info("Calculating distance to tectonic plate boundaries...")
        
        if not check_dependency('geopandas'):
            logging.error("geopandas is required for tectonic distance calculation.")
            return None
            
        if self.comcat_data is None:
            logging.warning("No earthquake data available. Please run clean_comcat_data first.")
            return None
            
        if self.tectonic_data is None:
            # Load tectonic data if not already loaded
            self.load_tectonic_plate_data()
            
        if self.tectonic_data is None:
            logging.warning("Failed to load tectonic plate data.")
            return None
            
        # Create GeoDataFrame from earthquake data
        eq_df = self.comcat_data.copy()
        
        if 'geometry' not in eq_df.columns:
            # Create geometry from lat/lon
            if all(col in eq_df.columns for col in ['lat', 'lon']):
                eq_df['geometry'] = eq_df.apply(
                    lambda row: Point(row['lon'], row['lat']), axis=1)
            else:
                logging.error("Earthquake data missing lat/lon columns.")
                return None
                
        # Convert to GeoDataFrame
        eq_gdf = gpd.GeoDataFrame(eq_df, geometry='geometry', crs="EPSG:4326")
        
        # Ensure tectonic data is GeoDataFrame
        tect_gdf = self.tectonic_data
        
        # Calculate distance to nearest plate boundary for each earthquake
        logging.info("Calculating distances to nearest plate boundary (this may take a while)...")
        
        # Create a faster implementation using spatial indexing
        from shapely.strtree import STRtree
        import numpy as np
        from tqdm import tqdm  # For progress bar
        
        # Convert to a format suitable for spatial indexing
        logging.info("Converting coordinates for distance calculation...")
        
        # Convert tectonic boundaries to projected coordinates for accurate distance measurement
        boundaries_meters = tect_gdf.to_crs("EPSG:3857").geometry
        
        # Build spatial index of tectonic boundaries for much faster lookups
        logging.info("Building spatial index for tectonic boundaries...")
        spatial_index = STRtree(boundaries_meters.values)
        boundaries_array = np.array(boundaries_meters)
        
        logging.info(f"Processing {len(eq_gdf)} earthquakes (using spatial index for speed)...")
        
        # Pre-calculate the projected points (this is faster than doing it inside the loop)
        points_meters = eq_gdf.to_crs("EPSG:3857").geometry
        
        # Calculate distances with progress bar
        distances = []
        for i, point in enumerate(tqdm(points_meters, desc="Calculating distances")):
            # Use spatial index to find nearest boundary (much faster than checking all)
            nearest_idx = spatial_index.nearest(point)
            # Calculate distance in meters then convert to km
            min_distance = point.distance(boundaries_array[nearest_idx]) / 1000
            distances.append(min_distance)
            
            # Log progress periodically
            if (i+1) % 10000 == 0:
                logging.info(f"Processed {i+1}/{len(points_meters)} earthquakes ({((i+1)/len(points_meters))*100:.1f}%)")
        
        # Assign results to DataFrame
        eq_gdf['distance_to_boundary_km'] = distances
        
        # Create bins for distance categories for easier analysis
        distance_bins = [0, 50, 100, 200, 500, float('inf')]
        distance_labels = ['0-50km', '50-100km', '100-200km', '200-500km', '>500km']
        
        eq_gdf['boundary_distance_category'] = pd.cut(
            eq_gdf['distance_to_boundary_km'], 
            bins=distance_bins, 
            labels=distance_labels
        )
        
        # Save results
        output_path = os.path.join(self.processed_earthquake_dir, "earthquake_tectonic_distance_with_wkt.csv")
        try:
            eq_gdf.to_csv(output_path, index=False)
            logging.info(f"Saved tectonic distance data to {output_path}")
        except Exception as e:
            logging.error(f"Error saving tectonic distance data: {str(e)}")
        
        # Save as GeoParquet for spatial analysis instead of GeoJSON (supports multiple geometry columns)
        output_parquet = os.path.join(self.processed_earthquake_dir, "earthquake_tectonic_distance.parquet")
        try:
            # Make sure 'geometry' is set as the active geometry column
            if 'geometry' in eq_gdf.columns:
                eq_gdf = eq_gdf.set_geometry('geometry')
                
                # Save to parquet file which supports multiple geometry columns
                eq_gdf.to_parquet(output_parquet)
                logging.info(f"Saved earthquake data with tectonic distances to {output_path} and {output_parquet}")
            else:
                logging.warning("No geometry column found in earthquake data")
                
        except Exception as e:
            logging.error(f"Error saving geospatial data: {e}")
            # As a fallback, save CSV with WKT geometry
            if 'geometry' in eq_gdf.columns:
                eq_gdf['geometry_wkt'] = eq_gdf['geometry'].apply(lambda geom: geom.wkt if geom else None)
            backup_path = os.path.join(self.processed_earthquake_dir, "earthquake_tectonic_distance_with_wkt.csv")
            eq_gdf.drop(columns=['geometry'] if 'geometry' in eq_gdf.columns else []).to_csv(backup_path, index=False)
            logging.info(f"Saved fallback CSV with WKT geometry to {backup_path}")
        
        return eq_gdf
    
    def create_integrated_dataset(self):
        """
        Create an integrated dataset for analysis
        
        Note: Integration with lunar data is now handled in Luna_VS_EarthPositions.py
        This function now only calculates tectonic boundary distances.
        
        Returns:
            pd.DataFrame: DataFrame with tectonic boundary distances
        """
        logging.info("Creating integrated dataset for analysis...")
        
        # First calculate tectonic boundary distances
        logging.info("Calculating tectonic boundary distances...")
        eq_tect_df = self.calculate_tectonic_distance()
        
        if eq_tect_df is None:
            logging.error("Failed to calculate tectonic distances.")
            return None
            
        # Save the tectonic distance data
        output_path = os.path.join(self.processed_earthquake_dir, "earthquake_tectonic_distance_with_wkt.csv")
        try:
            eq_tect_df.to_csv(output_path, index=False)
            logging.info(f"Saved tectonic distance data to {output_path}")
        except Exception as e:
            logging.error(f"Error saving tectonic distance data: {str(e)}")
        
        # Return the DataFrame with tectonic distances
        return eq_tect_df

def main():
    """Main function to run data cleaning and processing"""
    logging.info("Starting data cleaning process")
    
    # Initialize data cleaner
    cleaner = EarthquakeDataCleaner()
    
    # Define date range (20-year historical period)
    start_date = "2005-03-25"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Process earthquake data
    cleaner.clean_comcat_data(min_magnitude=4.0, start_date=start_date, end_date=end_date)
    
    # Process tectonic plate data
    cleaner.load_tectonic_plate_data()
    
    # Process lunar data (basic cleaning only, integration happens in Luna_VS_EarthPositions.py)
    cleaner.process_lunar_data(start_date=start_date, end_date=end_date)
    
    # Calculate distances to tectonic plate boundaries
    tectonic_data = cleaner.create_integrated_dataset()
    
    if tectonic_data is not None:
        logging.info(f"Successfully calculated tectonic distances for {len(tectonic_data)} records")
        logging.info("Run Luna_VS_EarthPositions.py to integrate with lunar data and create the final dataset")
    else:
        logging.error("Failed to calculate tectonic distances")
    
    logging.info("Data cleaning process complete")

if __name__ == "__main__":
    main()