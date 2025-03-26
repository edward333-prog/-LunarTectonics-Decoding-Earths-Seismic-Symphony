#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Download Script for Earthquake Prediction Project

This script downloads the three primary datasets needed for earthquake prediction:
1. Earthquake records from USGS
2. Tectonic plate boundaries 
3. Lunar cycle data for correlation analysis

The focus is on exploring relationships between seismic activity,
plate tectonics, and lunar influences.
"""

import os
import logging
import requests
import json
from datetime import datetime, timedelta
import importlib.util
from tqdm import tqdm
import pandas as pd
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Check for dependencies and import them if available
def check_dependency(package_name):
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package_name) is not None

# Initialize a dictionary to track which dependencies are available
DEPENDENCIES = {
    'pandas': check_dependency('pandas'),
    'skyfield': check_dependency('skyfield'),
}

# Log available and missing dependencies
for dep, available in DEPENDENCIES.items():
    if not available:
        logging.warning(f"{dep} not found. Some functionality will be limited.")

# Import optional dependencies if available
if DEPENDENCIES['pandas']:
    import pandas as pd

# For astronomical calculations
if DEPENDENCIES['skyfield']:
    from skyfield.api import load

# Define directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "RawData")
PROCESSED_DIR = os.path.join(DATA_DIR, "ProcessedData")

# Create subdirectories for each data type
EARTHQUAKE_DIR = os.path.join(RAW_DATA_DIR, "Earthquakes")
TECTONIC_DIR = os.path.join(RAW_DATA_DIR, "TectonicPlates")
LUNAR_DIR = os.path.join(RAW_DATA_DIR, "LunarData")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, 
                 EARTHQUAKE_DIR, TECTONIC_DIR, LUNAR_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def download_file(url, filename, directory, overwrite=False):
    """
    Download a file from a URL and save it to the specified directory.
    
    Args:
        url (str): URL to download the file from
        filename (str): Name to save the file as
        directory (str): Directory to save the file in
        overwrite (bool): Whether to overwrite the file if it exists
        
    Returns:
        str: Path to the downloaded file or None if download failed
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    filepath = os.path.join(directory, filename)
    
    # Check if file already exists
    if os.path.exists(filepath) and not overwrite:
        logging.info(f"File already exists: {filepath}")
        return filepath
    
    # Download the file
    try:
        logging.info(f"Downloading {filename} from {url}")
        
        # Use tqdm for a progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        logging.info(f"Successfully downloaded: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return None

def download_comcat_data(start_date=None, end_date=None, min_magnitude=4.5, chunk_size_days=365, keep_individual_files=False):
    """
    Download comprehensive earthquake data from USGS ComCat using direct API requests
    
    This provides more detailed earthquake data than the basic USGS API,
    including moment tensors, focal mechanisms, and more complete historical coverage.
    
    Args:
        start_date (datetime): Start date (defaults to 20 years ago)
        end_date (datetime): End date (defaults to today)
        min_magnitude (float): Minimum earthquake magnitude to include
        chunk_size_days (int): Size of date chunks in days (to avoid timeouts)
        keep_individual_files (bool): Whether to keep individual chunk files after combining
        
    Returns:
        list: Paths to the downloaded files or empty list if download failed
    """
    logging.info("Downloading ComCat data from {} to {}...".format(
        start_date.strftime('%Y-%m-%d') if start_date else "20 years ago",
        end_date.strftime('%Y-%m-%d') if end_date else "today"
    ))
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        # Default to 20 years of data
        start_date = end_date - timedelta(days=365 * 20)
    
    # Format for API requests
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download in chunks to avoid timeouts
    current_start = start_date
    chunk_files_json = []
    chunk_files_csv = []
    
    while current_start < end_date:
        # Calculate end of this chunk
        current_end = current_start + timedelta(days=chunk_size_days)
        if current_end > end_date:
            current_end = end_date
            
        # Format dates for this chunk
        chunk_start_str = current_start.strftime('%Y-%m-%d')
        chunk_end_str = current_end.strftime('%Y-%m-%d')
        
        logging.info(f"Downloading chunk: {chunk_start_str} to {chunk_end_str}")
        
        # Construct filename and URL
        filename = f"comcat_earthquakes_{chunk_start_str}_to_{chunk_end_str}.json"
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime={chunk_start_str}&endtime={chunk_end_str}&minmagnitude={min_magnitude}&orderby=time&includedeleted=false"
        
        # Download the chunk file
        json_file_path = download_file(url, filename, EARTHQUAKE_DIR)
        
        if json_file_path:
            chunk_files_json.append(json_file_path)
            
            # Convert to CSV if pandas is available
            if DEPENDENCIES['pandas']:
                try:
                    csv_file_path = json_file_path.replace('.json', '.csv')
                    
                    # Open and load the GeoJSON file
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        # Use json.load instead of pd.read_json to avoid pandas issues
                        data = json.load(f)
                    
                    # Process each feature one by one and build a proper list of dicts
                    features_list = []
                    
                    for feature in data.get('features', []):
                        if not feature or 'properties' not in feature:
                            continue
                            
                        # Start with an empty record
                        record = {}
                        
                        # Add all properties as flattened fields
                        if feature.get('properties'):
                            for key, value in feature['properties'].items():
                                record[key] = value
                        
                        # Add geometry if available
                        if feature.get('geometry') and feature['geometry'] is not None:
                            if feature['geometry'].get('coordinates') and len(feature['geometry']['coordinates']) >= 3:
                                coords = feature['geometry']['coordinates']
                                record['longitude'] = coords[0]
                                record['latitude'] = coords[1]
                                record['depth'] = coords[2]
                        
                        features_list.append(record)
                    
                    # Only proceed if we found features
                    if features_list:
                        # Create DataFrame with explicit handling for missing fields
                        df = pd.json_normalize(features_list)
                        
                        # Write to CSV
                        df.to_csv(csv_file_path, index=False)
                        chunk_files_csv.append(csv_file_path)
                        logging.info(f"Converted ComCat data to CSV: {csv_file_path}")
                    else:
                        logging.warning(f"No valid features found in {json_file_path}")
                except Exception as e:
                    logging.error(f"Error converting JSON to CSV: {e}")
        
        # Move to the next chunk
        current_start = current_end + timedelta(days=1)
    
    # If no files were downloaded, return empty list
    if not chunk_files_json:
        logging.warning("No ComCat data was downloaded")
        return []
    
    # Combine all the chunks into a single file
    combined_json_path = os.path.join(
        EARTHQUAKE_DIR,
        f"comcat_earthquakes_{start_str}_to_{end_str}_combined.json"
    )
    
    combined_csv_path = combined_json_path.replace('.json', '.csv')
    
    try:
        # Combine JSON files
        combined_data = {"type": "FeatureCollection", "features": []}
        
        for json_file in chunk_files_json:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunk_data = json.loads(f.read())
                combined_data["features"].extend(chunk_data["features"])
                
        with open(combined_json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f)
            
        logging.info(f"Created combined ComCat dataset: {combined_json_path}")
        
        # Combine CSV files if pandas is available
        if DEPENDENCIES['pandas'] and chunk_files_csv:
            # Read and combine files
            try:
                # First check if we have any CSV files
                if not chunk_files_csv:
                    logging.warning("No CSV files to combine")
                elif len(chunk_files_csv) == 1:
                    # If only one file, just copy it
                    shutil.copy(chunk_files_csv[0], combined_csv_path)
                    logging.info(f"Single CSV file copied to {combined_csv_path}")
                else:
                    # Read all CSV files with pandas
                    all_dfs = []
                    
                    for csv_file in chunk_files_csv:
                        try:
                            df = pd.read_csv(csv_file)
                            all_dfs.append(df)
                        except Exception as csv_err:
                            logging.warning(f"Could not read {csv_file}: {csv_err}")
                    
                    # Combine all dataframes, automatically aligning columns
                    if all_dfs:
                        # Use pandas.concat which handles different columns properly
                        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
                        combined_df.to_csv(combined_csv_path, index=False)
                        logging.info(f"Created combined ComCat CSV with {len(combined_df)} records: {combined_csv_path}")
                    else:
                        logging.warning("No valid dataframes to combine")
            except Exception as e:
                logging.error(f"Error creating combined CSV: {str(e)}")
                    
        # Clean up individual chunk files if not keeping them
        if not keep_individual_files:
            logging.info("Cleaning up individual chunk files...")
            for file_path in chunk_files_json + chunk_files_csv:
                try:
                    os.remove(file_path)
                    logging.debug(f"Removed temporary file: {file_path}")
                except Exception as e:
                    logging.warning(f"Could not remove {file_path}: {e}")
                    
    except Exception as e:
        logging.error(f"Error combining ComCat data: {e}")
    
    logging.info("ComCat data download complete")
    return [combined_json_path, combined_csv_path]

def download_tectonic_plate_boundaries():
    """Download global tectonic plate boundaries"""
    # Plate boundaries from GitHub (simplified version of Bird 2003 plates)
    plates_url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"
    filepath = download_file(plates_url, "tectonic_plate_boundaries.json", TECTONIC_DIR)
    
    # Also download the plates polygons
    plates_polygons_url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_plates.json"
    polygons_filepath = download_file(plates_polygons_url, "tectonic_plates_polygons.json", TECTONIC_DIR)
    
    # Download plate boundary types (orogens) - fixed URL
    plates_types_url = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_orogens.json"
    types_filepath = download_file(plates_types_url, "tectonic_plates_orogens.json", TECTONIC_DIR)
    
    return [filepath, polygons_filepath, types_filepath]

def download_lunar_position_data(start_date=None, end_date=None):
    """
    Download lunar ephemeris data and calculate lunar distance metrics
    that are most relevant for earthquake triggering analysis.
    
    This function focuses specifically on the lunar distance from Earth,
    which is the primary factor affecting gravitational pull. It identifies
    perigee (closest approach) and apogee (furthest distance) events,
    which represent the strongest and weakest gravitational influence periods.
    
    The moon's distance from Earth varies by approximately 50,000 km between
    perigee and apogee, causing approximately 20% difference in gravitational force.
    This is the primary lunar factor that could potentially influence global
    earthquake patterns.
    
    Args:
        start_date (datetime): Start date (defaults to 20 years ago)
        end_date (datetime): End date (defaults to today)
        
    Returns:
        str: Path to the saved file with lunar distance data
    """
    if not DEPENDENCIES['skyfield']:
        logging.warning("Missing required dependencies: skyfield")
        logging.warning("Skipping lunar data download due to missing dependencies")
        return None
    
    try:
        # Using skyfield to handle lunar ephemeris data
        logging.info("Downloading lunar ephemeris data via Skyfield")
        
        # Load ephemeris data (this will download DE421 if needed)
        eph = load('de421.bsp')
        logging.info("Lunar ephemeris data loaded successfully")
        
        # Extract Earth and Moon objects
        earth = eph['earth']
        moon = eph['moon']
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            # Default to match the earthquake data range (20 years)
            start_date = end_date - timedelta(days=365 * 20)
        
        # Create a time series for the analysis
        ts = load.timescale()
        days = (end_date - start_date).days
        logging.info(f"Calculating lunar distances for {days} days (from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        # Calculate lunar distances from Earth
        lunar_data = []
        
        # Generate data for each day in the time range
        current_date = start_date
        
        # For tracking perigee and apogee
        window_size = 15  # Days to look before and after for local min/max
        distance_buffer = []
        
        while current_date <= end_date:
            # Create time object for this date
            t = ts.utc(current_date.year, current_date.month, current_date.day)
            
            # Get position of moon relative to earth
            pos = earth.at(t).observe(moon)
            distance = pos.distance().km
            
            # Record date and distance
            lunar_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'distance_km': distance,
                'is_perigee': False,  # Will be updated later
                'is_apogee': False,   # Will be updated later
            })
            
            # Add to buffer for perigee/apogee detection
            distance_buffer.append((current_date, distance))
            if len(distance_buffer) > window_size * 2 + 1:
                distance_buffer.pop(0)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Identify perigee and apogee events (local minima and maxima)
        logging.info("Identifying perigee and apogee events...")
        
        for i in range(len(lunar_data) - window_size):
            if i < window_size:
                continue
                
            # Get the central date and its distance
            center_idx = i
            center_date_str = lunar_data[center_idx]['date']
            center_distance = lunar_data[center_idx]['distance_km']
            
            # Check for local minimum (perigee)
            is_perigee = True
            for j in range(max(0, i-window_size), min(len(lunar_data), i+window_size+1)):
                if j != i and lunar_data[j]['distance_km'] < center_distance:
                    is_perigee = False
                    break
            
            # Check for local maximum (apogee)
            is_apogee = True
            for j in range(max(0, i-window_size), min(len(lunar_data), i+window_size+1)):
                if j != i and lunar_data[j]['distance_km'] > center_distance:
                    is_apogee = False
                    break
            
            if is_perigee:
                lunar_data[center_idx]['is_perigee'] = True
                logging.info(f"Perigee (closest approach) identified on {center_date_str}: {center_distance:.0f} km")
            
            if is_apogee:
                lunar_data[center_idx]['is_apogee'] = True
                logging.info(f"Apogee (furthest distance) identified on {center_date_str}: {center_distance:.0f} km")
        
        # Calculate gravitational force index (proportional to 1/r²)
        # Higher values = stronger gravitational pull
        for item in lunar_data:
            distance_km = item['distance_km']
            # Normalize around average distance (384,400 km)
            item['gravity_index'] = (384400 / distance_km) ** 2
        
        # Save the data if pandas is available
        if DEPENDENCIES['pandas']:
            df = pd.DataFrame(lunar_data)
            lunar_data_path = os.path.join(LUNAR_DIR, "lunar_distance_data.csv")
            df.to_csv(lunar_data_path, index=False)
            logging.info(f"Lunar distance data saved to {lunar_data_path}")
            return lunar_data_path
        else:
            # Save as simple CSV if pandas not available
            lunar_data_path = os.path.join(LUNAR_DIR, "lunar_distance_data.csv")
            with open(lunar_data_path, 'w') as f:
                f.write("date,distance_km,is_perigee,is_apogee,gravity_index\n")
                for item in lunar_data:
                    f.write(f"{item['date']},{item['distance_km']},{item['is_perigee']},{item['is_apogee']},{item['gravity_index']}\n")
            logging.info(f"Lunar distance data saved to {lunar_data_path}")
            return lunar_data_path
    
    except Exception as e:
        logging.error(f"Error downloading or processing lunar data: {str(e)}")
        return None

def main():
    """Main function to run all data downloads"""
    logging.info("Starting data download process for Earthquake Prediction Project")
    
    # Track download status
    download_status = {
        'comcat': False,
        'tectonic_plates': False,
        'lunar_data': False
    }
    
    # Define our date range as datetime objects
    start_date = datetime.strptime("2005-03-26", "%Y-%m-%d")
    end_date = datetime.strptime("2025-03-21", "%Y-%m-%d")
    
    # Download ComCat data
    comcat_files = download_comcat_data(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=4.5,
        chunk_size_days=365,
        keep_individual_files=False
    )
    download_status['comcat'] = len(comcat_files) > 0

    # Download tectonic plate boundaries
    tectonic_files = download_tectonic_plate_boundaries()
    download_status['tectonic_plates'] = len(tectonic_files) > 0
    
    # Download lunar position data
    lunar_file = download_lunar_position_data(
        start_date=start_date,
        end_date=end_date
    )
    download_status['lunar_data'] = lunar_file is not None
    
    # Print download status summary
    logging.info("=== Download Status Summary ===")
    for data_type, status in download_status.items():
        logging.info(f"{data_type}: {'SUCCESS' if status else 'FAILED'}")
    
    successful_downloads = sum(1 for status in download_status.values() if status)
    total_downloads = len(download_status)
    logging.info(f"Overall: {successful_downloads}/{total_downloads} downloads successful")
    logging.info("Data download process complete")

if __name__ == "__main__":
    main()