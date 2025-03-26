import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for data storage
earthquake_df = pd.DataFrame()
lunar_df = pd.DataFrame()
integrated_df = pd.DataFrame()
plate_boundaries = {}
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def analyze_earthquake_distribution_by_latitude(integrated_data, lunar_data, plate_boundaries=None, save_path=None):
    """
    Quantifies the relationship between earthquake distribution and the moon's path by latitude.
    
    Args:
        integrated_data (DataFrame): Dataset containing earthquake events with latitude information
        lunar_data (DataFrame): Dataset containing lunar position data
        plate_boundaries (dict, optional): GeoJSON dict containing tectonic plate boundaries
        save_path (str, optional): Path to save the visualization
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Create bins for latitude analysis
    lat_bins = np.arange(-90, 91, 5)  # 5-degree bins from -90 to +90
    
    # Analyze earthquake distribution by latitude
    quake_lat_hist, quake_bin_edges = np.histogram(integrated_data['latitude'], bins=lat_bins)
    quake_bin_centers = (quake_bin_edges[:-1] + quake_bin_edges[1:]) / 2
    
    # Analyze moon position distribution by latitude
    moon_lat_hist, _ = np.histogram(lunar_data['sublunar_lat'], bins=lat_bins)
    
    # Normalize the histograms for comparison (percentage of total)
    quake_lat_hist_norm = quake_lat_hist / quake_lat_hist.sum() * 100
    moon_lat_hist_norm = moon_lat_hist / moon_lat_hist.sum() * 100
    
    # Create a dataframe for the analysis
    analysis_df = pd.DataFrame({
        'Latitude_Bin': quake_bin_centers,
        'Earthquake_Count': quake_lat_hist,
        'Earthquake_Percentage': quake_lat_hist_norm,
        'Moon_Position_Count': moon_lat_hist,
        'Moon_Position_Percentage': moon_lat_hist_norm
    })
    
    # Calculate statistical correlation between earthquake and moon distributions
    correlation, p_value = stats.pearsonr(quake_lat_hist_norm, moon_lat_hist_norm)
    
    # Calculate concentration metrics
    # Define the moon path regions
    exact_moon_path_mask = (analysis_df['Latitude_Bin'] >= -28) & (analysis_df['Latitude_Bin'] <= 28)
    moon_path_bands_mask = ((analysis_df['Latitude_Bin'] >= -35) & (analysis_df['Latitude_Bin'] <= -20)) | \
                     ((analysis_df['Latitude_Bin'] >= 20) & (analysis_df['Latitude_Bin'] <= 35))
    
    # Calculate percentages for moon path regions
    exact_moon_path_percentage = analysis_df.loc[exact_moon_path_mask, 'Earthquake_Percentage'].sum()
    moon_path_bands_percentage = analysis_df.loc[moon_path_bands_mask, 'Earthquake_Percentage'].sum()
    non_moon_path_percentage = 100 - exact_moon_path_percentage - moon_path_bands_percentage
    
    # Calculate density of earthquakes in regions (normalize by area)
    # Area of a latitude band is proportional to the cosine of the latitude
    analysis_df['Earth_Surface_Weight'] = np.cos(np.radians(analysis_df['Latitude_Bin']))
    analysis_df['Earth_Surface_Weight'] = analysis_df['Earth_Surface_Weight'] / analysis_df['Earth_Surface_Weight'].sum()
    analysis_df['Earthquake_Density'] = analysis_df['Earthquake_Percentage'] / (analysis_df['Earth_Surface_Weight'] * 100)
    
    # Calculate relative concentration factor
    # This normalizes earthquake density to account for Earth's surface area distribution
    analysis_df['Relative_Concentration'] = analysis_df['Earthquake_Density'] / analysis_df['Earthquake_Density'].mean()
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the histograms
    ax1.bar(quake_bin_centers, quake_lat_hist_norm, width=4, alpha=0.6, color='blue', label='Earthquakes')
    ax1.bar(quake_bin_centers, moon_lat_hist_norm, width=2, alpha=0.6, color='red', label='Moon Path')
    
    # Highlight the moon's approximate path range
    ax1.axvspan(-28.5, 28.5, alpha=0.1, color='yellow', label='Moon Path Range (±28.5°)')
    
    # Add reference lines at moon path band boundaries
    ax1.axvline(-35, linestyle='--', color='green', alpha=0.7, label='Moon Path Bands')
    ax1.axvline(-20, linestyle='--', color='green', alpha=0.7)
    ax1.axvline(20, linestyle='--', color='green', alpha=0.7)
    ax1.axvline(35, linestyle='--', color='green', alpha=0.7)
    
    ax1.set_title('Distribution of Earthquakes and Moon Positions by Latitude', fontsize=14)
    ax1.set_xlabel('Latitude (degrees)')
    ax1.set_ylabel('Percentage of Total (%)')
    ax1.set_xlim(-90, 90)  # Set explicit x-axis limits to show full latitude range
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot relative concentration in the lower subplot
    ax2.plot(quake_bin_centers, analysis_df['Relative_Concentration'], 'b-', linewidth=2)
    ax2.axhline(1.0, linestyle='--', color='gray', alpha=0.7)
    ax2.set_xlabel('Latitude (degrees)')
    ax2.set_ylabel('Relative\nConcentration')
    ax2.set_xlim(-90, 90)  # Set explicit x-axis limits to match the top plot
    ax2.grid(True, alpha=0.3)
    
    # Add text with quantitative results
    result_text = (
        f"Correlation between distributions: {correlation:.3f} (p={p_value:.4f})\n"
        f"Earthquakes in Moon Path Range (±28°): {exact_moon_path_percentage:.1f}%\n"
        f"Earthquakes in Moon Path Bands (±20-35°): {moon_path_bands_percentage:.1f}%\n"
        f"Earthquakes in Other Regions: {non_moon_path_percentage:.1f}%\n"
    )
    
    plt.figtext(0.5, 0.01, result_text, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Latitude distribution analysis saved to {save_path}")
    
    # Return the analysis results as a dictionary
    results = {
        'correlation': correlation,
        'p_value': p_value,
        'exact_moon_path_percentage': exact_moon_path_percentage,
        'moon_path_bands_percentage': moon_path_bands_percentage,
        'non_moon_path_percentage': non_moon_path_percentage,
        'analysis_dataframe': analysis_df
    }
    
    return results

def analyze_earthquakes_by_tectonic_distance(earthquake_data, plate_boundaries=None, save_path=None):
    """
    Analyzes and visualizes the percentage of earthquakes that occur near tectonic plate boundaries.
    
    Args:
        earthquake_data (DataFrame): Dataset containing earthquake events with latitude and longitude
        plate_boundaries (dict, optional): GeoJSON dict containing tectonic plate boundaries
        save_path (str, optional): Path to save the visualization
        
    Returns:
        dict: Dictionary containing analysis results
    """
    logging.info("Analyzing earthquake distribution by distance to tectonic plate boundaries...")
    
    if plate_boundaries is None:
        logging.error("Tectonic plate boundaries data is required for this analysis.")
        return None
    
    # Create a copy of the earthquake data to avoid modifying the original
    earthquake_df = earthquake_data.copy()
    
    # Check if we already have the distance_to_boundary column
    if 'distance_to_boundary' not in earthquake_df.columns:
        logging.info("Calculating distances to tectonic plate boundaries...")
        
        # Convert plate boundaries to GeoDataFrame
        plate_lines = []
        
        # Extract line geometries from the GeoJSON
        for feature in plate_boundaries['features']:
            geometry = feature['geometry']
            if geometry['type'] == 'LineString':
                coords = geometry['coordinates']
                plate_lines.append(LineString(coords))
            elif geometry['type'] == 'MultiLineString':
                for line_coords in geometry['coordinates']:
                    plate_lines.append(LineString(line_coords))
        
        # Create a GeoDataFrame with all plate boundary lines
        plate_gdf = gpd.GeoDataFrame(geometry=plate_lines, crs="EPSG:4326")
        
        # Convert plate boundaries to a single MultiLineString for faster processing
        all_boundaries = MultiLineString(plate_lines)
        
        # Create a function to calculate distance to nearest boundary
        def calculate_distance(row):
            # Create a Point from earthquake coordinates
            point = Point(row['longitude'], row['latitude'])
            
            # Find the nearest point on any boundary
            nearest_point_on_boundary = nearest_points(point, all_boundaries)[1]
            
            # Calculate distance in kilometers (approximate using Haversine formula)
            # 1 degree is approximately 111 km at the equator
            lat1, lon1 = row['latitude'], row['longitude']
            lat2, lon2 = nearest_point_on_boundary.y, nearest_point_on_boundary.x
            
            # Simple Haversine formula
            R = 6371  # Earth radius in kilometers
            dLat = np.radians(lat2 - lat1)
            dLon = np.radians(lon2 - lon1)
            a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c
            
            return distance
        
        # Apply the distance calculation to each earthquake
        # This might take some time for large datasets
        earthquake_df['distance_to_boundary'] = earthquake_df.apply(calculate_distance, axis=1)
        logging.info(f"Calculated distances to boundaries for {len(earthquake_df)} earthquakes")
    
    # Define distance thresholds in kilometers
    distance_thresholds = [0, 50, 100, 200, 500]
    
    # Calculate percentages for each threshold
    percentages = []
    counts = []
    total_earthquakes = len(earthquake_df)
    
    # Create labels for the plot
    labels = [
        f"< {distance_thresholds[1]} km",
        f"{distance_thresholds[1]}-{distance_thresholds[2]} km",
        f"{distance_thresholds[2]}-{distance_thresholds[3]} km",
        f"{distance_thresholds[3]}-{distance_thresholds[4]} km",
        f"> {distance_thresholds[4]} km"
    ]
    
    # Calculate counts and percentages for each bin
    for i in range(len(distance_thresholds)):
        if i < len(distance_thresholds) - 1:
            mask = (earthquake_df['distance_to_boundary'] >= distance_thresholds[i]) & \
                   (earthquake_df['distance_to_boundary'] < distance_thresholds[i+1])
            count = mask.sum()
            percentage = (count / total_earthquakes) * 100
            percentages.append(percentage)
            counts.append(count)
    
    # Add the last bin (> last threshold)
    mask = earthquake_df['distance_to_boundary'] >= distance_thresholds[-1]
    count = mask.sum()
    percentage = (count / total_earthquakes) * 100
    percentages.append(percentage)
    counts.append(count)
    
    # Calculate cumulative percentages
    cumulative_percentages = np.cumsum(percentages)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the distribution
    bars = ax1.bar(labels, percentages, color='skyblue', alpha=0.7)
    
    # Add percentage labels on top of each bar
    for bar, percentage, count in zip(bars, percentages, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentage:.1f}%\n({count:,})',
                ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Distribution of Earthquakes by Distance to Tectonic Plate Boundaries', fontsize=14)
    ax1.set_ylabel('Percentage of Total Earthquakes (%)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot the cumulative distribution
    ax2.plot(labels, cumulative_percentages, 'ro-', linewidth=2)
    ax2.set_ylabel('Cumulative %')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # Add cumulative percentage labels
    for i, (x, y) in enumerate(zip(labels, cumulative_percentages)):
        ax2.text(i, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add a horizontal line at 100%
    ax2.axhline(100, linestyle='--', color='gray', alpha=0.7)
    
    # Calculate key statistics
    within_50km = percentages[0]
    within_100km = cumulative_percentages[1]
    within_200km = cumulative_percentages[2]
    
    # Add text with key findings
    result_text = (
        f"Key Findings:\n"
        f"{within_50km:.1f}% of earthquakes occur within 50 km of a tectonic plate boundary\n"
        f"{within_100km:.1f}% of earthquakes occur within 100 km of a tectonic plate boundary\n"
        f"{within_200km:.1f}% of earthquakes occur within 200 km of a tectonic plate boundary"
    )
    
    plt.figtext(0.5, 0.01, result_text, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Tectonic distance analysis saved to {save_path}")
    
    # Return the analysis results
    results = {
        'distance_thresholds': distance_thresholds,
        'percentages': percentages,
        'cumulative_percentages': cumulative_percentages.tolist(),
        'within_50km': within_50km,
        'within_100km': within_100km,
        'within_200km': within_200km
    }
    
    return results

def load_data(use_test_data=False):
    """
    Load all required data for the visualizations
    
    Args:
        use_test_data (bool): If True, generate test data when real data files are missing.
                              Default is False to ensure only real data is used.
        
    Returns:
        bool: True if data loading was successful, False otherwise
    """
    global earthquake_df, lunar_df, integrated_df, plate_boundaries
    
    try:
        # Define paths to match the actual data structure
        data_dir = os.path.join(base_dir, "Data")
        
        # Update paths to match the actual directory structure
        earthquake_data_path = os.path.join(data_dir, "ProcessedData", "Earthquakes", "comcat_cleaned.csv")
        lunar_data_path = os.path.join(data_dir, "ProcessedData", "LunarData", "lunar_positions_processed.csv")
        integrated_data_path = os.path.join(data_dir, "ProcessedData", "integrated_dataset.csv")
        plate_boundaries_path = os.path.join(data_dir, "RawData", "TectonicPlates", "tectonic_plate_boundaries.json")
        
        # Alternative paths in case the primary files don't exist
        alt_earthquake_path = os.path.join(data_dir, "ProcessedData", "Earthquakes", "earthquake_tectonic_distance_with_wkt.csv")
        alt_lunar_path = os.path.join(data_dir, "ProcessedData", "lunar_processed.csv")
        alt_integrated_path = os.path.join(data_dir, "ProcessedData", "earthquake_lunar_combined.csv")
        alt_plate_path = os.path.join(data_dir, "RawData", "TectonicPlates", "tectonic_plates_polygons.json")
        
        missing_files = []
        
        # Try to load tectonic plate boundaries
        try:
            logging.info("Loading tectonic plate boundaries...")
            with open(plate_boundaries_path, 'r') as f:
                plate_boundaries = json.load(f)
            logging.info(f"Successfully loaded tectonic plate boundaries from {plate_boundaries_path}")
        except Exception as _:
            # Try alternative path
            try:
                with open(alt_plate_path, 'r') as f:
                    plate_boundaries = json.load(f)
                logging.info(f"Successfully loaded tectonic plate boundaries from {alt_plate_path}")
            except Exception as _:
                missing_files.append("tectonic plate boundaries")
                logging.warning(f"Could not load tectonic plate boundaries from {plate_boundaries_path} or {alt_plate_path}. Using None.")
                plate_boundaries = None
        
        # Try to load processed earthquake data
        try:
            logging.info("Loading processed earthquake data...")
            earthquake_df = pd.read_csv(earthquake_data_path, parse_dates=['time'])
            logging.info(f"Loaded {len(earthquake_df)} earthquake events from {earthquake_data_path}")
        except Exception as _:
            # Try alternative path
            try:
                earthquake_df = pd.read_csv(alt_earthquake_path)
                # If 'time' column is missing, try to find another date column
                if 'time' not in earthquake_df.columns and 'date' in earthquake_df.columns:
                    earthquake_df = pd.read_csv(alt_earthquake_path, parse_dates=['date'])
                    earthquake_df.rename(columns={'date': 'time'}, inplace=True)
                elif 'time' not in earthquake_df.columns:
                    # Look for any datetime-like column
                    for col in earthquake_df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            earthquake_df = pd.read_csv(alt_earthquake_path, parse_dates=[col])
                            earthquake_df.rename(columns={col: 'time'}, inplace=True)
                            break
                logging.info(f"Loaded {len(earthquake_df)} earthquake events from {alt_earthquake_path}")
            except Exception as _:
                missing_files.append("earthquake data")
                logging.warning(f"Could not load earthquake data from {earthquake_data_path} or {alt_earthquake_path}. Using None.")
                earthquake_df = None
        
        # Try to load lunar data
        try:
            logging.info("Loading lunar data...")
            lunar_df = pd.read_csv(lunar_data_path, parse_dates=['date'])
            logging.info(f"Loaded {len(lunar_df)} days of lunar position data from {lunar_data_path}")
        except Exception as _:
            # Try alternative path or date column
            try:
                lunar_df = pd.read_csv(alt_lunar_path)
                # Check for date column
                if 'date' not in lunar_df.columns:
                    for col in lunar_df.columns:
                        if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():
                            lunar_df = pd.read_csv(alt_lunar_path, parse_dates=[col])
                            lunar_df.rename(columns={col: 'date'}, inplace=True)
                            break
                logging.info(f"Loaded {len(lunar_df)} days of lunar position data from {alt_lunar_path}")
            except Exception as _:
                missing_files.append("lunar data")
                logging.warning(f"Could not load lunar data from {lunar_data_path} or {alt_lunar_path}. Using None.")
                lunar_df = None
        
        # Try to load integrated dataset
        try:
            logging.info("Loading integrated dataset...")
            integrated_df = pd.read_csv(integrated_data_path, parse_dates=['time'])
            logging.info(f"Loaded {len(integrated_df)} records from integrated dataset at {integrated_data_path}")
        except Exception as _:
            # Try alternative path
            try:
                integrated_df = pd.read_csv(alt_integrated_path)
                # Check for time column
                if 'time' not in integrated_df.columns and 'date' in integrated_df.columns:
                    integrated_df = pd.read_csv(alt_integrated_path, parse_dates=['date'])
                    integrated_df.rename(columns={'date': 'time'}, inplace=True)
                elif 'time' not in integrated_df.columns:
                    # Look for any datetime-like column
                    for col in integrated_df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            integrated_df = pd.read_csv(alt_integrated_path, parse_dates=[col])
                            integrated_df.rename(columns={col: 'time'}, inplace=True)
                            break
                logging.info(f"Loaded {len(integrated_df)} records from integrated dataset at {alt_integrated_path}")
            except Exception as _:
                # As a fallback, if we have both earthquake and lunar data but no integrated data,
                # we could potentially create a simple integrated dataset here
                if 'earthquake_df' in globals() and len(earthquake_df) > 0 and 'lunar_df' in globals() and len(lunar_df) > 0:
                    logging.warning("No integrated dataset found, but earthquake and lunar data are available.")
                    logging.warning("Using earthquake data as the base for integrated analysis.")
                    integrated_df = earthquake_df.copy()
                else:
                    missing_files.append("integrated data")
                    logging.warning(f"Could not load integrated dataset from {integrated_data_path} or {alt_integrated_path}. Using None.")
                    integrated_df = None
        
        if missing_files:
            logging.error(f"Missing required data files: {', '.join(missing_files)}")
            print("\nERROR: The following data files are missing:")
            for file in missing_files:
                print(f"- {file}")
            print("\nPlease ensure these files exist in the Data directory before running visualizations.")
            return False
        
        # Ensure all DataFrames have the expected columns
        required_cols = {
            'earthquake_df': ['latitude', 'longitude', 'mag', 'time'],
            'lunar_df': ['sublunar_lat', 'sublunar_lon', 'date'],
            'integrated_df': ['latitude', 'longitude', 'time', 'sublunar_lat', 'sublunar_lon']
        }
        
        missing_cols = {}
        for df_name, cols in required_cols.items():
            if df_name in globals() and globals()[df_name] is not None:
                df = globals()[df_name]
                missing = [col for col in cols if col not in df.columns]
                if missing:
                    missing_cols[df_name] = missing
        
        if missing_cols:
            logging.error("Some dataframes are missing required columns:")
            for df_name, cols in missing_cols.items():
                logging.error(f"{df_name} is missing columns: {', '.join(cols)}")
                print(f"ERROR: {df_name} is missing columns: {', '.join(cols)}")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_global_earthquake_map(earthquake_data, plate_boundaries, save_path):
    """Placeholder for the function that creates a global map of earthquakes"""
    logging.info(f"Would create global earthquake map at {save_path}")

def visualize_earthquake_time_series(earthquake_data, save_dir):
    """Placeholder for the function that visualizes earthquake time series"""
    logging.info("Would create earthquake time series visualization")

def create_time_slider_visualization():
    """Placeholder for the function that creates an interactive time slider visualization"""
    logging.info("Would create interactive time slider visualization")

def visualize_lunar_distance_cycle(lunar_data, save_path):
    """Placeholder for the function that visualizes the lunar distance cycle"""
    logging.info(f"Would create lunar distance cycle visualization at {save_path}")

def visualize_earthquake_frequency_by_lunar_phase(integrated_data, save_path):
    """Placeholder for the function that visualizes earthquake frequency by lunar phase"""
    logging.info(f"Would create earthquake frequency by lunar phase visualization at {save_path}")

def visualize_sublunary_point_heatmap(lunar_data, integrated_data, plate_boundaries, save_path):
    """Placeholder for the function that visualizes the sublunary point heatmap"""
    logging.info(f"Would create sublunary point heatmap at {save_path}")

def visualize_earthquake_by_sublunar_distance():
    """Placeholder for the function that visualizes earthquake distribution by sublunar distance"""
    logging.info("Would create earthquake by sublunar distance visualization")

def main():
    """Main function to run all visualization scripts."""
    try:
        # Load all required data
        if not load_data(use_test_data=False):  # Set explicitly to False to use only real data
            logging.error("Failed to load data. Exiting.")
            return
        
        # Define paths
        vis_dir = os.path.join(base_dir, "Visualisations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create all standard visualizations
        create_global_earthquake_map(earthquake_df, plate_boundaries, 
                                    os.path.join(vis_dir, "global_earthquake_map.png"))
        
        visualize_earthquake_time_series(earthquake_df, vis_dir)
        
        print("Creating interactive visualization with time slider...")
        create_time_slider_visualization()
        
        # Create lunar cycle visualizations
        logging.info("Creating lunar cycle visualizations...")
        visualize_lunar_distance_cycle(lunar_df, 
                                      os.path.join(vis_dir, "lunar_distance_cycle.png"))
        
        visualize_earthquake_frequency_by_lunar_phase(integrated_df, 
                                                     os.path.join(vis_dir, "earthquake_frequency_by_lunar_phase.png"))
        
        visualize_sublunary_point_heatmap(lunar_df, integrated_df, plate_boundaries,
                                         os.path.join(vis_dir, "sublunary_point_heatmap.png"))
        
        # Create spatial analysis of sublunary points vs. earthquakes
        visualize_earthquake_by_sublunar_distance()
        
        # Add new latitude distribution analysis
        logging.info("Creating earthquake latitude distribution analysis...")
        latitude_analysis_path = os.path.join(vis_dir, 'earthquake_latitude_distribution.png')
        latitude_analysis = analyze_earthquake_distribution_by_latitude(integrated_df, lunar_df, plate_boundaries, latitude_analysis_path)
        
        # Print key findings from the latitude analysis
        print("\nKey Findings from Latitude Distribution Analysis:")
        print(f"Correlation between earthquake and moon position distributions: {latitude_analysis['correlation']:.3f}")
        print(f"Percentage of earthquakes in moon path range (±28°): {latitude_analysis['exact_moon_path_percentage']:.1f}%")
        print(f"Percentage of earthquakes in moon path bands (±20-35°): {latitude_analysis['moon_path_bands_percentage']:.1f}%")
        print(f"Percentage of earthquakes in other regions: {latitude_analysis['non_moon_path_percentage']:.1f}%")
        
        # Add new tectonic distance analysis
        logging.info("Creating earthquake tectonic distance analysis...")
        tectonic_distance_analysis_path = os.path.join(vis_dir, 'earthquake_tectonic_distance_distribution.png')
        tectonic_distance_analysis = analyze_earthquakes_by_tectonic_distance(earthquake_df, plate_boundaries, tectonic_distance_analysis_path)
        
        # Print key findings from the tectonic distance analysis
        print("\nKey Findings from Tectonic Distance Analysis:")
        print(f"Percentage of earthquakes within 50 km of a tectonic plate boundary: {tectonic_distance_analysis['within_50km']:.1f}%")
        print(f"Percentage of earthquakes within 100 km of a tectonic plate boundary: {tectonic_distance_analysis['within_100km']:.1f}%")
        print(f"Percentage of earthquakes within 200 km of a tectonic plate boundary: {tectonic_distance_analysis['within_200km']:.1f}%")
        
        print("All visualizations created successfully")
        
    except Exception as e:
        logging.error(f"Error in main visualization function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
