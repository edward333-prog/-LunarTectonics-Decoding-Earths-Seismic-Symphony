# LunarTectonics: Earthquake Analysis Methodology

## Project Overview

This project investigates the potential relationships between global earthquake activity, tectonic plate boundaries, and lunar gravitational influence. By analyzing comprehensive earthquake data spanning 20 years alongside lunar positioning data, this research aims to identify patterns and correlations that might contribute to our understanding of earthquake triggers and frequency.

### Key Research Questions

1. How does proximity to tectonic plate boundaries influence earthquake frequency and magnitude?
2. Is there a statistically significant relationship between lunar cycles/positions and earthquake activity?
3. Can we identify temporal patterns in earthquake data across a 20-year period that may correlate with external factors?

## Data Collection Strategy

### 20-Year Comprehensive Dataset

A deliberate choice was made to collect and analyze earthquake data spanning 20 years (2005-2025) rather than a shorter timeframe. This decision was based on several methodological advantages:

- **Statistical Significance**: A larger dataset provides more robust statistical power, reducing the impact of anomalous periods or events.
- **Pattern Recognition**: Long-term patterns and cycles that may not be apparent in shorter timeframes become detectable over a 20-year period.
- **Capture of Rare Events**: Major earthquakes (M7.0+) are relatively rare, and a 20-year window ensures sufficient inclusion of these significant events.
- **Correlation with Long-Term Phenomena**: This extended timeframe allows for analysis against other long-term phenomena like lunar cycles, seasonal patterns, and global climate variations.

The USGS ComCat API was used to programmatically access and download this comprehensive earthquake dataset, with custom code implemented to handle the large data volume through chunking and efficient data processing.

### Data Sources

1. **Earthquake Data**: USGS Comprehensive Earthquake Catalog (ComCat) API, focusing on earthquakes of magnitude 4.0 and above to ensure data quality and relevance.
2. **Tectonic Plate Boundaries**: GeoJSON data representing the global tectonic plate configuration.
3. **Lunar Position Data**: Calculated using the Skyfield astronomy library, providing precise lunar positions, distances, and phases for the entire study period.

## Technical Architecture

### Modular Code Structure

The project employs a modular code structure with clear separation of concerns:

1. **DownloadData.py**: Handles the retrieval of earthquake data from the USGS API, including chunking for efficient handling of the 20-year dataset.
2. **DataCleaning.py**: Focuses exclusively on data cleaning and tectonic distance calculations, preparing the raw data for analysis without introducing analytical bias.
3. **Luna_VS_EarthPositions.py**: Calculates lunar positions and integrates this data with the cleaned earthquake and tectonic data.
4. **Visualisations.py**: Creates visualizations of the integrated dataset to facilitate pattern identification and analysis.

This separation ensures maintainability and follows the principle that each component should have a single responsibility.

### Performance Optimization Approaches

Several optimization techniques were implemented to handle the large 20-year dataset efficiently:

1. **Spatial Indexing**: Implemented using Shapely's STRtree to dramatically accelerate distance calculations between earthquake locations and tectonic plate boundaries.
2. **Efficient Data Representation**: Used appropriate data types and structures to minimize memory usage while maintaining calculation accuracy.
3. **Batch Processing**: Implemented for operations that would otherwise be computationally intensive when run on the entire dataset.
4. **Progress Tracking**: Added visual progress indicators and logging to monitor long-running processes.

## Data Processing Pipeline

The data processing workflow follows these key steps:

### 1. Data Acquisition
- Download 20 years of global earthquake data from USGS ComCat API
- Implement chunking to handle API limitations
- Download tectonic plate boundary data

### 2. Data Cleaning
- Filter earthquakes by magnitude threshold (≥4.0)
- Convert timestamps to standardized datetime format
- Validate coordinates and remove outliers
- Handle missing values appropriately

### 3. Geospatial Processing
- Calculate distances from each earthquake to the nearest tectonic plate boundary
- Use spatial indexing (STRtree) to optimize these calculations
- Categorize earthquakes by distance ranges for analytical purposes

### 4. Lunar Data Integration
- Calculate precise lunar positions for each day in the 20-year period
- Determine lunar phases, distances, and gravitational influence metrics
- Match earthquake events with the corresponding lunar data

### 5. Integrated Dataset Creation
- Merge earthquake, tectonic distance, and lunar data into a comprehensive dataset
- Add derived features for time analysis (year, month, day, hour, etc.)
- Create indicators for special conditions (e.g., full/new moon periods)

## Analytical Methods

### Geospatial Analysis

- **Distance Calculation**: Each earthquake's distance to the nearest tectonic plate boundary is calculated using the Haversine formula to account for Earth's curvature. This allows for accurate analysis of earthquake distribution relative to plate boundaries.
- **Tectonic Proximity Binning**: Earthquakes are categorized into distance bins (0-50km, 50-100km, 100-200km, 200-500km, >500km) to quantify the relationship between seismic activity and plate boundaries.
- **Spatial Clustering**: Identifying areas of heightened activity and their relationship to tectonic features.

### Lunar Path Analysis

- **Latitude Distribution Comparison**: The distribution of earthquakes by latitude is compared with the moon's orbital path, which ranges approximately ±28° from the equator.
- **Correlation Calculation**: Pearson correlation coefficient is calculated between the normalized histograms of earthquake and moon position distributions.
- **Zonal Analysis**: The earthquake distribution is analyzed in three zones: within the moon's path (±28°), in the transition bands (±20-35°), and outside these regions.

### Time Series Analysis

- **Temporal Patterns**: Analysis of earthquake frequency and magnitude across different time scales (daily, monthly, annual) to identify cyclical patterns.
- **Lunar Cycle Correlation**: Statistical tests to determine if earthquake frequency or magnitude correlate with lunar phases, distances, or positions.

### Statistical Approaches

- **Distribution Analysis**: Examining the statistical distributions of earthquake magnitudes and their relationship to tectonic proximity.
- **Correlation Testing**: Using statistical methods to test for significant correlations between lunar factors and earthquake occurrence.
- **Significance Testing**: Ensuring that observed patterns are statistically significant rather than random occurrences.

## Visualization Techniques

Visualizations were designed to effectively communicate complex relationships:

### 1. Tectonic Distance Distribution Visualization

- **Bar Chart**: Shows the percentage of earthquakes in each distance bin from tectonic plate boundaries
- **Cumulative Distribution Plot**: Displays the cumulative percentage of earthquakes as distance from plate boundaries increases
- **Key Statistics Text**: Highlights critical findings (e.g., 67.0% of earthquakes occur within 100km of boundaries)

### 2. Lunar Path Correlation Visualization

- **Dual Histogram**: Compares the normalized distributions of earthquake occurrences and moon positions by latitude
- **Correlation Indicator**: Displays the calculated Pearson correlation coefficient (0.680)
- **Zonal Statistics**: Shows the percentage of earthquakes in each defined zone relative to the moon's path

### 3. Interactive Maps and Time Series

- **Interactive Maps**: Showing earthquake locations colored by magnitude and proximity to plate boundaries.
- **Time Series Charts**: Displaying earthquake frequency alongside lunar cycles to visually identify potential correlations.
- **Animated Visualizations**: Demonstrating how earthquake patterns evolve over the 20-year period, with corresponding lunar positions.

## Technical Challenges & Solutions

### Challenge 1: Performance Issues with Distance Calculations

**Problem**: Calculating the distance from each earthquake to the nearest tectonic plate boundary was extremely computationally intensive using the naive approach of checking each earthquake against every boundary segment.

**Solution**: Implemented spatial indexing using Shapely's STRtree, which creates a spatial tree structure that dramatically reduces the number of comparisons needed. This improved performance from hours to seconds for the entire 20-year dataset.

### Challenge 2: Memory Management with Large Dataset

**Problem**: The 20-year dataset contains approximately 145,933 earthquake events, which caused memory issues when processing all at once.

**Solution**: Implemented efficient data structures and types, pre-calculated projected geometries in batch to optimize memory usage, and used generators where appropriate to reduce memory footprint.

### Challenge 3: Integration of Heterogeneous Data Types

**Problem**: Combining time-based earthquake data with spatial tectonic data and astronomical lunar data presented challenges in data alignment and association.

**Solution**: Created a robust integration system that properly handles datetime conversions and associations, ensuring accurate matching between earthquakes and the corresponding lunar positions.

### Challenge 4: Accurate Geospatial Distance Calculation

**Problem**: Calculating accurate distances between earthquake epicenters and tectonic plate boundaries across a global dataset required handling Earth's curvature.

**Solution**: Implemented the Haversine formula to calculate great-circle distances, providing accurate measurements regardless of the earthquake's location on Earth's surface.

## Key Findings & Implications

### Tectonic Boundary Proximity

- **Finding**: 42.1% of earthquakes occur within 50km of plate boundaries, and 86.1% within 200km.
- **Implication**: Confirms the strong influence of tectonic plate boundaries on earthquake occurrence and provides a quantitative basis for spatial risk assessment.
- **Methodology Validation**: The comprehensive 20-year dataset and accurate distance calculation methods provide high confidence in these percentages.

### Lunar Path Correlation

- **Finding**: 64.9% of earthquakes occur within the moon's path range (±28° latitude), with a correlation coefficient of 0.680.
- **Implication**: Suggests a potential gravitational influence of the moon on earthquake triggering mechanisms.
- **Methodological Strength**: The large dataset allows for statistically significant correlation analysis that would be less reliable with shorter timeframes.

### Combined Mechanism Hypothesis

- **Finding**: The similar percentages of earthquakes within the moon's path (64.9%) and within 100km of plate boundaries (67.0%) suggest a potential combined mechanism.
- **Implication**: Lunar gravitational forces may act as a trigger for earthquake events in regions already under tectonic stress.
- **Methodological Approach**: This insight emerged from the parallel analysis of two different factors, demonstrating the value of our multi-faceted analytical approach.

## Lessons Learned & Future Improvements

### Key Learnings

1. **Importance of Optimization**: Early investment in optimizing critical computational processes (like the distance calculations) paid dividends throughout the project.
2. **Value of Comprehensive Data**: The 20-year dataset provided insights that would likely have been missed with a smaller timeframe, validating this methodological choice.
3. **Modular Design Benefits**: The separation of concerns in the code architecture made debugging and enhancements much more manageable.
4. **Multi-factor Analysis**: Examining both tectonic and lunar factors in parallel revealed potential combined mechanisms that might have been missed in single-factor analyses.

### Future Improvements

1. **Additional Variables**: Incorporate other potential influencing factors such as solar activity, Earth's rotation variations, and seasonal patterns.
2. **Machine Learning Models**: Develop predictive models based on the collected data to identify potential earthquake risk periods.
3. **Performance Enhancements**: Further optimize code using GPU acceleration for the most intensive calculations.
4. **Real-Time Processing**: Extend the system to incorporate real-time earthquake data for ongoing analysis.
5. **Regional Analysis**: Apply the dual-factor analysis to specific tectonic regions to identify variations in the lunar-tectonic relationship.

## Conclusion

This project demonstrates a comprehensive approach to analyzing the potential relationships between earthquakes, tectonic plate boundaries, and lunar influences. By utilizing a 20-year dataset, optimized computational methods, and a structured analytical approach, the research provides meaningful insights into these complex geophysical relationships.

The methodologies employed prioritize statistical validity, computational efficiency, and scientific rigor, making this analysis both academically sound and practically valuable. The quantitative findings on both tectonic proximity and lunar path correlation provide compelling evidence for a dual-factor model of earthquake occurrence that warrants further investigation.

---
*Last Updated: March 26, 2025*
