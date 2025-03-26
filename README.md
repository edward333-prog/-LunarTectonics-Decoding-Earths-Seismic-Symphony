# -LunarTectonics-Decoding-Earths-Seismic-Symphony
Data Analytics and Data Science 

# LunarTectonics: Decoding Earth's Seismic Symphony

> "What if the most powerful force in earthquake prediction isn't buried deep within Earth's crust, but has been hanging in our night sky all along? For centuries, we've gazed at the Moon while standing on shifting ground—never realizing these two phenomena might be speaking the same cosmic language."

## Project Objective
This data analytics portfolio project aims to develop machine learning models to predict future earthquakes by analyzing a 20-year historical seismic dataset (2005-2025). The analysis explores relationships between earthquakes, tectonic plate boundaries, and lunar cycles. By demonstrating advanced data analysis techniques on real-world geophysical data, this project showcases skills in time series analysis, spatial data processing, and predictive modeling.

## Core Research Questions

This project focuses on three fundamental questions:

1. **Does the Moon influence earthquake occurrence?** 
   - Investigating whether lunar cycles, phases, and tidal forces act as triggers for seismic events
   - Quantifying any statistical correlations between lunar positions and earthquake frequency/magnitude

2. **What role do tectonic plate boundaries play in earthquake patterns?**
   - Analyzing the spatial distribution of earthquakes relative to plate boundaries
   - Examining how distance from plate boundaries affects earthquake frequency and magnitude

3. **Can machine learning models effectively predict future earthquakes?**
   - Developing predictive models that integrate multiple variables including tectonic and lunar factors
   - Evaluating the accuracy, reliability, and practical applications of these predictions

## Primary Data Sources
- **USGS Earthquake Catalog**: Comprehensive historical earthquake data including magnitude, depth, location, and time (20-year period)
- **Global Tectonic Plate Boundaries**: Vector data of major and minor plate boundaries (Bird 2003 dataset)
- **Lunar Position Data**: Moon phases and positions calculated for the entire 20-year period

## Methodology

### Data Collection and Preprocessing
- Download and integrate global earthquake data from USGS/ComCat APIs (2005-2025)
- Clean and normalize data to account for varying measurement techniques and standards
- Integrate tectonic plate boundary information for spatial analysis
- Calculate lunar positions and phases for the entire study period
- Address missing values through appropriate imputation techniques

### 20-Year Timeframe Rationale
- Provides statistical power through larger sample size (millions of earthquake events)
- Captures complete lunar cycles, including the 18.6-year nodal cycle
- Includes rare major seismic events (M7.0+) for comprehensive analysis
- Enables detection of long-term patterns and relationships
- Allows for robust time series model development and validation

### Feature Engineering
- Create distance-based features measuring proximity to tectonic plate boundaries
- Calculate temporal features capturing lunar phases and Earth-Moon distance
- Derive statistical measures of seismic activity in regions over different time scales
- Generate features based on historical earthquake patterns in specific regions

### Exploratory Data Analysis
- Analyze the spatial distribution of earthquakes in relation to tectonic plate boundaries
- Examine temporal patterns including possible lunar correlations
- Investigate magnitude-frequency relationships across different regions
- Create visualizations highlighting potential correlations and patterns

### Machine Learning Model Development
- Implement time-series forecasting approaches for temporal prediction
- Develop classification models for earthquake likelihood in specific regions
- Create regression models for magnitude estimation
- Apply ensemble methods to improve overall prediction accuracy
- Compare model performance with and without lunar cycle variables

### Model Evaluation
- Use appropriate metrics for imbalanced data (precision, recall, F1-score)
- Implement cross-validation strategies specific to time-series data
- Compare model performance against established baseline methods
- Assess practical utility of predictions

## Technologies Used
- **Python**: Primary programming language
- **Pandas/NumPy**: Data manipulation and numerical analysis
- **Scikit-learn/TensorFlow**: Machine learning implementation
- **GeoPandas**: Geographical data processing
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Skyfield**: Astronomical calculations for lunar positions

## Expected Outcomes
- Interactive visualizations demonstrating the relationship between earthquakes, plate boundaries, and lunar cycles
- Statistical analysis of the correlation between lunar positions and seismic activity
- Machine learning models for earthquake prediction with performance metrics
- Comprehensive GitHub repository showcasing data science workflow
- Portfolio-ready project demonstrating skills in handling large geospatial datasets

## Project Status
For current project status, progress updates, and development roadmap, please see [PROJECT_STATUS.md](PROJECT_STATUS.md)
