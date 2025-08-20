# MA-building-stock

# Massachusetts Building Analysis Dashboard

## Overview

An interactive web-based dashboard for analyzing building inventory data from the NSI(National Structural Inventory Dataset) enhanced USA Structures Dataset for Massachusetts. This dashboard can provide comprehensive visualization and analysis tools for exploring building patterns, temporal distributions, and material characteristics across different time periods.

## Features

### 8 Interactive Analysis Sections

1. **Overview**
   - Total building statistics
   - Building distribution by occupancy class
   - Construction timeline visualization
   - Export capabilities for charts and data

2. **Clustering Analysis**
   - K-means clustering results (K=5 to K=9)
   - Elbow method for optimal K selection
   - Interactive scatter plots and treemaps
   - Cluster statistics table

3. **Temporal Distribution**
   - Building construction patterns over time
   - Multiple visualization options (Stacked Area, Line Plot, Normalized, Cumulative)
   - Building type filters (All, Residential, Non-Residential)
   - Floor area analysis by year

4. **Pre-1940 Buildings**
   - Historic building analysis
   - Distribution by occupancy class
   - Residential vs Non-Residential comparison
   - Total floor area statistics

5. **Post-1940 Buildings**
   - Modern construction patterns (1940-present)
   - Annual construction data
   - Decade-by-decade comparison
   - Normalized stacking option

6. **Occupancy-Specific Clustering**
   - Detailed clustering by occupancy class
   - Balanced vs Random sampling options
   - Sample size selection (1,000 to 20,000 buildings)
   - Elbow method visualization per occupancy

7. **Materials & Foundation Analysis**
   - Material type vs Foundation type correlation heatmap
   - Interactive occupancy breakdown (click cells for details)
   - Material usage trends over time
   - Pre-1940 vs Post-1940 comparison

8. **Interactive Explorer**
   - Custom data filtering by year and area
   - 6 visualization types:
     - Box Plot
     - 3D Scatter Plot
     - Sunburst Chart
     - Parallel Coordinates
     - Violin Plot
     - Treemap

## Getting Started!

### Prerequisites

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Local web server (optional, but recommended for better performance)
- `building_data.json` file in the same directory as the HTML file

### Installation

1. Download the HTML file (`index.html`)
2. Place your `building_data.json` file in the same directory
3. Open the HTML file in a web browser

### Using a Local Server (Recommended)

```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Node.js (if http-server is installed)
http-server
```

Then navigate to `http://localhost:8000` in your browser.

## 📁 Data Requirements

### Data File Structure

The dashboard expects a `building_data.json` file with the following structure:

```json
{
  "metadata": {
    "total_buildings": 2500000,
    "date_processed": "2024-01-01T00:00:00Z",
    "source_file": "ma_buildings.csv"
  },
  "summary_stats": {
    "total_buildings": 2500000,
    "avg_year_built": 1978,
    "avg_area_sqm": 285.5,
    "occupancy_classes": ["Residential", "Commercial", ...]
  },
  "temporal_data": [
    {
      "year": 1900,
      "occupancy": "Residential",
      "count": 1000,
      "avg_area": 250,
      "total_area": 250000
    }
  ],
  "clustering": {
    "elbow_k_values": [2, 3, 4, ...],
    "elbow_wcss_values": [50000, 35000, ...],
    "clusters": [...]
  },
  "pre1940": {...},
  "post1940": {...},
  "materials_foundation": {...},
  "building_samples_random": [...],
  "building_samples_balanced": [...]
}
```

### Occupancy Classes

- Residential
- Commercial
- Industrial
- Agriculture
- Government
- Assembly
- Education
- Utility and Misc
- Unclassified

### Material Types

- **M**: Masonry
- **W**: Wood
- **H**: Manufactured
- **S**: Steel
- **C**: Concrete

### Foundation Types

- **C**: Crawl Space
- **B**: Basement
- **S**: Slab
- **P**: Pier
- **I**: Pile
- **F**: Fill
- **W**: Solid Wall

## Configuration

### Customization Options

The dashboard includes several configurable elements:

- **Cluster Count**: Adjustable from 5 to 9 clusters
- **Sample Size**: Choose from 1,000 to 20,000 buildings for visualization
- **Chart Types**: Multiple visualization options per section
- **Data Filters**: Year range, area range, building type filters
- **Export Options**: PNG export for individual charts, JSON export for all data


## Troubleshooting

### Common Issues

1. **"Error Loading Data" Message**
   - Ensure `building_data.json` is in the same directory
   - Check browser console for specific error messages
   - Verify JSON file format is valid

2. **Charts Not Displaying**
   - Check internet connection (Plotly.js loads from CDN)
   - Clear browser cache and reload
   - Ensure JavaScript is enabled

3. **Slow Performance**
   - Reduce sample size in Occupancy and Interactive Explorer sections
   - Use a modern browser with hardware acceleration
   - Close unnecessary browser tabs

## Data Quality Notes

- Some years may have incomplete data (2006, 2009-2011, 2013-2016)
- Pre-1940 buildings are aggregated for temporal analysis
- Sample data is used for visualization performance
- Full dataset statistics are calculated separately from samples

## Updates and Maintenance

### Version History

- **v1.0.0** (Current): Initial release with 8 analysis sections


## Support

For issues, questions, or suggestions regarding this dashboard, please contact: shao.la@northeastern.edu

---

*Last Updated: January 2025*
*Dashboard Version: 1.0.0*
