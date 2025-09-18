"""
Massachusetts Building Data Processor - Enhanced Version with Multi-dimensional Clustering and Soil Analysis
This script processes the building data and exports it to JSON for the updated web dashboard
Now includes pre-computed clustering results for different feature combinations and soil analysis
Split file version to handle GitHub 25MB limit
Fixed: Proper handling of NaN values for JSON export
Enhanced: Added compname analysis and data flow statistics
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def format_large_number(num, is_area=False):
    """Turn big numbers to readable numbers"""
    if num >= 1000000:
        return f"{num / 1000000:.2f}M"
    if num >= 1000:
        return f"{num / 1000:.2f}K" if is_area else f"{num / 1000:.1f}K"
    return str(round(num)) if is_area else str(num)

class BuildingDataProcessor:
    def __init__(self, csv_path='ma_structures_with_demolition_FINAL.csv'):
        """Initialize the processor with data path"""
        self.csv_path = csv_path
        self.df = None
        self.df_cleaned = None
        self.df_cluster = None
        self.preprocessor = None
        self.kmeans = None
        self.data_flow_stats = {}  # Track data flow statistics

    def load_data(self):
        """Load the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} records")

        # Track initial data stats
        self.data_flow_stats['initial_count'] = len(self.df)
        self.data_flow_stats['initial_columns'] = list(self.df.columns)

        return self

    def clean_data(self):
        """Clean the data with detailed tracking"""
        print("Cleaning data...")

        # Initialize detailed cleaning statistics
        cleaning_stats = {
            'initial_count': len(self.df),
            'initial_columns': list(self.df.columns)
        }

        # Step 1: Track invalid year_built (includes <= 0 and NaN)
        # Note: self.df['year_built'] > 0 returns False for both <= 0 and NaN values
        valid_year_mask = self.df['year_built'] > 0
        cleaning_stats['invalid_year_count'] = (~valid_year_mask).sum()
        cleaning_stats['invalid_year_details'] = {
            'negative_or_zero': (self.df['year_built'] <= 0).sum(),
            'nan_values': self.df['year_built'].isna().sum()
        }

        # Apply year filter using original logic (removes both invalid and NaN)
        self.df_cleaned = self.df[self.df['year_built'] > 0].copy()
        cleaning_stats['after_year_filter'] = len(self.df_cleaned)

        # Step 2: Track and remove missing area
        missing_area_mask = self.df_cleaned['Est GFA sqmeters'].isna()
        cleaning_stats['missing_area_count'] = missing_area_mask.sum()

        # Remove rows with missing area
        if cleaning_stats['missing_area_count'] > 0:
            self.df_cleaned = self.df_cleaned[~missing_area_mask].copy()
        cleaning_stats['after_missing_area'] = len(self.df_cleaned)

        # Step 3: Track and remove missing OCC_CLS
        missing_occ_mask = self.df_cleaned['OCC_CLS'].isna()
        cleaning_stats['missing_occ_count'] = missing_occ_mask.sum()

        # Remove rows with missing occupancy class
        if cleaning_stats['missing_occ_count'] > 0:
            self.df_cleaned = self.df_cleaned[~missing_occ_mask].copy()
        cleaning_stats['after_missing_occ'] = len(self.df_cleaned)

        # Step 4: Track and remove area outliers (optional step)
        cleaning_stats['area_outliers_count'] = 0
        cleaning_stats['area_outlier_threshold'] = None

        # Check if we have valid area data to calculate outliers
        if 'Est GFA sqmeters' in self.df_cleaned.columns and len(self.df_cleaned) > 0:
            # Calculate the 99.999th percentile for outlier detection
            area_threshold = self.df_cleaned['Est GFA sqmeters'].quantile(0.99999)
            outlier_mask = self.df_cleaned['Est GFA sqmeters'] > area_threshold
            cleaning_stats['area_outliers_count'] = outlier_mask.sum()
            cleaning_stats['area_outlier_threshold'] = float(area_threshold)

            # COMMENTED OUT: Don't remove outliers anymore
            # if cleaning_stats['area_outliers_count'] > 0:
            #     self.df_cleaned = self.df_cleaned[~outlier_mask].copy()

        cleaning_stats['after_outlier_removal'] = len(self.df_cleaned)  # This will be same as after_missing_occ

        # Calculate final statistics
        cleaning_stats['final_count'] = len(self.df_cleaned)
        cleaning_stats['total_removed'] = cleaning_stats['initial_count'] - cleaning_stats['final_count']
        cleaning_stats['removal_percentage'] = round(
            (cleaning_stats['total_removed'] / cleaning_stats['initial_count']) * 100, 2
        ) if cleaning_stats['initial_count'] > 0 else 0

        # Store cleaning statistics in data flow stats
        self.data_flow_stats['cleaning_pipeline'] = cleaning_stats

        if 'material_type' not in self.df_cleaned.columns:
            print("Warning: 'material_type' column not found. Filling with None.")
            self.df_cleaned['material_type'] = None

        if 'foundation_type' not in self.df_cleaned.columns:
            print("Warning: 'foundation_type' column not found. Filling with None.")
            self.df_cleaned['foundation_type'] = None

        # Print summary of cleaning process
        print(f"Cleaned data: {len(self.df_cleaned)} records")
        print(f"  Removed {cleaning_stats['invalid_year_count']} records with invalid year")
        print(f"    - Invalid/zero: {cleaning_stats['invalid_year_details']['negative_or_zero']}")
        print(f"    - NaN values: {cleaning_stats['invalid_year_details']['nan_values']}")
        print(f"  Removed {cleaning_stats['missing_area_count']} records with missing area")
        print(f"  Removed {cleaning_stats['missing_occ_count']} records with missing occupancy")
        print(f"  Removed {cleaning_stats['area_outliers_count']} area outliers")
        if cleaning_stats['area_outlier_threshold']:
            print(f"    - Outlier threshold: {cleaning_stats['area_outlier_threshold']:,.2f} sqm")
        print(f"  Total removed: {cleaning_stats['total_removed']} ({cleaning_stats['removal_percentage']}%)")

        # Store the cleaning stats for later use
        self.data_flow_stats['cleaning_stats'] = cleaning_stats

        return self

    def prepare_clustering_data(self, remove_outliers=False):
        """Prepare data for clustering"""
        print("Preparing clustering data...")

        # Choose features and drop NaN
        features = ['OCC_CLS', 'Est GFA sqmeters', 'year_built', 'material_type', 'foundation_type']
        self.df_cluster = self.df_cleaned[features].dropna().copy()

        if remove_outliers:
            # Calculate the 99.999th percentile for building area
            area_threshold = self.df_cluster['Est GFA sqmeters'].quantile(0.99999)
            print(f"Area threshold for outliers: {area_threshold:,.2f} sqm")

            # Filter out the outliers
            initial_count = len(self.df_cluster)
            self.df_cluster = self.df_cluster[self.df_cluster['Est GFA sqmeters'] < area_threshold].copy()
            outliers_removed = initial_count - len(self.df_cluster)
            print(f"Records after removing outliers: {len(self.df_cluster)}")

            self.data_flow_stats['outliers_removed'] = outliers_removed

        return self

    def process_hierarchical_distribution(self):
        """
        Processes hierarchical data for Sankey diagrams.
        It defines global bins for area, height, and year to ensure consistency,
        then generates the data structure for both the "all buildings" view and
        for each individual occupancy class.
        """
        print("Processing hierarchical distribution with shared binning...")

        df_work = self.df_cleaned.copy()

        # ISSUE #3 FIX: Use original drainage classes instead of grouping them.
        # We use .fillna() to handle missing values, preserving all other categories.
        if 'drainagecl' in df_work.columns:
            df_work['drainage_cat'] = df_work['drainagecl'].fillna('Unknown Drainage')


        # Ensure it's treated as a categorical column.
        df_work['drainage_cat'] = df_work['drainage_cat'].astype('category')

        # Define consistent, global bins for all data to use.
        area_percentiles = df_work['Est GFA sqmeters'].quantile([0.33, 0.67]).values
        area_bins = [0, area_percentiles[0], area_percentiles[1], float('inf')]
        area_labels = ['Small', 'Medium', 'Large']

        height_percentiles = df_work['PRED_HEIGHT'].quantile([0.33, 0.67]).values
        height_bins = [0, height_percentiles[0], height_percentiles[1], float('inf')]
        height_labels = ['Low', 'Mid', 'High']

        year_bins = [0, 1940, 1980, float('inf')]
        year_labels = ['Historic (<1940)', 'Mid-Century (40-80)', 'Modern (>1980)']

        drainage_labels = df_work['drainage_cat'].cat.categories.tolist()

        # Apply these bins to the entire working dataframe to create categorical columns.
        df_work['area_cat'] = pd.cut(df_work['Est GFA sqmeters'], bins=area_bins, labels=area_labels, right=False)
        df_work['height_cat'] = pd.cut(df_work['PRED_HEIGHT'], bins=height_bins, labels=height_labels, right=False)
        df_work['year_cat'] = pd.cut(df_work['year_built'], bins=year_bins, labels=year_labels, right=False)

        hierarchical_by_occupancy = {}

        # ISSUE #1 & #2 FIX: Call the NEW rewritten function for "All Buildings".
        # We also pass the bin definitions to be stored in the JSON for the frontend annotation.
        hierarchical_by_occupancy['all'] = self._process_all_buildings_hierarchy(
            df_work, area_bins, height_bins, year_bins,
            area_labels, height_labels, year_labels, drainage_labels
        )

        # Process the hierarchical structure for each individual occupancy class.
        occupancy_classes = df_work['OCC_CLS'].unique()
        for occ_class in occupancy_classes:
            occ_data = df_work[df_work['OCC_CLS'] == occ_class]
            if len(occ_data) > 100:
                hierarchical_by_occupancy[occ_class] = self._process_single_occupancy_hierarchy(
                    occ_data, occ_class, area_bins, height_bins, year_bins,
                    area_labels, height_labels, year_labels, drainage_labels
                )

        print(f"  Processed hierarchical data for {len(hierarchical_by_occupancy)} occupancy classes")
        return hierarchical_by_occupancy

    # In data_preprocessor.py, find and replace ONLY this one function.

    # In data_preprocessor.py, find and replace ONLY this one function.

    # In data_preprocessor.py, find and replace ONLY this one function.

    # In data_preprocessor.py, find and replace ONLY this one function.

    def _process_all_buildings_hierarchy(self, df_all, area_bins, height_bins, year_bins,
                                         area_labels, height_labels, year_labels, drainage_labels):
        """
        Processes hierarchical data for all buildings, ensuring nodes are arranged in a logical order.

        This version assigns specific vertical positions to nodes within each level to create a
        stable and visually intuitive Sankey diagram layout.

        Returns:
            dict: A dictionary containing 'nodes', 'links', 'total_buildings', and 'node_positions'
                  formatted for the Sankey diagram.
        """
        sankey_data = {
            'nodes': [],
            'links': [],
            'total_buildings': len(df_all),
            'node_positions': {}  # New: Store node position information.
        }
        MIN_COUNT = 0
        nodes = {}

        def add_node(name, level, position=None):
            """
            Helper function to add a new node if it doesn't exist.

            Args:
                name (str): The unique name of the node.
                level (int): The horizontal level (column) of the node in the Sankey diagram.
                position (float, optional): The normalized vertical position (0.0 to 1.0) of the node.
            """
            if name not in nodes:
                nodes[name] = {
                    'name': name,
                    'display_name': name,
                    'level': level,
                    'position': position  # Add position information to the node.
                }

        # Root node
        root_name = 'All Buildings'
        add_node(root_name, 0, 0.5)  # Centered in the first column

        # Occupancy categories - Sorted by count
        occ_counts = df_all['OCC_CLS'].value_counts()
        top_9_occ = occ_counts.nlargest(9).index.tolist()
        df_all['occ_cat'] = df_all['OCC_CLS'].apply(lambda x: x if x in top_9_occ else 'Other')

        # Position occupancy nodes evenly based on their rank by count.
        all_occ_cats = df_all['occ_cat'].value_counts().index.tolist()
        for i, label in enumerate(all_occ_cats):
            position = (i + 1) / (len(all_occ_cats) + 1)  # Distributes nodes evenly
            add_node(label, 1, position)

        # Area - Fixed order: Small -> Medium -> Large
        area_positions = {'Small': 0.25, 'Medium': 0.5, 'Large': 0.75}
        for label in area_labels:
            add_node(label, 2, area_positions.get(label, 0.5))  # Default to center

        # Height - Fixed order: Low -> Mid -> High
        height_positions = {'Low': 0.25, 'Mid': 0.5, 'High': 0.75}
        for label in height_labels:
            add_node(label, 3, height_positions.get(label, 0.5))

        # Year - Fixed order: Old -> Mid -> New
        year_positions = {
            'Historic (<1940)': 0.25,
            'Mid-Century (40-80)': 0.5,
            'Modern (>1980)': 0.75
        }
        for label in year_labels:
            add_node(label, 4, year_positions.get(label, 0.5))

        # Drainage - Fixed order: Good -> Poor
        drainage_positions = {
            'Excessively drained': 0.1,
            'Well drained': 0.2,
            'Moderately well drained': 0.3,
            'Somewhat excessively drained': 0.4,
            'Somewhat poorly drained': 0.5,
            'Poorly drained': 0.7,
            'Very poorly drained': 0.8,
            'Unknown Drainage': 0.9
        }

        # Filter out drainage labels that might have overlapping names with occupancy categories.
        valid_drainage_labels = [label for label in drainage_labels
                                 if label not in all_occ_cats]
        for label in valid_drainage_labels:
            add_node(label, 5, drainage_positions.get(label, 0.5))

        # Create links (maintaining original logic)
        # This section calculates the flow values between nodes at consecutive levels.
        level1_counts = df_all.groupby('occ_cat').size().reset_index(name='count')
        for _, row in level1_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': root_name,
                    'target': row['occ_cat'],
                    'value': row['count']
                })

        level2_counts = df_all.groupby(['occ_cat', 'area_cat'], observed=True).size().reset_index(name='count')
        for _, row in level2_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': str(row['occ_cat']),
                    'target': str(row['area_cat']),
                    'value': row['count']
                })

        level3_counts = df_all.groupby(['area_cat', 'height_cat'], observed=True).size().reset_index(name='count')
        for _, row in level3_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': str(row['area_cat']),
                    'target': str(row['height_cat']),
                    'value': row['count']
                })

        level4_counts = df_all.groupby(['height_cat', 'year_cat'], observed=True).size().reset_index(name='count')
        for _, row in level4_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': str(row['height_cat']),
                    'target': str(row['year_cat']),
                    'value': row['count']
                })

        level5_counts = df_all.groupby(['year_cat', 'drainage_cat'], observed=True).size().reset_index(name='count')
        for _, row in level5_counts.iterrows():
            if row['count'] >= MIN_COUNT and str(row['drainage_cat']) in valid_drainage_labels:
                sankey_data['links'].append({
                    'source': str(row['year_cat']),
                    'target': str(row['drainage_cat']),
                    'value': row['count']
                })

        # Calculate node counts for display purposes (e.g., in tooltips).
        node_counts = {root_name: len(df_all)}
        node_counts.update(df_all.groupby('occ_cat').size().to_dict())
        node_counts.update(df_all.groupby('area_cat', observed=True).size().to_dict())
        node_counts.update(df_all.groupby('height_cat', observed=True).size().to_dict())
        node_counts.update(df_all.groupby('year_cat', observed=True).size().to_dict())
        node_counts.update(df_all.groupby('drainage_cat', observed=True).size().to_dict())

        # Identify active nodes (those with at least one link).
        active_nodes_names = set(link['source'] for link in sankey_data['links']) | \
                             set(link['target'] for link in sankey_data['links'])

        # Format the final list of nodes, including position information.
        final_nodes = []
        for name, node_info in nodes.items():
            if name in active_nodes_names:
                node_info['count'] = node_counts.get(name, 0)
                final_nodes.append(node_info)
                # Store the final position for use in the visualization library.
                sankey_data['node_positions'][name] = node_info.get('position', 0.5)

        sankey_data['nodes'] = final_nodes

        # Add descriptive binning information for the UI.
        sankey_data['bin_info'] = {
            'Area': f"Small (<{area_bins[1]:.0f} sqm), Medium ({area_bins[1]:.0f}-{area_bins[2]:.0f} sqm), Large (>{area_bins[2]:.0f} sqm)",
            'Height': f"Low (<{height_bins[1]:.1f}m), Mid ({height_bins[1]:.1f}-{height_bins[2]:.1f}m), High (>{height_bins[2]:.1f}m)",
            'Year': f"Historic (<{year_bins[1]}), Mid-Century ({year_bins[1]}-{year_bins[2]}), Modern (>{year_bins[2]})",
            'Drainage': "Multiple classes including Well, Moderately, Poorly drained, etc."
        }

        print(f"    all: {len(sankey_data['nodes'])} nodes, {len(sankey_data['links'])} links processed")
        return sankey_data

    def _process_single_occupancy_hierarchy(self, df_subset, name,
                                            area_bins, height_bins, year_bins,
                                            area_labels, height_labels, year_labels, drainage_labels):
        """
        Processes the hierarchical distribution for a single occupancy category.

        This function creates a Sankey diagram starting from the specified occupancy
        type as the root node and assigns fixed vertical positions to subsequent nodes.

        Args:
            df_subset (pd.DataFrame): DataFrame filtered for a single occupancy type.
            name (str): The name of the occupancy category, to be used as the root node.
            (other args): Bin and label information, same as the function above.

        Returns:
            dict: A dictionary formatted for the Sankey diagram.
        """
        sankey_data = {
            'nodes': [],
            'links': [],
            'total_buildings': len(df_subset),
            'node_positions': {}  # Store node position information.
        }
        nodes = {}



        def add_node(name, level, position=None):
            """Helper function to add a new node if it doesn't already exist."""
            if name not in nodes:
                nodes[name] = {
                    'name': name,
                    'display_name': name,
                    'level': level,
                    'position': position
                }

        # Root node (the occupancy category itself)
        root_name = name
        add_node(root_name, 0, 0.5)  # Centered

        # Area - Fixed order
        area_positions = {'Small': 0.25, 'Medium': 0.5, 'Large': 0.75}
        for label in area_labels:
            add_node(label, 1, area_positions.get(label, 0.5))

        # Height - Fixed order
        height_positions = {'Low': 0.25, 'Mid': 0.5, 'High': 0.75}
        for label in height_labels:
            add_node(label, 2, height_positions.get(label, 0.5))

        # Year - Fixed order
        year_positions = {
            'Historic (<1940)': 0.25,
            'Mid-Century (40-80)': 0.5,
            'Modern (>1980)': 0.75
        }
        for label in year_labels:
            add_node(label, 3, year_positions.get(label, 0.5))

        # Drainage - Fixed order
        drainage_positions = {
            'Excessively drained': 0.1,
            'Well drained': 0.2,
            'Moderately well drained': 0.3,
            'Somewhat excessively drained': 0.4,
            'Somewhat poorly drained': 0.5,
            'Poorly drained': 0.7,
            'Very poorly drained': 0.8,
            'Unknown Drainage': 0.9
        }
        for label in drainage_labels:
            add_node(label, 4, drainage_positions.get(label, 0.5))

        MIN_COUNT = 0

        # Create links (maintaining original logic)
        # Links from Root (Occupancy Name) to Area
        area_counts = df_subset['area_cat'].value_counts()
        for area_cat, count in area_counts.items():
            if count >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': root_name,
                    'target': str(area_cat),
                    'value': count
                })

        # Links from Area to Height
        area_height_counts = df_subset.groupby(['area_cat', 'height_cat'], observed=True).size().reset_index(
            name='count')
        for _, row in area_height_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': str(row['area_cat']),
                    'target': str(row['height_cat']),
                    'value': row['count']
                })

        # Links from Height to Year
        height_year_counts = df_subset.groupby(['height_cat', 'year_cat'], observed=True).size().reset_index(
            name='count')
        for _, row in height_year_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': str(row['height_cat']),
                    'target': str(row['year_cat']),
                    'value': row['count']
                })

        # Links from Year to Drainage
        year_drainage_counts = df_subset.groupby(['year_cat', 'drainage_cat'], observed=True).size().reset_index(
            name='count')
        for _, row in year_drainage_counts.iterrows():
            if row['count'] >= MIN_COUNT:
                sankey_data['links'].append({
                    'source': str(row['year_cat']),
                    'target': str(row['drainage_cat']),
                    'value': row['count']
                })

        # Calculate node counts by summing the values of incoming links.
        node_counts = {root_name: len(df_subset)}
        for link in sankey_data['links']:
            target_node = link['target']
            if target_node not in node_counts:
                node_counts[target_node] = 0
            # This sums up flows into a node to get its total value
            # It's an approximation, assuming no nodes are both source and target in different links
            # A more robust method is to groupby the original dataframe for each category
        # Recalculating with groupby for accuracy
        node_counts.update(df_subset.groupby('area_cat', observed=True).size().to_dict())
        node_counts.update(df_subset.groupby('height_cat', observed=True).size().to_dict())
        node_counts.update(df_subset.groupby('year_cat', observed=True).size().to_dict())
        node_counts.update(df_subset.groupby('drainage_cat', observed=True).size().to_dict())

        # Identify active nodes (those with at least one link).
        active_nodes = set([link['source'] for link in sankey_data['links']] +
                           [link['target'] for link in sankey_data['links']])

        # Format final list of nodes.
        for node_name, node_info in nodes.items():
            if node_name in active_nodes:
                node_info['count'] = node_counts.get(node_name, 0)
                sankey_data['nodes'].append(node_info)
                sankey_data['node_positions'][node_name] = node_info.get('position', 0.5)

        # Add descriptive binning information for the UI.
        sankey_data['bin_info'] = {
            'Area': f"Small (<{area_bins[1]:.0f} sqm), Medium ({area_bins[1]:.0f}-{area_bins[2]:.0f} sqm), Large (>{area_bins[2]:.0f} sqm)",
            'Height': f"Low (<{height_bins[1]:.1f}m), Mid ({height_bins[1]:.1f}-{height_bins[2]:.1f}m), High (>{height_bins[2]:.1f}m)",
            'Year': f"Historic (<{year_bins[1]}), Mid-Century ({year_bins[1]}-{year_bins[2]}), Modern (>{year_bins[2]})",
            'Drainage': ", ".join(drainage_labels)
        }

        print(f"    {name}: {len(sankey_data['nodes'])} nodes, {len(sankey_data['links'])} links")
        return sankey_data

    def process_occupancy_hierarchy(self):
        """Processes the hierarchy from OCC_CLS to PRIM_OCC for a Sankey diagram, showing ALL categories."""
        print("Processing OCC_CLS to PRIM_OCC hierarchy (showing all categories)...")

        df = self.df_cleaned[['OCC_CLS', 'PRIM_OCC']].dropna()
        all_links = df.groupby(['OCC_CLS', 'PRIM_OCC']).size().reset_index(name='value')

        final_links = all_links[all_links['value'] > 0].copy()


        final_links['OCC_CLS_mod'] = final_links['OCC_CLS'].astype(str) + ' (Class)'


        condition = final_links['PRIM_OCC'] == 'Unclassified'
        true_values = 'Unclassified (from ' + final_links['OCC_CLS'] + ')'
        false_values = final_links['PRIM_OCC'].astype(str) + ' (Type)'

        final_links['PRIM_OCC_mod'] = np.where(condition, true_values, false_values)

        occ_cls_nodes = final_links['OCC_CLS_mod'].unique().tolist()
        prim_occ_nodes = final_links['PRIM_OCC_mod'].unique().tolist()

        all_node_labels = occ_cls_nodes + prim_occ_nodes
        node_map = {name: i for i, name in enumerate(all_node_labels)}

        sankey_nodes = [{'name': name} for name in all_node_labels]
        sankey_links = {
            'source': final_links['OCC_CLS_mod'].map(node_map).tolist(),
            'target': final_links['PRIM_OCC_mod'].map(node_map).tolist(),
            'value': final_links['value'].tolist()
        }

        return {
            'nodes': sankey_nodes,
            'links': sankey_links
        }

    def perform_clustering(self, n_clusters=7):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")

        # Set up preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['Est GFA sqmeters', 'year_built']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['OCC_CLS'])
            ])

        # Transform data
        X_prepared = self.preprocessor.fit_transform(self.df_cluster[['Est GFA sqmeters', 'year_built', 'OCC_CLS']])

        # Run K-means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.df_cluster['cluster'] = self.kmeans.fit_predict(X_prepared)

        print("Clustering complete")
        return self

    def calculate_elbow_scores(self, k_range=range(2, 16)):
        """Calculate WCSS scores for elbow method"""
        print("Calculating elbow scores...")

        features = ['OCC_CLS', 'Est GFA sqmeters', 'year_built']
        df_temp = self.df_cleaned[features].dropna()

        # Remove outliers
        area_threshold = df_temp['Est GFA sqmeters'].quantile(0.99999)
        df_temp = df_temp[df_temp['Est GFA sqmeters'] < area_threshold].copy()

        # Preprocess
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['Est GFA sqmeters', 'year_built']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['OCC_CLS'])
            ])
        X_prepared = preprocessor.fit_transform(df_temp)

        # Calculate WCSS
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X_prepared)
            wcss.append(kmeans.inertia_)
            print(f"  Computed k={k}")

        return list(k_range), wcss

    def _get_cluster_assignments_for_df(self, df_subset, feature_combo, k):
        """
        Perform clustering and return cluster assignments for each row in the dataframe.
        feature_combo can be: 'base', 'material', 'foundation', 'both'
        """
        if len(df_subset) < k:
            return None

        # Prepare features based on combination
        numerical_features = ['Est GFA sqmeters', 'year_built']
        categorical_features = ['OCC_CLS']

        if feature_combo == 'material' or feature_combo == 'both':
            if 'material_type' in df_subset.columns and df_subset['material_type'].notna().any():
                categorical_features.append('material_type')
        if feature_combo == 'foundation' or feature_combo == 'both':
            if 'foundation_type' in df_subset.columns and df_subset['foundation_type'].notna().any():
                categorical_features.append('foundation_type')

        # Check if all features exist
        all_features = numerical_features + categorical_features
        for feat in all_features:
            if feat not in df_subset.columns:
                print(f"    Warning: Feature '{feat}' not found for clustering. Skipping.")
                return None

        # Drop rows with NaN in any of the selected features for this specific clustering run
        df_clusterable = df_subset[all_features].dropna()
        if len(df_clusterable) < k:
            return None

        # Setup preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='passthrough')

        try:
            # Transform and cluster
            X_prepared = preprocessor.fit_transform(df_clusterable[all_features])
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_prepared)

            # Create a Series with the original index to map results back
            cluster_series = pd.Series(clusters, index=df_clusterable.index)

            # Return a series that can be aligned back to the original df_subset
            return cluster_series

        except Exception as e:
            print(f"    Error in clustering with {feature_combo} for assignments: {e}")
            return None

    def _perform_clustering_with_features(self, df_subset, feature_combo, k):
        """
        Perform clustering with specific feature combination
        feature_combo can be: 'base', 'material', 'foundation', 'both'
        """
        if len(df_subset) < k:
            return None

        # Prepare features based on combination
        numerical_features = ['Est GFA sqmeters', 'year_built']
        categorical_features = ['OCC_CLS']

        if feature_combo == 'material' or feature_combo == 'both':
            categorical_features.append('material_type')
        if feature_combo == 'foundation' or feature_combo == 'both':
            categorical_features.append('foundation_type')

        # Check if all features exist
        all_features = numerical_features + categorical_features
        for feat in all_features:
            if feat not in df_subset.columns:
                return None

        # Setup preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        try:
            # Transform and cluster
            X_prepared = preprocessor.fit_transform(df_subset[all_features])
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_prepared)

            # Calculate statistics
            cluster_stats = []
            for cluster_id in range(k):
                cluster_mask = clusters == cluster_id
                cluster_data = df_subset[cluster_mask]

                if len(cluster_data) == 0:
                    continue

                stats = {
                    'cluster_id': cluster_id,
                    'count': len(cluster_data),
                    'avg_area': float(cluster_data['Est GFA sqmeters'].mean()),
                    'avg_year': int(cluster_data['year_built'].mean()),
                    'std_area': float(cluster_data['Est GFA sqmeters'].std(ddof=0)),
                    'std_year': float(cluster_data['year_built'].std(ddof=0))
                }

                # Add dominant material/foundation if applicable
                if 'material_type' in categorical_features:
                    material_counts = cluster_data['material_type'].value_counts()
                    if len(material_counts) > 0:
                        stats['dominant_material'] = material_counts.index[0]

                if 'foundation_type' in categorical_features:
                    foundation_counts = cluster_data['foundation_type'].value_counts()
                    if len(foundation_counts) > 0:
                        stats['dominant_foundation'] = foundation_counts.index[0]

                cluster_stats.append(stats)

            return {
                'wcss': float(kmeans.inertia_),
                'clusters': cluster_stats
            }
        except Exception as e:
            print(f"    Error in clustering with {feature_combo}: {e}")
            return None

    def _get_cluster_stats_for_df(self, df_to_cluster):
        """Helper function to perform clustering and get stats for a given dataframe"""
        k_results = {}

        if len(df_to_cluster) < 10:
            return None

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_to_cluster[['Est GFA sqmeters', 'year_built']])

        # Perform clustering for different k values (2-7)
        for k in range(2, 8):
            if len(df_to_cluster) < k:
                continue

            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            df_to_cluster[f'cluster_k{k}'] = clusters

            # Analyze clusters
            cluster_stats = []
            for cluster_id in range(k):
                cluster_data = df_to_cluster[df_to_cluster[f'cluster_k{k}'] == cluster_id]

                if len(cluster_data) == 0: continue

                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'count': len(cluster_data),
                    'avg_area': float(cluster_data['Est GFA sqmeters'].mean()),
                    'avg_year': int(cluster_data['year_built'].mean()),
                    'std_area': float(cluster_data['Est GFA sqmeters'].std(ddof=0)),
                    'std_year': float(cluster_data['year_built'].std(ddof=0))
                })

            k_results[k] = {
                'wcss': float(kmeans.inertia_),
                'clusters': cluster_stats
            }
        return k_results

    def get_overview_occupancy_counts(self):
        """Get overall occupancy counts for all buildings (not just pre-1940)"""
        print("Calculating overview occupancy counts...")

        # Use all cleaned data
        occ_counts = self.df_cleaned['OCC_CLS'].value_counts()

        return occ_counts.to_dict()

    def process_temporal_data(self):
        """Process data for temporal analysis"""
        print("Processing temporal data...")

        temporal_data = []

        # Process by year
        for year in self.df_cluster['year_built'].unique():
            year_data = self.df_cluster[self.df_cluster['year_built'] == year]

            for occ_cls in year_data['OCC_CLS'].unique():
                occ_data = year_data[year_data['OCC_CLS'] == occ_cls]

                temporal_data.append({
                    'year': int(year),
                    'display_year': 'pre-1940' if int(year) < 1940 else str(int(year)),
                    'occupancy': occ_cls,
                    'count': len(occ_data),
                    'avg_area': float(occ_data['Est GFA sqmeters'].mean()),
                    'total_area': float(occ_data['Est GFA sqmeters'].sum())
                })

        return temporal_data

    def process_pre1940_data(self):
        """Process pre-1940 building data"""
        print("Processing pre-1940 data...")

        df_pre_1940 = self.df_cleaned[self.df_cleaned['year_built'] < 1940].copy()

        # Get occupancy counts
        occ_counts = df_pre_1940['OCC_CLS'].value_counts()

        pre1940_data = {
            'total_count': len(df_pre_1940),
            'occupancy_counts': occ_counts.to_dict(),
            'residential_count': int(occ_counts.get('Residential', 0)),
            'non_residential_count': int(occ_counts.drop('Residential', errors='ignore').sum()),
            'percentage_of_total': round(len(df_pre_1940) / len(self.df_cleaned) * 100, 2)
        }

        return pre1940_data

    def process_post1940_data(self):
        """Process post-1940 building data"""
        print("Processing post-1940 data...")

        df_post_1940 = self.df_cleaned[self.df_cleaned['year_built'] >= 1940].copy()

        # Process by decade
        decade_data = {}
        for decade in range(1940, 2030, 10):
            decade_df = df_post_1940[
                (df_post_1940['year_built'] >= decade) &
                (df_post_1940['year_built'] < decade + 10)
            ]

            if len(decade_df) > 0:
                decade_counts = decade_df['OCC_CLS'].value_counts()
                decade_data[f"{decade}s"] = {
                    'total': len(decade_df),
                    'occupancy_counts': decade_counts.to_dict()
                }

        return decade_data

    def process_occupancy_clusters_enhanced(self):
        """
        Process clustering for each occupancy class with multiple k values
        AND different feature combinations (base, +material, +foundation, +both)
        """
        print("Processing enhanced occupancy-specific clusters with feature combinations...")
        occupancy_clusters = {}

        # Feature combinations to test
        feature_combos = ['base', 'material', 'foundation', 'both']

        # First, process for "all" classes
        print("  Processing 'all' with multiple feature combinations...")
        features_extended = ['Est GFA sqmeters', 'year_built', 'OCC_CLS', 'material_type', 'foundation_type']

        df_all = self.df_cleaned[features_extended].dropna().copy()

        if len(df_all) > 10:
            all_results = {
                'total_buildings': len(df_all),
                'feature_combinations': {}
            }

            for combo in feature_combos:
                print(f"    Computing clustering for feature combo: {combo}")
                combo_results = {}

                for k in range(2, 8):
                    result = self._perform_clustering_with_features(df_all, combo, k)
                    if result:
                        combo_results[k] = result

                if combo_results:
                    all_results['feature_combinations'][combo] = combo_results

            occupancy_clusters['all'] = all_results

        # Then, process for each individual occupancy class
        for occ_class in self.df_cleaned['OCC_CLS'].unique():
            print(f"  Processing '{occ_class}' with multiple feature combinations...")
            df_occ = self.df_cleaned[self.df_cleaned['OCC_CLS'] == occ_class][features_extended].dropna().copy()

            if len(df_occ) > 10:
                occ_results = {
                    'total_buildings': len(df_occ),
                    'feature_combinations': {}
                }

                for combo in feature_combos:
                    print(f"    Computing clustering for {occ_class} with feature combo: {combo}")
                    combo_results = {}

                    for k in range(2, 8):
                        result = self._perform_clustering_with_features(df_occ, combo, k)
                        if result:
                            combo_results[k] = result

                    if combo_results:
                        occ_results['feature_combinations'][combo] = combo_results

                occupancy_clusters[occ_class] = occ_results

        return occupancy_clusters

    def process_occupancy_clusters(self):
        """Keep original method for backward compatibility"""
        print("Processing occupancy-specific clusters (original method)...")
        occupancy_clusters = {}
        features = ['Est GFA sqmeters', 'year_built']

        # First, process for "all" classes
        print("  Processing 'all'...")
        df_all = self.df_cleaned[features].dropna().copy()
        k_results_all = self._get_cluster_stats_for_df(df_all)
        if k_results_all:
            occupancy_clusters['all'] = {
                'total_buildings': len(df_all),
                'k_values': k_results_all
            }

        # Then, process for each individual occupancy class
        for occ_class in self.df_cleaned['OCC_CLS'].unique():
            print(f"  Processing '{occ_class}'...")
            df_occ = self.df_cleaned[self.df_cleaned['OCC_CLS'] == occ_class][features].dropna().copy()

            k_results_occ = self._get_cluster_stats_for_df(df_occ)
            if k_results_occ:
                occupancy_clusters[occ_class] = {
                    'total_buildings': len(df_occ),
                    'k_values': k_results_occ
                }

        return occupancy_clusters

    def process_materials_foundation(self):
        """Process building materials and foundation data with occupancy breakdown AND Est GFA"""
        print("Processing materials and foundation data with occupancy breakdown and Est GFA...")

        # Process real data with occupancy breakdown and Est GFA
        materials_data = {}

        for filter_type, df_filtered in [
            ('all', self.df_cleaned),
            ('pre1940', self.df_cleaned[self.df_cleaned['year_built'] < 1940]),
            ('post1940', self.df_cleaned[self.df_cleaned['year_built'] >= 1940])
        ]:
            # Create contingency table for counts
            contingency = pd.crosstab(
                df_filtered['material_type'],
                df_filtered['foundation_type']
            )

            # Create contingency table for Est GFA
            area_contingency = pd.crosstab(
                df_filtered['material_type'],
                df_filtered['foundation_type'],
                values=df_filtered['Est GFA sqmeters'],
                aggfunc='sum'
            ).fillna(0)

            # Calculate occupancy breakdown for each material/foundation combination
            occupancy_breakdown = {}

            for mat in contingency.index:
                for found in contingency.columns:
                    # Get all buildings with this material/foundation combo
                    mask = (df_filtered['material_type'] == mat) & (df_filtered['foundation_type'] == found)
                    combo_buildings = df_filtered[mask]

                    if len(combo_buildings) > 0:
                        # Get occupancy counts and areas for this combination
                        occ_counts = combo_buildings['OCC_CLS'].value_counts()
                        occ_areas = combo_buildings.groupby('OCC_CLS')['Est GFA sqmeters'].sum()

                        key = f"{mat}_{found}"
                        occupancy_breakdown[key] = {
                            'total': len(combo_buildings),
                            'total_area': float(combo_buildings['Est GFA sqmeters'].sum()),
                            'occupancy_counts': occ_counts.to_dict(),
                            'occupancy_areas': occ_areas.to_dict()
                        }

            materials_data[filter_type] = {
                'matrix': contingency.values.tolist(),
                'area_matrix': area_contingency.values.tolist(),
                'materials': contingency.index.tolist(),
                'foundations': contingency.columns.tolist(),
                'occupancy_breakdown': occupancy_breakdown
            }

        return materials_data

    def process_soil_analysis(self):
        """
        Process all soil-related data.
        This includes mapping numerical engineering properties to categorical labels,
        calculating statistics for various soil features, and performing risk analysis.
        Enhanced: Now includes compname analysis
        """
        print("Processing soil data analysis...")

        # --- START: New code block for mapping numerical 'eng_property' to categories ---
        # Check if the 'eng_property' column exists and contains numeric data before attempting to map it.
        if 'eng_property' in self.df_cleaned.columns and pd.api.types.is_numeric_dtype(self.df_cleaned['eng_property']):
            print("  Mapping numerical engineering properties to categories based on defined ranges...")

            # Define the bin edges for the ranges. Using -inf and inf ensures all values are included.
            # You can adjust these bin edges based on your data's specific meaning.
            # Example ranges: (-inf, 0.17], (0.17, 0.24], (0.24, 0.32], (0.32, inf]
            bins = [-float('inf'), 0.17, 0.24, 0.32, float('inf')]

            # Define the string labels that correspond to each bin.
            labels = ['Favorable', 'Fair', 'Poor', 'Very poor']

            # Use the pandas 'cut' function to segment the data into the bins and assign the appropriate label.
            # This overwrites the original numeric 'eng_property' column with the new categorical data.
            self.df_cleaned['eng_property'] = pd.cut(self.df_cleaned['eng_property'], bins=bins, labels=labels,
                                                     right=True)
        # --- END: New code block ---

        soil_columns = ['drainagecl', 'wtdepannmin', 'flodfreqcl', 'eng_property',
                        'compname', 'comppct_r', 'MUSYM', 'mukey', 'LONGITUDE', 'LATITUDE']

        # Check which soil-related columns exist in the dataframe.
        existing_soil_cols = [col for col in soil_columns if col in self.df_cleaned.columns]


        # Initialize the dictionary to hold all soil analysis results.
        soil_analysis = {
            'drainage_class_stats': {},
            'flooding_freq_stats': {},
            'water_table_stats': {},
            'engineering_property_stats': {},
            'compname_stats': {},  # NEW: Added compname statistics
            'soil_by_occupancy': {},
            'spatial_distribution': [],
            'soil_risk_analysis': {}
        }

        if 'drainagecl' in self.df_cleaned.columns:
            drainage_counts = self.df_cleaned['drainagecl'].value_counts(dropna=False)
            counts_dict = drainage_counts.to_dict()
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)
                counts_dict['NaN (Missing)'] = nan_val

            soil_analysis['drainage_class_stats'] = {
                'counts': counts_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in counts_dict.items()}
            }

        # Calculate flooding frequency statistics if the column exists.
        if 'flodfreqcl' in self.df_cleaned.columns:
            flood_counts = self.df_cleaned['flodfreqcl'].value_counts(dropna=False)
            counts_dict = flood_counts.to_dict()
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)
                counts_dict['NaN (Missing)'] = nan_val

            soil_analysis['flooding_freq_stats'] = {
                'counts': counts_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in counts_dict.items()}
            }

        # Calculate water table depth statistics if the column exists.
        if 'wtdepannmin' in self.df_cleaned.columns:
            water_table = self.df_cleaned['wtdepannmin'].dropna()
            soil_analysis['water_table_stats'] = {
                'mean': float(water_table.mean()),
                'median': float(water_table.median()),
                'std': float(water_table.std()),
                'min': float(water_table.min()),
                'max': float(water_table.max()),
                'q25': float(water_table.quantile(0.25)),
                'q75': float(water_table.quantile(0.75))
            }

        # Calculate engineering property statistics if the column exists.
        if 'eng_property' in self.df_cleaned.columns:
            eng_counts = self.df_cleaned['eng_property'].value_counts(dropna=False)
            counts_dict = eng_counts.to_dict()
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)
                counts_dict['NaN (Missing)'] = nan_val

            soil_analysis['engineering_property_stats'] = {
                'counts': counts_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in counts_dict.items()}
            }

        # NEW: Calculate compname statistics if the column exists
        if 'compname' in self.df_cleaned.columns:
            comp_counts = self.df_cleaned['compname'].value_counts(dropna=False)
            counts_dict = comp_counts.to_dict()
            nan_val = None
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)

            # Get top 20 most common soil component names from the non-NaN data
            top_comp_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)[:20])

            # Re-add the NaN count if it exists
            if nan_val is not None:
                top_comp_dict['NaN (Missing)'] = nan_val

            soil_analysis['compname_stats'] = {
                'counts': top_comp_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in top_comp_dict.items()},
                'total_unique': len(comp_counts),
                'top_20_coverage': (sum(top_comp_dict.values()) - (nan_val or 0)) / len(self.df_cleaned) * 100
            }

        # Group soil properties by occupancy class.
        for occ_class in self.df_cleaned['OCC_CLS'].unique():
            occ_data = self.df_cleaned[self.df_cleaned['OCC_CLS'] == occ_class]
            occ_soil_stats = {}

            if 'drainagecl' in occ_data.columns:
                counts = occ_data['drainagecl'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                if np.nan in counts_dict:
                    counts_dict['NaN (Missing)'] = counts_dict.pop(np.nan)
                occ_soil_stats['drainage_distribution'] = counts_dict

            if 'flodfreqcl' in occ_data.columns:
                counts = occ_data['flodfreqcl'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                if np.nan in counts_dict:
                    counts_dict['NaN (Missing)'] = counts_dict.pop(np.nan)
                occ_soil_stats['flooding_distribution'] = counts_dict

            if 'eng_property' in occ_data.columns:
                counts = occ_data['eng_property'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                if np.nan in counts_dict:
                    counts_dict['NaN (Missing)'] = counts_dict.pop(np.nan)
                occ_soil_stats['engineering_distribution'] = counts_dict

            if 'compname' in occ_data.columns:
                counts = occ_data['compname'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                nan_val = None
                if np.nan in counts_dict:
                    nan_val = counts_dict.pop(np.nan)

                top_10_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)[:10])

                if nan_val is not None:
                    top_10_dict['NaN (Missing)'] = nan_val
                occ_soil_stats['compname_distribution'] = top_10_dict

            if 'wtdepannmin' in occ_data.columns:
                water_table_occ = occ_data['wtdepannmin'].dropna()
                if len(water_table_occ) > 0:
                    occ_soil_stats['water_table_stats'] = {
                        'mean': float(water_table_occ.mean()),
                        'median': float(water_table_occ.median()),
                        'std': float(water_table_occ.std())
                    }
            soil_analysis['soil_by_occupancy'][occ_class] = occ_soil_stats

        # Prepare a sample of data for the spatial map visualization.
        if 'LONGITUDE' in self.df_cleaned.columns and 'LATITUDE' in self.df_cleaned.columns:
            sample_size = min(75000, len(self.df_cleaned))
            spatial_sample = self.df_cleaned.sample(n=sample_size, random_state=337)

            for _, row in spatial_sample.iterrows():
                point_data = {
                    'lon': float(row['LONGITUDE']) if pd.notna(row['LONGITUDE']) else None,
                    'lat': float(row['LATITUDE']) if pd.notna(row['LATITUDE']) else None,
                    'occupancy': row['OCC_CLS'],
                    'year_built': int(row['year_built']),
                    'area': float(row['Est GFA sqmeters'])
                }
                if 'drainagecl' in row and pd.notna(row['drainagecl']):
                    point_data['drainage'] = row['drainagecl']
                if 'flodfreqcl' in row and pd.notna(row['flodfreqcl']):
                    point_data['flooding'] = row['flodfreqcl']
                if 'eng_property' in row and pd.notna(row['eng_property']):
                    point_data['eng_property'] = row['eng_property']
                if 'wtdepannmin' in row and pd.notna(row['wtdepannmin']):
                    point_data['water_table'] = float(row['wtdepannmin'])
                if 'compname' in row and pd.notna(row['compname']):
                    point_data['compname'] = row['compname']  # NEW: Added compname to spatial data

                if point_data['lon'] is not None and point_data['lat'] is not None:
                    soil_analysis['spatial_distribution'].append(point_data)

        # Perform risk analysis based on high-risk soil properties.
        if 'drainagecl' in self.df_cleaned.columns and 'flodfreqcl' in self.df_cleaned.columns:
            high_risk_drainage = ['Poorly drained', 'Very poorly drained']
            high_risk_flooding = ['High']

            high_risk_buildings = self.df_cleaned[
                (self.df_cleaned['drainagecl'].isin(high_risk_drainage)) |
                (self.df_cleaned['flodfreqcl'].isin(high_risk_flooding))
                ]

            soil_analysis['soil_risk_analysis'] = {
                'high_risk_count': len(high_risk_buildings),
                'high_risk_percentage': round(len(high_risk_buildings) / len(self.df_cleaned) * 100, 2),
                'high_risk_by_occupancy': high_risk_buildings['OCC_CLS'].value_counts().to_dict(),
                'high_risk_avg_year': int(high_risk_buildings['year_built'].mean()) if len(
                    high_risk_buildings) > 0 else 0,
                'high_risk_total_area': float(high_risk_buildings['Est GFA sqmeters'].sum())
            }

        return soil_analysis

    def calculate_nsi_data_sources(self):
        """Return hardcoded NSI methodology statistics"""
        print("Returning NSI data source methodology...")

        # These are fixed values representing NSI dataset methodology
        # Not specific to your MA dataset
        nsi_stats = {
            'methodology': 'NSI Dataset Construction',
            'note': 'These values represent the general NSI dataset methodology, not this specific MA subset'
        }

        return nsi_stats

    def get_cluster_analysis(self):
        """Get cluster analysis results"""
        print("Analyzing clusters...")

        cluster_analysis = self.df_cluster.groupby('cluster').agg({
            'Est GFA sqmeters': ['mean', 'median', 'std'],
            'year_built': ['mean', 'median', 'std'],
            'OCC_CLS': [('count', 'size'), ('most_common', lambda x: x.value_counts().index[0])]
        })

        # Flatten column names
        cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns]

        # Convert to list of dictionaries
        clusters = []
        for cluster_id in cluster_analysis.index:
            row = cluster_analysis.loc[cluster_id]
            clusters.append({
                'cluster_id': int(cluster_id),
                'count': int(row['OCC_CLS_count']),
                'most_common_occ': row['OCC_CLS_most_common'],
                'area_mean': float(row['Est GFA sqmeters_mean']),
                'area_median': float(row['Est GFA sqmeters_median']),
                'area_std': float(row['Est GFA sqmeters_std']) if not pd.isna(row['Est GFA sqmeters_std']) else 0,
                'year_mean': int(row['year_built_mean']),
                'year_median': int(row['year_built_median']),
                'year_std': float(row['year_built_std']) if not pd.isna(row['year_built_std']) else 0
            })

        return clusters

    def prepare_enhanced_samples(self):
        """
        Create samples with pre-computed clusters for all feature combinations
        Returns DataFrames for export
        """
        print("Creating enhanced samples with multi-dimensional clustering...")

        # Prepare base features
        features = ['Est GFA sqmeters', 'year_built', 'OCC_CLS', 'material_type', 'foundation_type']

        # Add soil features if they exist
        soil_features = ['drainagecl', 'flodfreqcl', 'eng_property', 'wtdepannmin', 'compname', 'LONGITUDE', 'LATITUDE']
        for sf in soil_features:
            if sf in self.df_cleaned.columns:
                features.append(sf)

        df_for_samples = self.df_cleaned[features].dropna(subset=['Est GFA sqmeters', 'year_built', 'OCC_CLS']).copy()

        # Remove outliers
        area_threshold = df_for_samples['Est GFA sqmeters'].quantile(0.99999)
        df_for_samples = df_for_samples[df_for_samples['Est GFA sqmeters'] < area_threshold]

        # Create random sample
        random_sample_size = min(75000, len(df_for_samples))
        random_sample_df = df_for_samples.sample(n=random_sample_size, random_state=337).copy()

        # Create balanced sample
        SAMPLES_PER_CLASS = 2500
        balanced_sample_df = df_for_samples.groupby('OCC_CLS', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=337)
        ).copy()

        # Reset indices
        random_sample_df = random_sample_df.reset_index(drop=True)
        balanced_sample_df = balanced_sample_df.reset_index(drop=True)

        # Perform REAL clustering for all feature combinations for both samples
        for sample_df, sample_name in [(random_sample_df, 'random'), (balanced_sample_df, 'balanced')]:
            print(f"  Performing REAL clustering on {sample_name} sample...")

            feature_combos = ['base', 'material', 'foundation', 'both']

            for combo in feature_combos:
                print(f"    - Clustering with feature combo: {combo}")
                for k in range(2, 10):
                    # Call our new function to get actual cluster assignments
                    cluster_assignments = self._get_cluster_assignments_for_df(sample_df, combo, k)

                    # The result is a Series, which will automatically align by index.
                    # Buildings that couldn't be clustered (e.g., due to missing data for that combo)
                    # will have NaN, which becomes null in JSON.
                    sample_df[f'cluster_{combo}_k{k}'] = cluster_assignments

            # Add compatibility aliases and a default cluster column
            print(f"  - Finalizing cluster columns for {sample_name} sample...")
            for k in range(2, 10):
                if f'cluster_base_k{k}' in sample_df.columns:
                    sample_df[f'cluster_k{k}'] = sample_df[f'cluster_base_k{k}']

            if 'cluster_base_k7' in sample_df.columns:
                sample_df['cluster'] = sample_df['cluster_base_k7']
            else:
                sample_df['cluster'] = None

        print(f"  Random sample size: {len(random_sample_df)}")
        print(f"  Balanced sample size: {len(balanced_sample_df)}")

        # Return DataFrames, not lists
        return random_sample_df, balanced_sample_df

    def clean_for_json(self, obj):
        """Recursively clean data for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.floating, np.complexfloating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return self.clean_for_json(obj.tolist())
        elif pd.isna(obj):
            return None
        else:
            return obj

    def export_to_json(self, output_path='building_data.json'):
        """Export all processed data to JSON - Split into main and multiple sample files"""
        print("Exporting data to JSON (split into multiple files)...")

        # Get elbow scores
        k_range, wcss = self.calculate_elbow_scores()

        # Pre-calculate enhanced occupancy clusters
        occupancy_clusters_enhanced = self.process_occupancy_clusters_enhanced()

        # Also keep original occupancy clusters for backward compatibility
        occupancy_clusters_data = self.process_occupancy_clusters()

        # Get overview occupancy counts
        overview_occupancy_counts = self.get_overview_occupancy_counts()

        # Process soil analysis with compname
        soil_analysis_data = self.process_soil_analysis()

        # OCC type sankey
        occupancy_hierarchy_sankey = self.process_occupancy_hierarchy()

        # Calculate NSI data source statistics
        nsi_data_sources = self.calculate_nsi_data_sources()


        hierarchical_distribution = self.process_hierarchical_distribution()

        # Get enhanced samples as DataFrames
        random_sample_df, balanced_sample_df = self.prepare_enhanced_samples()

        # Prepare MAIN export data (without samples)
        main_data = {
            'metadata': {
                'total_buildings': len(self.df_cleaned),
                'date_processed': datetime.now().isoformat(),
                'source_file': self.csv_path,
                'version': '3.2',  # Version 3.2 includes compname and data flow analysis
                'has_samples_file': True,
                'samples_split': True,
                'samples_files': []
            },
            'hierarchical_distribution': hierarchical_distribution,
            'summary_stats': {
                'total_buildings': len(self.df_cleaned),
                'avg_year_built': int(self.df_cleaned['year_built'].mean()),
                'avg_area_sqm': float(self.df_cleaned['Est GFA sqmeters'].dropna().mean()),
                'min_year': int(self.df_cleaned['year_built'].min()),
                'max_year': int(self.df_cleaned['year_built'].max()),
                'occupancy_classes': sorted(self.df_cleaned['OCC_CLS'].unique().tolist())
            },
            'overview_occupancy_counts': overview_occupancy_counts,
            'clustering': {
                'elbow_k_values': k_range,
                'elbow_wcss_values': wcss,
                'clusters': self.get_cluster_analysis()
            },
            'temporal_data': self.process_temporal_data(),
            'pre1940': self.process_pre1940_data(),
            'post1940': self.process_post1940_data(),
            'occupancy_clusters': occupancy_clusters_data,
            'occupancy_clusters_enhanced': occupancy_clusters_enhanced,
            'materials_foundation': self.process_materials_foundation(),
            'soil_analysis': soil_analysis_data,  # Now includes compname analysis
            'occupancy_hierarchy_sankey': occupancy_hierarchy_sankey,
            'data_flow_stats': self.data_flow_stats,  # NEW: Data flow statistics
            'nsi_data_sources': nsi_data_sources  # NEW: NSI data source statistics
        }

        # Clean main data for JSON
        main_data = self.clean_for_json(main_data)

        # Split samples into chunks
        CHUNK_SIZE = 5000

        # Convert to list for chunking and clean for JSON
        random_samples_list = [self.clean_for_json(row) for row in random_sample_df.to_dict(orient='records')]
        balanced_samples_list = [self.clean_for_json(row) for row in balanced_sample_df.to_dict(orient='records')]

        # Split random samples into chunks
        random_chunks = [random_samples_list[i:i + CHUNK_SIZE]
                         for i in range(0, len(random_samples_list), CHUNK_SIZE)]

        # Split balanced samples into chunks
        balanced_chunks = [balanced_samples_list[i:i + CHUNK_SIZE]
                           for i in range(0, len(balanced_samples_list), CHUNK_SIZE)]

        sample_files_info = []
        total_samples_size = 0

        # Save random sample chunks
        for i, chunk in enumerate(random_chunks):
            filename = output_path.replace('.json', f'_samples_random_{i + 1}.json')
            chunk_data = {
                'metadata': {
                    'type': 'random',
                    'chunk_index': i + 1,
                    'total_chunks': len(random_chunks),
                    'chunk_size': len(chunk),
                    'date_generated': datetime.now().isoformat()
                },
                'samples': chunk
            }

            with open(filename, 'w') as f:
                json.dump(chunk_data, f, separators=(',', ':'))  # Compact format

            chunk_size_mb = len(json.dumps(chunk_data, separators=(',', ':'))) / 1024 / 1024
            total_samples_size += chunk_size_mb

            sample_files_info.append({
                'filename': filename.split('/')[-1],
                'type': 'random',
                'chunk_index': i + 1,
                'sample_count': len(chunk),
                'size_mb': round(chunk_size_mb, 2)
            })

            print(f"  Saved {filename} ({chunk_size_mb:.2f} MB, {len(chunk)} samples)")

        # Save balanced sample chunks
        for i, chunk in enumerate(balanced_chunks):
            filename = output_path.replace('.json', f'_samples_balanced_{i + 1}.json')
            chunk_data = {
                'metadata': {
                    'type': 'balanced',
                    'chunk_index': i + 1,
                    'total_chunks': len(balanced_chunks),
                    'chunk_size': len(chunk),
                    'date_generated': datetime.now().isoformat()
                },
                'samples': chunk
            }

            with open(filename, 'w') as f:
                json.dump(chunk_data, f, separators=(',', ':'))

            chunk_size_mb = len(json.dumps(chunk_data, separators=(',', ':'))) / 1024 / 1024
            total_samples_size += chunk_size_mb

            sample_files_info.append({
                'filename': filename.split('/')[-1],
                'type': 'balanced',
                'chunk_index': i + 1,
                'sample_count': len(chunk),
                'size_mb': round(chunk_size_mb, 2)
            })

            print(f"  Saved {filename} ({chunk_size_mb:.2f} MB, {len(chunk)} samples)")

        # Update main data with sample files info
        main_data['metadata']['samples_files'] = sample_files_info
        main_data['metadata']['total_random_samples'] = len(random_samples_list)
        main_data['metadata']['total_balanced_samples'] = len(balanced_samples_list)
        main_data['metadata']['random_chunks'] = len(random_chunks)
        main_data['metadata']['balanced_chunks'] = len(balanced_chunks)

        # Save main data
        with open(output_path, 'w') as f:
            json.dump(main_data, f, indent=2)

        main_size = len(json.dumps(main_data)) / 1024 / 1024

        print(f"\n{'=' * 60}")
        print(f"Export Complete!")
        print(f"{'=' * 60}")
        print(f"Main data exported to: {output_path} ({main_size:.2f} MB)")
        print(f"Sample files created: {len(sample_files_info)} files")
        print(f"  - Random samples: {len(random_chunks)} files ({len(random_samples_list)} total samples)")
        print(f"  - Balanced samples: {len(balanced_chunks)} files ({len(balanced_samples_list)} total samples)")
        print(f"Total samples size: {total_samples_size:.2f} MB")
        print(f"Average file size: {total_samples_size / len(sample_files_info):.2f} MB")
        print(f"Soil analysis data included: Yes (with compname analysis)")
        print(f"Data flow statistics included: Yes")
        print(f"NSI data source analysis included: Yes")

        # Check if any file exceeds 25MB
        for file_info in sample_files_info:
            if file_info['size_mb'] > 25:
                print(f"WARNING: {file_info['filename']} exceeds 25MB ({file_info['size_mb']} MB)")
                print(f"Consider reducing CHUNK_SIZE to {int(CHUNK_SIZE * 20 / file_info['size_mb'])}")

        return main_data

def main():
    """Main processing function"""
    print("="*60)
    print("Massachusetts Building Data Processing - Multi-dimensional Enhanced Version with Soil and Data Flow Analysis")
    print("="*60)

    # Initialize processor
    processor = BuildingDataProcessor('ma_structures_with_demolition_FINAL.csv')

    # Process data
    processor.load_data()
    processor.clean_data()
    processor.prepare_clustering_data(remove_outliers=False)
    processor.perform_clustering(n_clusters=7)

    # Export to JSON
    export_data = processor.export_to_json('building_data.json')

    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total buildings processed: {export_data['metadata']['total_buildings']:,}")
    print(f"Overview occupancy classes: {len(export_data.get('overview_occupancy_counts', {}))} types")
    print(f"Temporal data points: {len(export_data.get('temporal_data', []))}")
    print(f"Occupancy-specific clusters: {len(export_data.get('occupancy_clusters', {}))} classes")
    print(f"Enhanced clusters with features: {len(export_data.get('occupancy_clusters_enhanced', {}))} classes")
    print(f"Soil analysis included: Yes (with compname analysis)")
    print(f"Data flow analysis included: Yes")
    print("\nData exported to: building_data.json and building_data_samples_*.json files")
    print("You can now open the updated HTML dashboard to visualize the data including all new analyses")

if __name__ == "__main__":
    main()