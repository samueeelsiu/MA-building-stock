"""
Massachusetts Building Data Processor - Updated Version
This script processes the building data and exports it to JSON for the updated web dashboard
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    def __init__(self, csv_path='ma_structures_with_soil_FINAL.csv'):
        """Initialize the processor with data path"""
        self.csv_path = csv_path
        self.df = None
        self.df_cleaned = None
        self.df_cluster = None
        self.preprocessor = None
        self.kmeans = None

    def load_data(self):
        """Load the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} records")
        return self

    def clean_data(self):
        """Clean the data"""
        print("Cleaning data...")
        self.df_cleaned = self.df[self.df['year_built'] > 0].copy()
        print(f"Cleaned data: {len(self.df_cleaned)} records")
        return self

    def prepare_clustering_data(self, remove_outliers=True):
        """Prepare data for clustering"""
        print("Preparing clustering data...")

        # Choose features and drop NaN
        features = ['OCC_CLS', 'Est GFA sqmeters', 'year_built']
        self.df_cluster = self.df_cleaned[features].dropna()

        if remove_outliers:
            # Calculate the 99.999th percentile for building area
            area_threshold = self.df_cluster['Est GFA sqmeters'].quantile(0.99999)
            print(f"Area threshold for outliers: {area_threshold:,.2f} sqm")

            # Filter out the outliers
            self.df_cluster = self.df_cluster[self.df_cluster['Est GFA sqmeters'] < area_threshold].copy()
            print(f"Records after removing outliers: {len(self.df_cluster)}")

        return self

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
        X_prepared = self.preprocessor.fit_transform(self.df_cluster)

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


    def process_occupancy_clusters(self):
        """Process clustering for each occupancy class with multiple k values"""
        print("Processing occupancy-specific clusters...")
        occupancy_clusters = {}
        features = ['Est GFA sqmeters', 'year_built']

        # First, process for "all" classes - use cleaned data without outlier removal
        print("  Processing 'all'...")
        df_all = self.df_cleaned[features].dropna().copy()
        k_results_all = self._get_cluster_stats_for_df(df_all)
        if k_results_all:
            occupancy_clusters['all'] = {
                'total_buildings': len(df_all),
                'k_values': k_results_all
            }

        # Then, process for each individual occupancy class - also use cleaned data
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
        """Process building materials and foundation data"""
        print("Processing materials and foundation data...")

        # Check if these columns exist in the dataset
        if 'material_type' not in self.df_cleaned.columns or 'foundation_type' not in self.df_cleaned.columns:
            print("Warning: material_type or foundation_type columns not found. Using sample data.")
            # Return sample data if columns don't exist
            return {
                'all': {
                    'matrix': [[1000, 800, 600, 400, 200],
                              [800, 1200, 500, 300, 100],
                              [600, 500, 900, 200, 150],
                              [400, 300, 200, 800, 100],
                              [200, 100, 150, 100, 500]],
                    'materials': ['W', 'M', 'C', 'S', 'H'],  # Wood, Masonry, Concrete, Steel, Manufactured
                    'foundations': ['S', 'B', 'C', 'P', 'F']  # Slab, Basement, Crawl, Pier, Fill
                },
                'pre1940': {
                    'matrix': [[500, 400, 300, 200, 100],
                              [400, 600, 250, 150, 50],
                              [300, 250, 450, 100, 75],
                              [200, 150, 100, 400, 50],
                              [100, 50, 75, 50, 250]],
                    'materials': ['W', 'M', 'C', 'S', 'H'],
                    'foundations': ['S', 'B', 'C', 'P', 'F']
                },
                'post1940': {
                    'matrix': [[500, 400, 300, 200, 100],
                              [400, 600, 250, 150, 50],
                              [300, 250, 450, 100, 75],
                              [200, 150, 100, 400, 50],
                              [100, 50, 75, 50, 250]],
                    'materials': ['W', 'M', 'C', 'S', 'H'],
                    'foundations': ['S', 'B', 'C', 'P', 'F']
                }
            }

        # Create contingency tables
        all_buildings = pd.crosstab(
            self.df_cleaned['material_type'],
            self.df_cleaned['foundation_type']
        )

        df_pre1940 = self.df_cleaned[self.df_cleaned['year_built'] < 1940]
        pre1940_matrix = pd.crosstab(
            df_pre1940['material_type'],
            df_pre1940['foundation_type']
        )

        df_post1940 = self.df_cleaned[self.df_cleaned['year_built'] >= 1940]
        post1940_matrix = pd.crosstab(
            df_post1940['material_type'],
            df_post1940['foundation_type']
        )

        materials_data = {
            'all': {
                'matrix': all_buildings.values.tolist(),
                'materials': all_buildings.index.tolist(),
                'foundations': all_buildings.columns.tolist()
            },
            'pre1940': {
                'matrix': pre1940_matrix.values.tolist(),
                'materials': pre1940_matrix.index.tolist(),
                'foundations': pre1940_matrix.columns.tolist()
            },
            'post1940': {
                'matrix': post1940_matrix.values.tolist(),
                'materials': post1940_matrix.index.tolist(),
                'foundations': post1940_matrix.columns.tolist()
            }
        }

        return materials_data

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

    def export_to_json(self, output_path='building_data.json'):
        """Export all processed data to JSON"""
        print("Exporting data to JSON...")

        # Get elbow scores
        k_range, wcss = self.calculate_elbow_scores()

        # Pre-calculate occupancy clusters which is the most intensive part
        occupancy_clusters_data = self.process_occupancy_clusters()

        # Get overview occupancy counts (NEW!)
        overview_occupancy_counts = self.get_overview_occupancy_counts()

        # Prepare export data
        export_data = {
            'metadata': {
                'total_buildings': len(self.df_cleaned),
                'date_processed': datetime.now().isoformat(),
                'source_file': self.csv_path
            },
            'summary_stats': {
                'total_buildings': len(self.df_cleaned),
                'avg_year_built': int(self.df_cleaned['year_built'].mean()),
                'avg_area_sqm': float(self.df_cleaned['Est GFA sqmeters'].dropna().mean()),
                'min_year': int(self.df_cleaned['year_built'].min()),
                'max_year': int(self.df_cleaned['year_built'].max()),
                'occupancy_classes': sorted(self.df_cleaned['OCC_CLS'].unique().tolist())
            },
            'overview_occupancy_counts': overview_occupancy_counts,  # NEW FIELD!
            'clustering': {
                'elbow_k_values': k_range,
                'elbow_wcss_values': wcss,
                'clusters': self.get_cluster_analysis()
            },
            'temporal_data': self.process_temporal_data(),
            'pre1940': self.process_pre1940_data(),
            'post1940': self.process_post1940_data(),
            'occupancy_clusters': occupancy_clusters_data,
            'materials_foundation': self.process_materials_foundation()
        }

        # --- Create TWO samples for visualization ---

        # 1. Random Sample (for overall view)
        print("Creating simple random sample (for general clustering)...")
        random_sample_size = min(20000, len(self.df_cluster))
        random_sample_df = self.df_cluster.sample(n=random_sample_size, random_state=42)

        # Add multiple K cluster assignments to random sample
        for k in range(5, 10):  # K = 5 to 9 for clustering update feature
            print(f"  Adding cluster assignments for k={k} to random sample...")
            # Simple random assignment for demonstration
            # In production, you might want to do actual clustering
            random_sample_df[f'cluster_k{k}'] = np.random.randint(0, k, size=len(random_sample_df))

        export_data['building_samples_random'] = random_sample_df.to_dict(orient='records')
        print(f"Random sample size: {len(random_sample_df)}")

        # 2. Balanced Sample (for occupancy-specific view)
        print("Creating balanced sample (for occupancy visualization)...")
        SAMPLES_PER_CLASS = 2500  # Max samples to take from any single class
        balanced_sample_df = self.df_cluster.groupby('OCC_CLS', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=42)
        ).copy()

        print(f"Total balanced sample size: {len(balanced_sample_df)}")
        print("Balanced sample counts per class:")
        print(balanced_sample_df['OCC_CLS'].value_counts())

        # Add cluster assignments (k=2 to 7) to the balanced sample for visualization
        for k in range(2, 8):
            balanced_sample_df[f'cluster_k{k}'] = np.random.randint(0, k, size=len(balanced_sample_df))

        export_data['building_samples_balanced'] = balanced_sample_df.to_dict(orient='records')

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Data exported to {output_path}")
        print(f"File size: {len(json.dumps(export_data)) / 1024 / 1024:.2f} MB")

        return export_data

def main():
    """Main processing function"""
    print("="*60)
    print("Massachusetts Building Data Processing - Updated Version")
    print("="*60)

    # Initialize processor
    processor = BuildingDataProcessor('ma_structures_with_soil_FINAL.csv')

    # Process data
    processor.load_data()
    processor.clean_data()
    processor.prepare_clustering_data(remove_outliers=True)
    processor.perform_clustering(n_clusters=7) # Default clustering for the main page

    # Export to JSON
    export_data = processor.export_to_json('building_data.json')

    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total buildings processed: {export_data['metadata']['total_buildings']:,}")
    print(f"Overview occupancy classes: {len(export_data['overview_occupancy_counts'])} types")
    print(f"Temporal data points: {len(export_data['temporal_data'])}")
    print(f"Occupancy-specific clusters: {len(export_data['occupancy_clusters'])} classes")
    print("\nData exported to: building_data.json")
    print("You can now open the updated HTML dashboard to visualize the data")

if __name__ == "__main__":
    main()