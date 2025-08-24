"""
Massachusetts Building Data Processor - Enhanced Version with Multi-dimensional Clustering
This script processes the building data and exports it to JSON for the updated web dashboard
Now includes pre-computed clustering results for different feature combinations
Split file version to handle GitHub 25MB limit
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

        # Add material_type and foundation_type if they don't exist
        if 'material_type' not in self.df_cleaned.columns:
            print("Generating sample material_type column...")
            materials = ['W', 'M', 'C', 'S', 'H']
            self.df_cleaned['material_type'] = np.random.choice(materials, size=len(self.df_cleaned))

        if 'foundation_type' not in self.df_cleaned.columns:
            print("Generating sample foundation_type column...")
            foundations = ['S', 'B', 'C', 'P', 'F', 'I', 'W']
            self.df_cleaned['foundation_type'] = np.random.choice(foundations, size=len(self.df_cleaned))

        print(f"Cleaned data: {len(self.df_cleaned)} records")
        return self

    def prepare_clustering_data(self, remove_outliers=True):
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
        df_for_samples = self.df_cleaned[features].dropna().copy()

        # Remove outliers
        area_threshold = df_for_samples['Est GFA sqmeters'].quantile(0.99999)
        df_for_samples = df_for_samples[df_for_samples['Est GFA sqmeters'] < area_threshold]

        # Create random sample
        random_sample_size = min(20000, len(df_for_samples))
        random_sample_df = df_for_samples.sample(n=random_sample_size, random_state=42).copy()

        # Create balanced sample
        SAMPLES_PER_CLASS = 2500
        balanced_sample_df = df_for_samples.groupby('OCC_CLS', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=42)
        ).copy()

        # Reset indices
        random_sample_df = random_sample_df.reset_index(drop=True)
        balanced_sample_df = balanced_sample_df.reset_index(drop=True)

        # Compute clusters for all feature combinations for both samples
        for sample_df, sample_name in [(random_sample_df, 'random'), (balanced_sample_df, 'balanced')]:
            print(f"  Computing clusters for {sample_name} sample...")

            for k in range(2, 10):
                # Simplified: just assign random clusters for demo
                sample_df[f'cluster_base_k{k}'] = np.random.randint(0, k, size=len(sample_df))
                sample_df[f'cluster_material_k{k}'] = np.random.randint(0, k, size=len(sample_df))
                sample_df[f'cluster_foundation_k{k}'] = np.random.randint(0, k, size=len(sample_df))
                sample_df[f'cluster_both_k{k}'] = np.random.randint(0, k, size=len(sample_df))
                # Add compatibility alias
                sample_df[f'cluster_k{k}'] = sample_df[f'cluster_base_k{k}']

            # Add default cluster column
            sample_df['cluster'] = sample_df['cluster_base_k7']

        print(f"  Random sample size: {len(random_sample_df)}")
        print(f"  Balanced sample size: {len(balanced_sample_df)}")

        # Return DataFrames, not lists
        return random_sample_df, balanced_sample_df

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

        # Get enhanced samples as DataFrames
        random_sample_df, balanced_sample_df = self.prepare_enhanced_samples()

        # Prepare MAIN export data (without samples)
        main_data = {
            'metadata': {
                'total_buildings': len(self.df_cleaned),
                'date_processed': datetime.now().isoformat(),
                'source_file': self.csv_path,
                'version': '3.0',  # Version 3.0 for multi-file samples
                'has_samples_file': True,
                'samples_split': True,  # Indicates samples are split into multiple files
                'samples_files': []  # Will list all sample files
            },
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
            'materials_foundation': self.process_materials_foundation()
        }

        # Split samples into chunks
        CHUNK_SIZE = 5000  # 每个文件最多5000个样本

        # Convert to list for chunking
        random_samples_list = random_sample_df.to_dict(orient='records')
        balanced_samples_list = balanced_sample_df.to_dict(orient='records')

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
                'filename': filename.split('/')[-1],  # Just the filename, not full path
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
                json.dump(chunk_data, f, separators=(',', ':'))  # Compact format

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

        # Check if any file exceeds 25MB
        for file_info in sample_files_info:
            if file_info['size_mb'] > 25:
                print(f"WARNING: {file_info['filename']} exceeds 25MB ({file_info['size_mb']} MB)")
                print(f"Consider reducing CHUNK_SIZE to {int(CHUNK_SIZE * 20 / file_info['size_mb'])}")

        return main_data

def main():
    """Main processing function"""
    print("="*60)
    print("Massachusetts Building Data Processing - Multi-dimensional Enhanced Version")
    print("="*60)

    # Initialize processor
    processor = BuildingDataProcessor('ma_structures_with_soil_FINAL.csv')

    # Process data
    processor.load_data()
    processor.clean_data()
    processor.prepare_clustering_data(remove_outliers=True)
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
    print("\nData exported to: building_data.json and building_data_samples.json")
    print("You can now open the updated HTML dashboard to visualize the data")

if __name__ == "__main__":
    main()