import numpy as np
import pandas as pd


class BucketingHelper:
    @staticmethod
    def create_fixed_width_buckets(
            df,
            value_column,
            max_value=None,
            bucket_size=250,
            min_samples=70):
        """Create fixed-width buckets for a given DataFrame column"""
        if max_value is not None:
            df = df[df[value_column] <= max_value].copy()
        else:
            df = df.copy()

        df = df.sort_values(value_column)
        min_val = df[value_column].min()
        max_val = df[value_column].max()

        # Initial bucket creation with target size
        n_buckets = max(1, len(df) // bucket_size)  # Ensure integer division
        bucket_width = (max_val - min_val) / n_buckets

        # Create one extra bucket edge to ensure all values are included
        bucket_edges = np.linspace(min_val, max_val, num=n_buckets + 1)

        # Assign initial buckets
        df['bucket'] = pd.cut(df[value_column],
                              bins=bucket_edges,
                              labels=False,
                              include_lowest=True)

        # Handle NaN values that might occur at edges
        df['bucket'] = df['bucket'].fillna(n_buckets - 1)

        # Merge small buckets
        bucket_counts = df['bucket'].value_counts().sort_index()
        current_bucket = 0
        new_bucket_map = {}

        # Initialize with first bucket
        current_count = int(bucket_counts.get(0, 0))  # Ensure integer counts

        # Iterate through buckets and merge if needed
        for bucket in range(
                1, int(bucket_counts.index.max() + 1)):  # Ensure integer range
            bucket_count = int(
                bucket_counts.get(
                    bucket,
                    0))  # Ensure integer counts

            if current_count < min_samples:
                # Current bucket is too small, merge with next
                current_count += bucket_count
                new_bucket_map[bucket] = current_bucket
            else:
                # Current bucket is large enough, start new bucket
                current_bucket += 1
                current_count = bucket_count
                new_bucket_map[bucket] = current_bucket

        # Apply bucket mapping
        df['bucket'] = df['bucket'].map(
            lambda x: new_bucket_map.get(int(x), 0))

        # Recalculate width based on final buckets
        final_bucket_counts = df['bucket'].value_counts()
        n_final_buckets = len(final_bucket_counts)
        final_bucket_width = (max_val - min_val) / max(1,
                                                       n_final_buckets)  # Prevent division by zero

        return df, final_bucket_width

    @staticmethod
    def calculate_bucket_stats(df, bucket_column='bucket', metrics=None):
        """Calculate statistics for each bucket"""
        if metrics is None:
            return df.groupby(bucket_column).size().reset_index(name='count')

        stats = {}
        for metric_name, metric_info in metrics.items():
            if isinstance(metric_info, dict):
                required_cols = metric_info.get('columns', [])
                metric_func = metric_info.get('func')

                if metric_func and all(
                        col in df.columns for col in required_cols):
                    try:
                        stats[metric_name] = df.groupby(
                            bucket_column).apply(metric_func)
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {str(e)}")
                        stats[metric_name] = df.groupby(
                            bucket_column).size() * 0  # Create zero-filled series
            else:
                try:
                    stats[metric_name] = df.groupby(bucket_column)[metric_info].agg([
                        'mean', 'min', 'max', 'count'])
                except Exception as e:
                    print(f"Error calculating {metric_name}: {str(e)}")
                    stats[metric_name] = pd.Series(
                        0, index=df[bucket_column].unique())

        result = pd.DataFrame(stats).reset_index()

        # Add sample count if not already included
        if 'count' not in result.columns:
            counts = df.groupby(bucket_column).size().reset_index(name='count')
            result = result.merge(counts, on=bucket_column)

        return result
