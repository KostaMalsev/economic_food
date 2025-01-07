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
        """Create fixed-width buckets for a given DataFrame column with minimum sample size enforcement
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        value_column : str
            Column name to bucket
        max_value : float, optional
            Maximum value to consider
        bucket_size : int, default=250
            Target number of samples per bucket
        min_samples : int, default=70
            Minimum number of samples required per bucket
            
        Returns:
        --------
        tuple
            (DataFrame with bucket column, final bucket width)
        """
        if max_value is not None:
            df = df[df[value_column] <= max_value].copy()
        else:
            df = df.copy()

        # Sort data and get range
        df = df.sort_values(value_column)
        min_val = df[value_column].min()
        max_val = df[value_column].max()

        # Initial bucket creation with target size
        n_buckets = max(1, len(df) // bucket_size)
        bucket_width = (max_val - min_val) / n_buckets

        # Create bucket edges
        bucket_edges = np.linspace(min_val, max_val, num=n_buckets + 1)

        # Assign initial buckets
        df['bucket'] = pd.cut(df[value_column],
                            bins=bucket_edges,
                            labels=False,
                            include_lowest=True)

        # Handle NaN values that might occur at edges
        df['bucket'] = df['bucket'].fillna(n_buckets - 1)

        # Get bucket counts
        bucket_counts = df['bucket'].value_counts().sort_index()
        
        # Initialize variables for new merging logic
        new_bucket_map = {}
        current_bucket = 0
        accumulated_samples = 0
        buckets_to_merge = []

        # Forward pass: merge buckets until min_samples is reached
        for bucket in range(int(bucket_counts.index.max() + 1)):
            count = int(bucket_counts.get(bucket, 0))
            accumulated_samples += count
            buckets_to_merge.append(bucket)

            if accumulated_samples >= min_samples:
                # Assign all accumulated buckets to current bucket
                for b in buckets_to_merge:
                    new_bucket_map[b] = current_bucket
                # Reset accumulators
                current_bucket += 1
                accumulated_samples = 0
                buckets_to_merge = []

        # Handle remaining buckets
        if buckets_to_merge:
            # If we have a previous bucket, merge with it
            if current_bucket > 0:
                target_bucket = current_bucket - 1
                # Reassign the last bucket's samples to merge with previous
                last_full_bucket = max(b for b in new_bucket_map.values())
                for bucket in buckets_to_merge:
                    new_bucket_map[bucket] = last_full_bucket
            else:
                # If no previous bucket, create new one
                target_bucket = 0
                for bucket in buckets_to_merge:
                    new_bucket_map[bucket] = target_bucket

        # Verify all buckets are mapped
        all_buckets = set(range(int(bucket_counts.index.max() + 1)))
        unmapped_buckets = all_buckets - set(new_bucket_map.keys())
        if unmapped_buckets:
            last_bucket = max(new_bucket_map.values())
            for bucket in unmapped_buckets:
                new_bucket_map[bucket] = last_bucket

        # Apply the new mapping to the DataFrame
        df['bucket'] = df['bucket'].map(lambda x: new_bucket_map.get(int(x), 0))

        # Recalculate width based on final buckets
        final_bucket_counts = df['bucket'].value_counts()
        n_final_buckets = len(final_bucket_counts)
        final_bucket_width = (max_val - min_val) / max(1, n_final_buckets)

        # Verify minimum samples requirement is met
        final_counts = df['bucket'].value_counts()
        if any(count < min_samples for count in final_counts):
            print(f"Warning: Some buckets still have fewer than {min_samples} samples")
            print("Final bucket counts:", final_counts.to_dict())

        return df, final_bucket_width

    @staticmethod
    def calculate_bucket_stats(df, bucket_column='bucket', metrics=None):
        """Calculate statistics for each bucket
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with bucket column
        bucket_column : str, default='bucket'
            Name of the bucket column
        metrics : dict, optional
            Dictionary of metrics to calculate for each bucket
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with calculated statistics
        """
        if metrics is None:
            return df.groupby(bucket_column).size().reset_index(name='count')

        stats = {}
        for metric_name, metric_info in metrics.items():
            if isinstance(metric_info, dict):
                required_cols = metric_info.get('columns', [])
                metric_func = metric_info.get('func')

                if metric_func and all(col in df.columns for col in required_cols):
                    try:
                        stats[metric_name] = df.groupby(bucket_column).apply(metric_func)
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {str(e)}")
                        stats[metric_name] = df.groupby(bucket_column).size() * 0
            else:
                try:
                    stats[metric_name] = df.groupby(bucket_column)[metric_info].agg([
                        'mean', 'min', 'max', 'count'])
                except Exception as e:
                    print(f"Error calculating {metric_name}: {str(e)}")
                    stats[metric_name] = pd.Series(0, index=df[bucket_column].unique())

        result = pd.DataFrame(stats).reset_index()

        # Add sample count if not already included
        if 'count' not in result.columns:
            counts = df.groupby(bucket_column).size().reset_index(name='count')
            result = result.merge(counts, on=bucket_column)

        return result