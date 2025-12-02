"""
Data Preparation for Session-Rec Framework

This module prepares real estate browsing data for session-based recommendation.
Converts raw data into the format required by session-rec framework.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys


class SessionDataPreparator:
    """Prepare session data for session-rec framework."""
    
    def __init__(self, input_path, output_path, min_session_length=2, min_item_support=5):
        """
        Initialize data preparator.
        
        Args:
            input_path: Path to raw data file (parquet/csv)
            output_path: Path to save processed data
            min_session_length: Minimum number of events per session
            min_item_support: Minimum number of times an item must appear
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.min_session_length = min_session_length
        self.min_item_support = min_item_support
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self):
        """Load raw data from file."""
        print(f"\nLoading raw data from: {self.input_path}")
        
        if self.input_path.suffix == '.parquet':
            df = pd.read_parquet(self.input_path)
        elif self.input_path.suffix == '.csv':
            df = pd.read_csv(self.input_path)
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        
        print(f"Loaded {len(df):,} events")
        return df
    
    def filter_by_date(self, df, start_date=None, end_date=None):
        """Filter data by date range."""
        if start_date is None and end_date is None:
            return df
        
        print("\nFiltering by date range...")
        
        # Ensure Time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'])
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['Time'] >= start_date]
            print(f"  Start date: {start_date}")
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['Time'] <= end_date]
            print(f"  End date: {end_date}")
        
        print(f"  Remaining events: {len(df):,}")
        return df
    
    def filter_sessions(self, df):
        """Filter sessions by minimum length."""
        print(f"\nFiltering sessions (min length: {self.min_session_length})...")
        
        session_lengths = df.groupby('SessionId').size()
        valid_sessions = session_lengths[session_lengths >= self.min_session_length].index
        
        print(f"  Sessions before: {len(session_lengths):,}")
        print(f"  Sessions after: {len(valid_sessions):,}")
        
        df = df[df['SessionId'].isin(valid_sessions)]
        print(f"  Events after: {len(df):,}")
        
        return df
    
    def filter_items(self, df):
        """Filter items by minimum support."""
        print(f"\nFiltering items (min support: {self.min_item_support})...")
        
        item_counts = df.groupby('ItemId').size()
        valid_items = item_counts[item_counts >= self.min_item_support].index
        
        print(f"  Items before: {len(item_counts):,}")
        print(f"  Items after: {len(valid_items):,}")
        
        df = df[df['ItemId'].isin(valid_items)]
        print(f"  Events after: {len(df):,}")
        
        return df
    
    def create_splits(self, df, train_ratio=0.8):
        """Create train/test split by time."""
        print("\nCreating train/test split...")
        
        # Sort by time
        df = df.sort_values('Time')
        
        # Find split point
        n_sessions = df['SessionId'].nunique()
        split_idx = int(n_sessions * train_ratio)
        
        sessions = df['SessionId'].unique()
        train_sessions = set(sessions[:split_idx])
        
        train = df[df['SessionId'].isin(train_sessions)]
        test = df[~df['SessionId'].isin(train_sessions)]
        
        print(f"  Train: {len(train):,} events, {len(train_sessions):,} sessions")
        print(f"  Test: {len(test):,} events, {test['SessionId'].nunique():,} sessions")
        
        return train, test
    
    def save_session_rec_format(self, train, test, prefix='dataset'):
        """Save data in session-rec format (TSV files)."""
        print("\nSaving in session-rec format...")
        
        # Ensure required columns
        required_cols = ['SessionId', 'ItemId', 'Time']
        for col in required_cols:
            if col not in train.columns or col not in test.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Select and order columns
        train = train[required_cols].copy()
        test = test[required_cols].copy()
        
        # Save as TSV
        train_path = self.output_path / f"{prefix}_train.txt"
        test_path = self.output_path / f"{prefix}_test.txt"
        
        train.to_csv(train_path, sep='\t', index=False)
        test.to_csv(test_path, sep='\t', index=False)
        
        print(f"  Train saved to: {train_path}")
        print(f"  Test saved to: {test_path}")
        
        # Save summary
        summary = {
            'train': {
                'events': len(train),
                'sessions': train['SessionId'].nunique(),
                'items': train['ItemId'].nunique(),
                'date_range': f"{train['Time'].min()} to {train['Time'].max()}"
            },
            'test': {
                'events': len(test),
                'sessions': test['SessionId'].nunique(),
                'items': test['ItemId'].nunique(),
                'date_range': f"{test['Time'].min()} to {test['Time'].max()}"
            }
        }
        
        return summary
    
    def prepare(self, start_date=None, end_date=None, train_ratio=0.8, prefix='dataset'):
        """
        Full data preparation pipeline.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            train_ratio: Ratio of sessions for training
            prefix: Prefix for output files
        
        Returns:
            summary: Dictionary with preparation summary
        """
        print("\n" + "="*80)
        print("DATA PREPARATION PIPELINE")
        print("="*80)
        
        # Load data
        df = self.load_raw_data()
        
        # Filter by date
        df = self.filter_by_date(df, start_date, end_date)
        
        # Filter sessions and items
        df = self.filter_sessions(df)
        df = self.filter_items(df)
        
        # Iteratively filter until stable
        prev_len = 0
        iteration = 0
        while len(df) != prev_len:
            iteration += 1
            prev_len = len(df)
            print(f"\nIteration {iteration} - Refining filters...")
            df = self.filter_sessions(df)
            df = self.filter_items(df)
        
        # Create splits
        train, test = self.create_splits(df, train_ratio)
        
        # Save in session-rec format
        summary = self.save_session_rec_format(train, test, prefix)
        
        print("\n" + "="*80)
        print("PREPARATION SUMMARY")
        print("="*80)
        print(f"Train set:")
        print(f"  Events: {summary['train']['events']:,}")
        print(f"  Sessions: {summary['train']['sessions']:,}")
        print(f"  Items: {summary['train']['items']:,}")
        print(f"  Date range: {summary['train']['date_range']}")
        print(f"\nTest set:")
        print(f"  Events: {summary['test']['events']:,}")
        print(f"  Sessions: {summary['test']['sessions']:,}")
        print(f"  Items: {summary['test']['items']:,}")
        print(f"  Date range: {summary['test']['date_range']}")
        print("="*80)
        
        return summary


if __name__ == '__main__':
    # Example usage
    preparator = SessionDataPreparator(
        input_path='/home/hygo2025/Documents/data/processed_data/sessions.parquet',
        output_path='./session_rec_format',
        min_session_length=2,
        min_item_support=5
    )
    
    preparator.prepare(
        start_date='2024-02-29',
        end_date='2024-03-14',
        prefix='realestate'
    )
