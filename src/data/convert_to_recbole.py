import pandas as pd
from pathlib import Path


def convert_to_recbole():
    """Convert session-rec format to RecBole format."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / "session_rec_format" / "realestate"
    output_dir = base_dir / "recbole_data" / "realestate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train and test
    print(f"Loading {input_dir / 'realestate_train_full.parquet'}...")
    train_df = pd.read_parquet(input_dir / "realestate_train_full.parquet")
    
    print(f"Loading {input_dir / 'realestate_test.parquet'}...")
    test_df = pd.read_parquet(input_dir / "realestate_test.parquet")
    
    # Combine
    print("Combining train and test...")
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # RecBole expects: session_id, item_id, timestamp
    # Rename columns to match RecBole format
    df = df.rename(columns={
        'SessionId': 'session_id:token',
        'ItemId': 'item_id:token',
        'Time': 'timestamp:float'
    })
    
    # Select only needed columns
    df = df[['session_id:token', 'item_id:token', 'timestamp:float']]
    
    # Sort by session and timestamp
    df = df.sort_values(['session_id:token', 'timestamp:float'])
    
    # Save to .inter file (tab-separated)
    output_file = output_dir / "realestate.inter"
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, sep='\t', index=False)
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Sessions: {df['session_id:token'].nunique():,}")
    print(f"Items: {df['item_id:token'].nunique():,}")
    print(f"Interactions: {len(df):,}")
    print(f"\nOutput: {output_file}")
    print("="*60)


if __name__ == "__main__":
    convert_to_recbole()
