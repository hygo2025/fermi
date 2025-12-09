import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


class RecBoleConverter:
    """Converte dados parquet para formato RecBole"""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def log(self, message: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def convert_slice(self, slice_id: int):
        """Converte um slice específico para formato RecBole"""
        self.log(f"Converting slice {slice_id}...")
        
        slice_dir = self.input_path / f"slice_{slice_id}"
        train_path = slice_dir / "train"
        test_path = slice_dir / "test"
        
        # Load train and test
        self.log(f"  Loading train data...")
        train_df = pd.read_parquet(train_path)
        
        self.log(f"  Loading test data...")
        test_df = pd.read_parquet(test_path)
        
        # Create output directory
        dataset_name = f"realestate_slice{slice_id}"
        output_dir = self.output_path / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save with correct naming (dataset_name.*.inter)
        # RecBole expects: {data_path}/{dataset_name}/{dataset_name}.inter
        self.log(f"  Processing train data...")
        train_inter = self._prepare_interactions(train_df)
        self._save_inter_file(train_inter, output_dir / f"{dataset_name}.train.inter")
        
        self.log(f"  Processing test data...")
        test_inter = self._prepare_interactions(test_df)
        self._save_inter_file(test_inter, output_dir / f"{dataset_name}.test.inter")
        
        # Create combined .inter file (required by RecBole)
        self.log(f"  Creating combined .inter...")
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined_inter = self._prepare_interactions(combined)
        self._save_inter_file(combined_inter, output_dir / f"{dataset_name}.inter")
        
        self.log(f"  Saved to {output_dir}")
        
    def _prepare_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara DataFrame para formato RecBole .inter"""
        # Select only needed columns
        inter = df[['session_id', 'item_id', 'timestamp']].copy()
        
        # Convert timestamp to Unix timestamp (float)
        inter['timestamp'] = inter['timestamp'].astype('int64') / 1e9
        
        # Ensure correct types
        inter['session_id'] = inter['session_id'].astype(str)
        inter['item_id'] = inter['item_id'].astype(int)
        
        # Sort by session and timestamp
        inter = inter.sort_values(['session_id', 'timestamp']).reset_index(drop=True)
        
        return inter
    
    def _save_inter_file(self, df: pd.DataFrame, output_path: Path):
        """Salva DataFrame como arquivo .inter do RecBole"""
        # Write header + data
        with open(output_path, 'w') as f:
            # RecBole format: field_name:type
            f.write("session_id:token\titem_id:token\ttimestamp:float\n")
            
            # Write data (tab-separated)
            for _, row in df.iterrows():
                f.write(f"{row['session_id']}\t{row['item_id']}\t{row['timestamp']}\n")
        
        size_mb = output_path.stat().st_size / 1024 / 1024
        self.log(f"    Saved {len(df):,} interactions ({size_mb:.1f} MB) to {output_path.name}")
    
    def run(self):
        """Converte todos os slices"""
        self.log("="*80)
        self.log("RecBole Data Converter")
        self.log("="*80)
        
        # Find all slices
        slices = sorted([
            int(p.name.split('_')[1]) 
            for p in self.input_path.glob('slice_*') 
            if p.is_dir()
        ])
        
        self.log(f"Found {len(slices)} slices: {slices}")
        print()
        
        for slice_id in slices:
            self.convert_slice(slice_id)
            print()
        
        self.log("="*80)
        self.log("Conversion completed!")
        self.log("="*80)
        
        # Print summary
        print("\nOutput structure:")
        for slice_id in slices:
            output_dir = self.output_path / f"realestate_slice{slice_id}"
            print(f"  {output_dir}/")
            for f in sorted(output_dir.glob("*.inter")):
                size = f.stat().st_size / 1024 / 1024
                print(f"    {f.name} ({size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Convert sliding window data to RecBole format')
    parser.add_argument('--input', type=str,
                       default='data/sliding_window',
                       help='Input directory with slices')
    parser.add_argument('--output', type=str,
                       default='recbole_data',
                       help='Output directory for RecBole data')
    
    args = parser.parse_args()
    
    converter = RecBoleConverter(
        input_path=args.input,
        output_path=args.output
    )
    
    converter.run()


if __name__ == '__main__':
    main()
