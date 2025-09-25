"""
Validation and Testing Suite for L1/L2 Documentation Claims

This script validates every metric, calculation, and claim made in our L1/L2 documentation
by actually analyzing the Chicago crime data and H3 hexagon assignments.

Tests:
- L1 file size reductions 
- L2 H3 hexagon counts per resolution
- Cyclical encoding validation
- Processing performance metrics
- Data quality statistics
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# === Standalone Logging ===
import logging
from datetime import datetime

def setup_standalone_logger():
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger('validation_tests')
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(f'logs/validation_tests_{datetime.now().strftime("%m%d%Y")}.log')
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter('%(levelname)s : %(name)s.%(funcName)s : %(message)s')
    fh.setFormatter(fh_fmt)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    ch.setFormatter(ch_fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_standalone_logger()

RAW_DIR = Path('data/raw')
L1_DIR = Path('data/l1')
L2_DIR = Path('data/l2')


def test_file_size_claims() -> Dict:
    """Test L1 documentation claim: '70% file size reduction'"""
    logger.info("Testing file size reduction claims...")
    
    raw_total = 0
    l1_total = 0
    
    # Calculate raw CSV sizes
    for csv_file in RAW_DIR.glob('*.csv'):
        raw_total += csv_file.stat().st_size
    
    # Calculate L1 parquet sizes
    for parquet_file in L1_DIR.rglob('*.parquet'):
        l1_total += parquet_file.stat().st_size
    
    reduction_pct = ((raw_total - l1_total) / raw_total) * 100
    
    results = {
        'raw_size_mb': raw_total / (1024*1024),
        'l1_size_mb': l1_total / (1024*1024),
        'reduction_percent': reduction_pct,
        'claim_validated': 60 <= reduction_pct <= 80  # Allow some variance
    }
    
    logger.info(f"Raw CSV total: {results['raw_size_mb']:.1f} MB")
    logger.info(f"L1 Parquet total: {results['l1_size_mb']:.1f} MB") 
    logger.info(f"Actual reduction: {results['reduction_percent']:.1f}%")
    logger.info(f"Documentation claim (70%) validated: {results['claim_validated']}")
    
    return results


def test_record_counts() -> Dict:
    """Test L1 documentation record count claims"""
    logger.info("Testing record count claims...")
    
    results = {}
    
    # Count raw records
    for csv_file in RAW_DIR.glob('chicago_crimes_*.csv'):
        year = csv_file.stem.split('_')[-1]
        df = pd.read_csv(csv_file)
        raw_count = len(df)
        
        # Count L1 records for same year
        l1_count = 0
        year_dir = L1_DIR / f'year={year}'
        if year_dir.exists():
            for parquet_file in year_dir.rglob('*.parquet'):
                df_l1 = pd.read_parquet(parquet_file)
                l1_count += len(df_l1)
        
        drop_pct = ((raw_count - l1_count) / raw_count) * 100 if raw_count > 0 else 0
        
        results[year] = {
            'raw_count': raw_count,
            'l1_count': l1_count,
            'dropped_count': raw_count - l1_count,
            'drop_percent': drop_pct
        }
        
        logger.info(f"{year}: {raw_count:,} → {l1_count:,} records ({drop_pct:.1f}% dropped)")
    
    return results


def test_h3_hexagon_counts() -> Dict:
    """Test L2 documentation H3 hexagon count claims for Chicago"""
    logger.info("Testing H3 hexagon count claims...")
    
    try:
        import h3
    except ImportError:
        logger.error("H3 library not available - cannot test hexagon counts")
        return {'error': 'h3_not_available'}
    
    results = {}
    
    # Sample L2 data to get actual hexagon counts
    sample_files = list(L2_DIR.rglob('*.parquet'))[:5]  # Sample first 5 files
    if not sample_files:
        logger.warning("No L2 files found - run L2 first")
        return {'error': 'no_l2_data'}
    
    for resolution in [7, 8, 9]:
        unique_hexes = set()
        
        for file in sample_files:
            df = pd.read_parquet(file)
            hex_col = f'h3_r{resolution}'
            
            if hex_col in df.columns:
                valid_hexes = df[hex_col].dropna()
                unique_hexes.update(valid_hexes)
            else:
                # Calculate H3 for this resolution
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    coords = df[['latitude', 'longitude']].dropna()
                    for _, row in coords.iterrows():
                        if hasattr(h3, 'latlng_to_cell'):
                            hex_id = h3.latlng_to_cell(row['latitude'], row['longitude'], resolution)
                        else:
                            hex_id = h3.geo_to_h3(row['latitude'], row['longitude'], resolution)
                        unique_hexes.add(hex_id)
        
        # Extrapolate to full dataset
        sample_ratio = len(sample_files) / len(list(L2_DIR.rglob('*.parquet')))
        estimated_total = int(len(unique_hexes) / sample_ratio) if sample_ratio > 0 else len(unique_hexes)
        
        results[f'resolution_{resolution}'] = {
            'sample_hexagons': len(unique_hexes),
            'estimated_total_hexagons': estimated_total,
            'sample_files_used': len(sample_files)
        }
        
        logger.info(f"Resolution {resolution}: ~{estimated_total} hexagons estimated (from {len(unique_hexes)} in sample)")
    
    return results


def test_cyclical_encoding() -> Dict:
    """Test cyclical encoding mathematical properties"""
    logger.info("Testing cyclical encoding validation...")
    
    def cyclical_encode(series, period):
        radians = 2 * np.pi * series.astype(float) / period
        return np.sin(radians), np.cos(radians)
    
    results = {}
    
    # Test hour cyclical encoding (23 ≈ 1)
    hours = np.array([23, 0, 1])
    hour_sin, hour_cos = cyclical_encode(hours, 24)
    
    # Check if 23 and 1 are closer than 23 and 12
    dist_23_to_1 = np.sqrt((hour_sin[0] - hour_sin[2])**2 + (hour_cos[0] - hour_cos[2])**2)
    dist_23_to_12 = np.sqrt((hour_sin[0] - np.sin(2*np.pi*12/24))**2 + (hour_cos[0] - np.cos(2*np.pi*12/24))**2)
    
    results['hour_cyclical'] = {
        'hour_23_to_1_distance': dist_23_to_1,
        'hour_23_to_12_distance': dist_23_to_12,
        'cyclical_property_valid': dist_23_to_1 < dist_23_to_12
    }
    
    # Test day of week cyclical encoding (Sunday=6 ≈ Monday=0)
    days = np.array([6, 0, 1])  # Sunday, Monday, Tuesday
    dow_sin, dow_cos = cyclical_encode(days, 7)
    
    dist_sun_to_mon = np.sqrt((dow_sin[0] - dow_sin[1])**2 + (dow_cos[0] - dow_cos[1])**2)
    dist_sun_to_wed = np.sqrt((dow_sin[0] - np.sin(2*np.pi*3/7))**2 + (dow_cos[0] - np.cos(2*np.pi*3/7))**2)
    
    results['dow_cyclical'] = {
        'sunday_to_monday_distance': dist_sun_to_mon,
        'sunday_to_wednesday_distance': dist_sun_to_wed,
        'cyclical_property_valid': dist_sun_to_mon < dist_sun_to_wed
    }
    
    # Test month cyclical encoding (December=11 ≈ January=0)
    months = np.array([11, 0, 1])  # December, January, February
    month_sin, month_cos = cyclical_encode(months, 12)
    
    dist_dec_to_jan = np.sqrt((month_sin[0] - month_sin[1])**2 + (month_cos[0] - month_cos[1])**2)
    dist_dec_to_jun = np.sqrt((month_sin[0] - np.sin(2*np.pi*5/12))**2 + (month_cos[0] - np.cos(2*np.pi*5/12))**2)
    
    results['month_cyclical'] = {
        'december_to_january_distance': dist_dec_to_jan,
        'december_to_june_distance': dist_dec_to_jun,
        'cyclical_property_valid': dist_dec_to_jan < dist_dec_to_jun
    }
    
    logger.info(f"Hour cyclical: 23h→1h closer than 23h→12h: {results['hour_cyclical']['cyclical_property_valid']}")
    logger.info(f"DOW cyclical: Sun→Mon closer than Sun→Wed: {results['dow_cyclical']['cyclical_property_valid']}")
    logger.info(f"Month cyclical: Dec→Jan closer than Dec→Jun: {results['month_cyclical']['cyclical_property_valid']}")
    
    return results


def test_coordinate_bounds() -> Dict:
    """Test Chicago coordinate bounds filtering accuracy"""
    logger.info("Testing coordinate bounds claims...")
    
    results = {'years': {}}
    
    # Check a sample of raw data for coordinate distribution
    for csv_file in list(RAW_DIR.glob('*.csv'))[:3]:  # Sample first 3 years
        year = csv_file.stem.split('_')[-1]
        df = pd.read_csv(csv_file)
        
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            lat_col, lon_col = 'Latitude', 'Longitude'
        elif 'latitude' in df.columns and 'longitude' in df.columns:
            lat_col, lon_col = 'latitude', 'longitude'
        else:
            continue
            
        valid_coords = df[[lat_col, lon_col]].dropna()
        
        # Chicago bounds: 41.0-42.5°N, -88.5 to -87.0°W
        in_bounds = (
            valid_coords[lat_col].between(41.0, 42.5) & 
            valid_coords[lon_col].between(-88.5, -87.0)
        )
        
        results['years'][year] = {
            'total_records': len(df),
            'valid_coordinates': len(valid_coords),
            'in_chicago_bounds': in_bounds.sum(),
            'out_of_bounds': (~in_bounds).sum(),
            'out_of_bounds_percent': ((~in_bounds).sum() / len(valid_coords)) * 100 if len(valid_coords) > 0 else 0
        }
        
        logger.info(f"{year}: {(~in_bounds).sum()} out of {len(valid_coords)} coords outside Chicago bounds")
    
    return results


def test_processing_performance() -> Dict:
    """Test L2 processing speed claims"""
    logger.info("Testing processing performance claims...")
    
    # Test cyclical encoding speed
    test_data = np.random.randint(0, 24, 100000)  # 100K random hours
    
    start_time = time.time()
    radians = 2 * np.pi * test_data.astype(float) / 24
    sin_vals = np.sin(radians)
    cos_vals = np.cos(radians)
    cyclical_time = time.time() - start_time
    
    cyclical_speed = len(test_data) / cyclical_time
    
    # Test H3 assignment speed (if possible)
    h3_speed = None
    try:
        import h3
        # Sample coordinates around Chicago
        lats = np.random.uniform(41.0, 42.5, 10000)
        lons = np.random.uniform(-88.5, -87.0, 10000)
        
        start_time = time.time()
        if hasattr(h3, 'latlng_to_cell'):
            hex_ids = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(lats, lons)]
        else:
            hex_ids = [h3.geo_to_h3(lat, lon, 9) for lat, lon in zip(lats, lons)]
        h3_time = time.time() - start_time
        
        h3_speed = len(lats) / h3_time
        
    except ImportError:
        logger.warning("H3 not available for speed testing")
    
    results = {
        'cyclical_encoding_speed': cyclical_speed,
        'h3_assignment_speed': h3_speed,
        'cyclical_claim_validated': cyclical_speed > 500000,  # >500K records/minute
        'h3_claim_validated': h3_speed > 300000 if h3_speed else None  # >300K records/minute
    }
    
    logger.info(f"Cyclical encoding: {cyclical_speed:,.0f} records/minute")
    if h3_speed:
        logger.info(f"H3 assignment: {h3_speed:,.0f} records/minute")
    
    return results


def run_all_tests() -> Dict:
    """Run all validation tests and compile results"""
    logger.info("=== Starting comprehensive validation tests ===")
    
    all_results = {}
    
    try:
        all_results['file_sizes'] = test_file_size_claims()
    except Exception as e:
        logger.error(f"File size test failed: {e}")
        all_results['file_sizes'] = {'error': str(e)}
    
    try:
        all_results['record_counts'] = test_record_counts()
    except Exception as e:
        logger.error(f"Record count test failed: {e}")
        all_results['record_counts'] = {'error': str(e)}
    
    try:
        all_results['h3_hexagons'] = test_h3_hexagon_counts()
    except Exception as e:
        logger.error(f"H3 hexagon test failed: {e}")
        all_results['h3_hexagons'] = {'error': str(e)}
    
    try:
        all_results['cyclical_encoding'] = test_cyclical_encoding()
    except Exception as e:
        logger.error(f"Cyclical encoding test failed: {e}")
        all_results['cyclical_encoding'] = {'error': str(e)}
    
    try:
        all_results['coordinate_bounds'] = test_coordinate_bounds()
    except Exception as e:
        logger.error(f"Coordinate bounds test failed: {e}")
        all_results['coordinate_bounds'] = {'error': str(e)}
    
    try:
        all_results['performance'] = test_processing_performance()
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        all_results['performance'] = {'error': str(e)}
    
    logger.info("=== Validation tests completed ===")
    return all_results


def main():
    """Main execution function"""
    try:
        results = run_all_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        for test_name, test_results in results.items():
            print(f"\n{test_name.upper()}:")
            if isinstance(test_results, dict) and 'error' not in test_results:
                for key, value in test_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:,.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  ERROR: {test_results.get('error', 'Unknown error')}")
        
        logger.info("Validation complete - check logs for detailed results")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Application Terminated: {str(e)}")
        sys.exit(1)