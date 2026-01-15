#!/usr/bin/env python3
"""
JPEG Pleno Processor - Complete workflow (Steps 1-6)
Steps 1-3: Encoding (Preprocessing + PPM to PGX + Encoding)
Steps 4-6: Decoding (Decoding + PGX to PPM + Postprocessing)
Usage: jpl_processor.py [-h] --lenslet | --no-lenslet [--steps STEPS] [--lambda LAMBDA_VALUE] dataset_name
"""

from math import sqrt
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time
from PIL import Image

class JPEGPlenoProcessor:
    def __init__(self, jplm_build_path, base_datasets_path):
        self.jplm_build_path = Path(jplm_build_path)
        self.base_datasets_path = Path(base_datasets_path)
        self.bins = self.jplm_build_path / "../bin"
        
        if not self.bins.exists():
            raise FileNotFoundError(f"JPLM binaries not found at {self.bins}")

    def run_command(self, cmd, description=""):
        """Esegue un comando shell con timing"""
        print(f"[RUNNING] {description}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                check=True,
                text=True
            )
            end_time = time.time()
            execution_time = end_time - start_time
            
            print("-" * 60)
            print(f"[SUCCESS] Completed in {execution_time:.2f}s")
            return True, execution_time
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print("-" * 60)
            print(f"[ERROR] Process returned {e.returncode} after {execution_time:.2f}s")
            return False, execution_time
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print("-" * 60)
            print(f"[ERROR] Exception after {execution_time:.2f}s: {e}")
            return False, execution_time

    def step1_lenslet_preprocessing(self, dataset_name, grid_t, grid_s):
        """STEP 1: Lenslet Preprocessing (encode)"""
        print(f"\n{'='*50}")
        print(f"STEP 1: LENSLET PREPROCESSING - {dataset_name}")
        print(f"{'='*50}")
        
        dataset_base_path = self.base_datasets_path / dataset_name
        originals_input_path = dataset_base_path / "RAW" / "PPM" 
        raw_shifted_ppm_path = dataset_base_path / "RAW" / "PPM_shifted"
        
        originals_input_path_abs = originals_input_path.resolve()
        raw_shifted_ppm_path_abs = raw_shifted_ppm_path.resolve()
        
        print(f"Input:  {originals_input_path_abs}")
        print(f"Output: {raw_shifted_ppm_path_abs}")
        print(f"Grid: {grid_t}x{grid_s}")
        
        raw_shifted_ppm_path_abs.mkdir(parents=True, exist_ok=True)
        
        if not originals_input_path_abs.exists():
            print(f"ERROR: Original PPM path not found: {originals_input_path_abs}")
            return False, 0
        
        original_ppm_files = list(originals_input_path_abs.glob("*.ppm"))
        if not original_ppm_files:
            print(f"ERROR: No PPM files found in {originals_input_path_abs}")
            return False, 0
        
        print(f"Found {len(original_ppm_files)} original PPM files")

        cmd = [
            str(self.bins / "utils" / "lenslet_13x13_shifter"),
            "-i", str(originals_input_path)+"/",
            "-o", str(raw_shifted_ppm_path)+"/",
            "-d", "encode"
        ]
        
        success, exec_time = self.run_command(cmd, "Lenslet preprocessing")
        if not success:
            return False, exec_time
        
        shifted_ppm_files = list(raw_shifted_ppm_path_abs.glob("*.ppm"))
        print(f"Created {len(shifted_ppm_files)} shifted PPM files")
        
        return True, exec_time

    def step2_convert_ppm_to_pgx(self, dataset_name, lenslet=True):
        """STEP 2: Convert shifted PPM to PGX"""
        print(f"\n{'='*50}")
        print(f"STEP 2: CONVERT PPM TO PGX - {dataset_name}")
        print(f"{'='*50}")
        
        dataset_base_path = self.base_datasets_path / dataset_name
        raw_shifted_ppm_path = dataset_base_path / "RAW" / "PPM_shifted" if lenslet else dataset_base_path / "RAW" / "PPM"
        raw_shifted_pgx_path = dataset_base_path / "RAW" / "PGX"
        
        raw_shifted_ppm_path_abs = raw_shifted_ppm_path.resolve()
        raw_shifted_pgx_path_abs = raw_shifted_pgx_path.resolve()
        
        print(f"PPM Input:  {raw_shifted_ppm_path_abs}")
        print(f"PGX Output: {raw_shifted_pgx_path_abs}")
        
        raw_shifted_pgx_path_abs.mkdir(parents=True, exist_ok=True)
        
        if not raw_shifted_ppm_path_abs.exists():
            print(f"ERROR: Shifted PPM path not found: {raw_shifted_ppm_path_abs}")
            print("Run Step 1 first!")
            return False, 0
        
        ppm_files = list(raw_shifted_ppm_path_abs.glob("*.ppm"))
        if not ppm_files:
            print(f"ERROR: No PPM files found in {raw_shifted_ppm_path_abs}")
            return False, 0
        
        print(f"Found {len(ppm_files)} shifted PPM files to convert")

        total_time = 0
        converted_count = 0
        for ppm_file in ppm_files:
            cmd = [
                str(self.bins / "utils" / "convert_ppm_to_pgx"),
                "-i", str(ppm_file),
                "-o", str(raw_shifted_pgx_path_abs) + "/"
            ]
            
            success, exec_time = self.run_command(cmd, f"Converting {ppm_file.name}")
            total_time += exec_time
            if success:
                converted_count += 1
        
        pgx_files = list(raw_shifted_pgx_path_abs.glob("*.pgx"))
        print(f"Created {len(pgx_files)} PGX files")
        print(f"Converted: {converted_count}/{len(ppm_files)} files")
        print(f"Total conversion time: {total_time:.2f}s")
        
        return True, total_time

    #def detect_actual_dimensions(self, pgx_path):
    #    """Auto-rileva dimensioni reali dal primo file PGX"""
    #    try:
    #        pgx_files = list(pgx_path.glob("*.pgx"))
    #        if pgx_files:
    #            first_file = pgx_files[0]
    #            with open(first_file, 'rb') as f:
    #                header = f.readline().decode('ascii')
    #                parts = header.strip().split()
    #                if len(parts) >= 5:
    #                    width = int(parts[3])
    #                    height = int(parts[4])
    #                    return width, height
    #    except Exception as e:
    #        print(f"WARNING: Could not auto-detect dimensions: {e}")
    #    return None, None

    def calculate_optimal_block_size(self, width, height):
        """Calcola la dimensione ottimale del blocco per DCT"""
        return 32
        #min_dimension = min(width, height)
        #
        #if min_dimension >= 1024:
        #    optimal_size = 64
        #elif min_dimension >= 512:
        #    optimal_size = 32
        #elif min_dimension >= 256:
        #    optimal_size = 16
        #else:
        #    optimal_size = 8
        #
        #optimal_size = self.nearest_power_of_two(optimal_size)
        #max_supported = 64
        #return min(optimal_size, max_supported)

    #def nearest_power_of_two(self, n):
    #    """Trova la potenza di 2 più vicina"""
    #    powers = [4, 8, 16, 32, 64]
    #    return min(powers, key=lambda x: abs(x - n))

    def calculate_auto_lambda(self, width, height, grid_t, grid_s):
        """Calcola lambda automaticamente in base alla dimensione totale del light field"""
        return 700
        #total_pixels = width * height * grid_t * grid_s
        #
        #if total_pixels > 100000000:
        #    return 5000
        #elif total_pixels > 50000000:
        #    return 10000
        #else:
        #    return 20000

    def step3_jpl_encoding(self, dataset_name, grid_t, grid_s, width, height, lambda_value=None):
        """STEP 3: JPEG Pleno Encoding con parametri dinamici"""
        print(f"\n{'='*50}")
        print(f"STEP 3: JPEG PLENO ENCODING - {dataset_name}")
        print(f"{'='*50}")
        
        dataset_base_path = self.base_datasets_path / dataset_name
        raw_shifted_pgx_path = dataset_base_path / "RAW" / "PGX"
        
        #actual_width, actual_height = self.detect_actual_dimensions(raw_shifted_pgx_path)
        #if actual_width and actual_height:
        #    width, height = actual_width, actual_height
        #    print(f"Auto-detected dimensions: {width}x{height}")
        
        output_jpl_path = dataset_base_path / "encoded"
        output_jpl_file = output_jpl_path / f"{dataset_name}.jpl"
        
        raw_shifted_pgx_path_abs = raw_shifted_pgx_path.resolve()
        output_jpl_path_abs = output_jpl_path.resolve()
        
        print(f"Encoding Parameters:")
        print(f"  Input PGX:  {raw_shifted_pgx_path_abs}")
        print(f"  Output JPL: {output_jpl_file}")
        print(f"  Grid: {grid_t}x{grid_s}")
        print(f"  Dimensions: {width}x{height}")
        
        if lambda_value is None:
            lambda_value = self.calculate_auto_lambda(width, height, grid_t, grid_s)
            lambda_source = "auto-calculated"
        else:
            lambda_source = "user-specified"
        
        output_jpl_path_abs.mkdir(parents=True, exist_ok=True)
        
        if not raw_shifted_pgx_path_abs.exists():
            print(f"ERROR: Shifted PGX path not found: {raw_shifted_pgx_path_abs}")
            return False, 0

        inter_view_vertical = grid_s
        inter_view_horizontal = grid_t
        intra_view_max = self.calculate_optimal_block_size(width, height)
        intra_view_min = 4
        
        print(f"Calculated Parameters:")
        print(f"  Inter-view: {inter_view_horizontal}x{inter_view_vertical}")
        print(f"  Intra-view: {intra_view_min} to {intra_view_max}")
        print(f"  Lambda: {lambda_value} ({lambda_source})")
        
        cmd = [
            str(self.bins / "jpl-encoder-bin"),
            "--show-progress-bar",
            "--show-runtime-statistics", 
            "--part", "2",
            "--type", "0",
            "--enum-cs", "YCbCr_2",
            "-u", str(width),
            "-v", str(height),  
            "-t", str(grid_t),
            "-s", str(grid_s),
            "-nc", "3",
            "--show-error-estimate",
            "--border_policy", "1",
            "--lambda", str(lambda_value),
            "--transform_size_maximum_inter_view_vertical", str(inter_view_vertical),
            "--transform_size_maximum_inter_view_horizontal", str(inter_view_horizontal),
            "--transform_size_maximum_intra_view_vertical", str(intra_view_max), 
            "--transform_size_maximum_intra_view_horizontal", str(intra_view_max),
            "--transform_size_minimum_inter_view_vertical", str(inter_view_vertical),
            "--transform_size_minimum_inter_view_horizontal", str(inter_view_horizontal), 
            "--transform_size_minimum_intra_view_vertical", str(intra_view_min),
            "--transform_size_minimum_intra_view_horizontal", str(intra_view_min),
            "--input", str(raw_shifted_pgx_path_abs) + "/",
            "--output", str(output_jpl_file)
        ]
        
        success, exec_time = self.run_command(cmd, "JPEG Pleno encoding")
        if not success:
            return False, exec_time
        
        if output_jpl_file.exists():
            file_size = output_jpl_file.stat().st_size
            print(f"Created encoded file: {output_jpl_file}")
            print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            return True, exec_time
        else:
            print(f"ERROR: Encoded file not created: {output_jpl_file}")
            return False, exec_time

    def step4_jpl_decoding(self, dataset_name, lenslet=True):
        """STEP 4: JPEG Pleno Decoding"""
        print(f"\n{'='*50}")
        print(f"STEP 4: JPEG PLENO DECODING - {dataset_name}")
        print(f"{'='*50}")
        
        dataset_base_path = self.base_datasets_path / dataset_name
        
        input_jpl_file = dataset_base_path / "encoded" / f"{dataset_name}.jpl"
        decoded_path = dataset_base_path / "decoded"
        decoded_pgx_shifted = decoded_path / "PGX" / f"{dataset_name}_shifted" if lenslet else decoded_path / "PGX" / dataset_name
        
        input_jpl_file_abs = input_jpl_file.resolve()
        decoded_pgx_shifted_abs = decoded_pgx_shifted.resolve()
        
        print(f"Decoding Parameters:")
        print(f"  Input JPL:  {input_jpl_file_abs}")
        print(f"  Output PGX: {decoded_pgx_shifted_abs}")
        
        decoded_pgx_shifted_abs.mkdir(parents=True, exist_ok=True)
        
        if not input_jpl_file_abs.exists():
            print(f"ERROR: Encoded JPL file not found: {input_jpl_file_abs}")
            print("Run Step 3 first!")
            return False, 0

        cmd = [
            str(self.bins / "jpl-decoder-bin"),
            "--show-progress-bar",
            "--input", str(input_jpl_file_abs),
            "--output", str(decoded_pgx_shifted_abs)
        ]
        
        success, exec_time = self.run_command(cmd, "JPEG Pleno decoding")
        if not success:
            return False, exec_time
        
        pgx_files = list(decoded_pgx_shifted_abs.glob("**/*.pgx"))
        print(f"Created {len(pgx_files)} decoded PGX files")
        
        return True, exec_time

    def step5_convert_pgx_to_ppm(self, dataset_name, lenslet=True):
        """STEP 5: Convert decoded PGX to PPM"""
        print(f"\n{'='*50}")
        print(f"STEP 5: CONVERT PGX TO PPM - {dataset_name}")
        print(f"{'='*50}")
        
        dataset_base_path = self.base_datasets_path / dataset_name
        
        decoded_pgx_shifted = dataset_base_path / "decoded" / "PGX" / f"{dataset_name}_shifted" if lenslet else dataset_base_path / "decoded" / "PGX" / dataset_name
        decoded_ppm_shifted = dataset_base_path / "decoded" / "PPM" / f"{dataset_name}_shifted" if lenslet else dataset_base_path / "decoded" / "PPM" / dataset_name
        
        decoded_pgx_shifted_abs = decoded_pgx_shifted.resolve()
        decoded_ppm_shifted_abs = decoded_ppm_shifted.resolve()
        
        print(f"Conversion Parameters:")
        print(f"  PGX Input:  {decoded_pgx_shifted_abs}")
        print(f"  PPM Output: {decoded_ppm_shifted_abs}")
        
        decoded_ppm_shifted_abs.mkdir(parents=True, exist_ok=True)
        
        if not decoded_pgx_shifted_abs.exists():
            print(f"ERROR: Decoded PGX path not found: {decoded_pgx_shifted_abs}")
            print("Run Step 4 first!")
            return False, 0
        
        pgx_files_path = decoded_pgx_shifted_abs / "0"
        if not pgx_files_path.exists():
            pgx_files_path = decoded_pgx_shifted_abs
        
        pgx_files = list(pgx_files_path.glob("*.pgx"))
        if not pgx_files:
            print(f"ERROR: No PGX files found in {pgx_files_path}")
            return False, 0
        
        print(f"Found {len(pgx_files)} decoded PGX files to convert")

        total_time = 0
        converted_count = 0
        for pgx_file in pgx_files:
            current_view_filename = f"{pgx_file.stem}.ppm"
            ppm_output_path = decoded_ppm_shifted_abs / current_view_filename
            
            cmd = [
                str(self.bins / "utils" / "convert_pgx_to_ppm"),
                "--input", str(decoded_pgx_shifted_abs),
                "--output", str(ppm_output_path)
            ]
            
            success, exec_time = self.run_command(cmd, f"Converting {pgx_file.name}")
            total_time += exec_time
            if success:
                converted_count += 1
        
        ppm_files = list(decoded_ppm_shifted_abs.glob("*.ppm"))
        print(f"Created {len(ppm_files)} PPM files")
        print(f"Converted: {converted_count}/{len(pgx_files)} files")
        print(f"Total conversion time: {total_time:.2f}s")
        
        return True, total_time

    def step6_lenslet_postprocessing(self, dataset_name):
        """STEP 6: Lenslet Postprocessing (decode)"""
        print(f"\n{'='*50}")
        print(f"STEP 6: LENSLET POSTPROCESSING - {dataset_name}")
        print(f"{'='*50}")
        
        dataset_base_path = self.base_datasets_path / dataset_name
        
        decoded_ppm_shifted = dataset_base_path / "decoded" / "PPM" / f"{dataset_name}_shifted"
        decoded_ppm_path = dataset_base_path / "decoded" / "PPM" / dataset_name
        
        decoded_ppm_shifted_abs = decoded_ppm_shifted.resolve()
        decoded_ppm_path_abs = decoded_ppm_path.resolve()
        
        print(f"Postprocessing Parameters:")
        print(f"  Input (shifted):  {decoded_ppm_shifted_abs}")
        print(f"  Output (final):   {decoded_ppm_path_abs}")
        
        decoded_ppm_path_abs.mkdir(parents=True, exist_ok=True)
        
        if not decoded_ppm_shifted_abs.exists():
            print(f"ERROR: Decoded shifted PPM path not found: {decoded_ppm_shifted_abs}")
            print("Run Step 5 first!")
            return False, 0
        
        ppm_files = list(decoded_ppm_shifted_abs.glob("*.ppm"))
        if not ppm_files:
            print(f"ERROR: No PPM files found in {decoded_ppm_shifted_abs}")
            return False, 0
        
        print(f"Found {len(ppm_files)} shifted PPM files to process")

        cmd = [
            str(self.bins / "utils" / "lenslet_13x13_shifter"),
            "--verbose",
            "-i", str(decoded_ppm_shifted_abs) + "/",
            "-o", str(decoded_ppm_path_abs) + "/",
            "-d", "decode"
        ]
        
        success, exec_time = self.run_command(cmd, "Lenslet postprocessing")
        if not success:
            return False, exec_time
        
        final_ppm_files = list(decoded_ppm_path_abs.glob("*.ppm"))
        print(f"Created {len(final_ppm_files)} final PPM files")
        
        return True, exec_time

    def process_dataset(self, dataset_name, grid_t, grid_s, width, height, steps=None, lambda_value=None, lenslet=True):
        """Processa tutto il workflow Step 1-6 con timing"""
        if steps is None:
            steps = [1, 2, 3, 4, 5, 6]
        
        print(f"\n{'='*80}")
        print(f"STARTING JPEG PLENO PROCESSING - {dataset_name}")
        print(f"Grid: {grid_t}x{grid_s}, Dimensions: {width}x{height}")
        if lambda_value is not None:
            print(f"Lambda: {lambda_value} (user-specified)")
        print(f"Steps to execute: {steps}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        step_times = {}
        total_start_time = time.time()
        
        # ENCODING PIPELINE
        if 1 in steps and lenslet:
            success, time_step1 = self.step1_lenslet_preprocessing(dataset_name, grid_t, grid_s)
            step_times[1] = time_step1
            if not success:
                print(f"STEP 1 FAILED - stopping")
                return False, step_times
        
        if 2 in steps:
            success, time_step2 = self.step2_convert_ppm_to_pgx(dataset_name, lenslet)
            step_times[2] = time_step2
            if not success:
                print(f"STEP 2 FAILED - stopping")
                return False, step_times
        
        if 3 in steps:
            success, time_step3 = self.step3_jpl_encoding(dataset_name, grid_t, grid_s, width, height, lambda_value)
            step_times[3] = time_step3
            if not success:
                print(f"STEP 3 FAILED - stopping")
                return False, step_times
        
        # DECODING PIPELINE
        if 4 in steps:
            success, time_step4 = self.step4_jpl_decoding(dataset_name, lenslet)
            step_times[4] = time_step4
            if not success:
                print(f"STEP 4 FAILED - stopping")
                return False, step_times
        
        if 5 in steps:
            success, time_step5 = self.step5_convert_pgx_to_ppm(dataset_name, lenslet)
            step_times[5] = time_step5
            if not success:
                print(f"STEP 5 FAILED - stopping")
                return False, step_times
        
        if 6 in steps and lenslet:
            success, time_step6 = self.step6_lenslet_postprocessing(dataset_name)
            step_times[6] = time_step6
            if not success:
                print(f"STEP 6 FAILED - stopping")
                return False, step_times
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\nPROCESSING COMPLETED for {dataset_name}!")
        print(f"Final output: {self.base_datasets_path / dataset_name / 'decoded' / 'PPM' / dataset_name}")
        
        # Summary timing
        print(f"\nTIMING SUMMARY:")
        print(f"{'Step':<6} {'Time (s)':<10} {'% of Total':<12}")
        print(f"{'-'*6} {'-'*10} {'-'*12}")
        for step in sorted(step_times.keys()):
            time_val = step_times[step]
            percentage = (time_val / total_time) * 100 if total_time > 0 else 0
            print(f"{step:<6} {time_val:<10.2f} {percentage:<11.1f}%")
        
        print(f"{'Total':<6} {total_time:<10.2f} {'100.0':<11}%")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True, step_times

def get_ppm_dimensions(folder_path):
    """
    Trova il primo file .ppm in una cartella e restituisce (larghezza, altezza).
    Richiede la libreria Pillow.
    """
    # 1. Cerca i file .ppm
    files = list(folder_path.glob("*.ppm"))
    
    if not files:
        print("Nessun file .ppm trovato nella cartella.")
        return None
    
    # 2. Apre l'immagine e legge le dimensioni
    try:
        with Image.open(files[0]) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Errore nella lettura del file {files[0]}: {e}")
        return None
    
def get_grid_dimensions(folder_path):
    """
    Calcola le dimensioni della griglia (t, s) basate sul numero di file .ppm nella cartella.
    Assumiamo che i file rappresentino una griglia completa.
    """
    
    files = list(folder_path.glob("*.ppm"))
    grid_dim = sqrt(len(files))
    return int(grid_dim), int(grid_dim)

def main():
    parser = argparse.ArgumentParser(
        description="JPEG Pleno Processor - Complete workflow (Steps 1-6) with dynamic parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete workflow with auto parameters
  python jpl_processor.py Bikes --lenslet

  # Custom lambda value (higher = more compression, lower = higher quality)
  python jpl_processor.py Bikes --lenslet --lambda 1000

  # Only encoding steps with custom lambda
  python jpl_processor.py Bikes --lenslet --steps 1,2,3 --lambda 20000

Lambda values guide:
  - 1000-5000:    Low compression, excellent quality
  - 5000-20000:   Medium compression, medium quality
  - 20000-50000:  High compression, low quality
  - auto:         Default value 700
        """
    )
    
    parser.add_argument("dataset_name", help="Name of the dataset folder (e.g., bikes, cars)")
    #parser.add_argument("grid_t", type=int, help="Number of horizontal views (t)")
    #parser.add_argument("grid_s", type=int, help="Number of vertical views (s)") 
    parser.add_argument("--lenslet", action=argparse.BooleanOptionalAction, help="Whether the dataset is of the lenslet type. If it isn't, steps 1 and 6 are skipped.", required=True)
    #parser.add_argument("width", type=int, help="Image width in pixels")
    #parser.add_argument("height", type=int, help="Image height in pixels")
    parser.add_argument("--steps", help="Comma-separated steps to execute (1-6), e.g., '1,2,3' or '4,5,6'")
    parser.add_argument("--lambda", type=float, dest="lambda_value", help="Lambda parameter for quality/compression trade-off")
    
    args = parser.parse_args()
    
    # Configurazione paths
    JPLM_BUILD_PATH = "/home/losor2002/jpeg-pleno-refsw/build"
    BASE_DATASETS_PATH = "./datasets"
    
    dataset_base_path = Path(BASE_DATASETS_PATH) / args.dataset_name
    if not dataset_base_path.exists():
        print(f"Folder not found: {dataset_base_path}")
        sys.exit(1)
    
    jplm_build_path = Path(JPLM_BUILD_PATH)
    if not jplm_build_path.exists():
        print(f"JPLM build not found: {jplm_build_path}")
        sys.exit(1)
    
    # Parse steps
    if args.steps:
        try:
            steps = [int(step.strip()) for step in args.steps.split(',')]
            for step in steps:
                if step < 1 or step > 6:
                    raise ValueError(f"Step {step} must be between 1 and 6")
        except ValueError as e:
            print(f"ERROR: Invalid steps format: {e}")
            sys.exit(1)
    else:
        steps = [1, 2, 3, 4, 5, 6]
    
    if 1 in steps and not args.lenslet:
        steps.remove(1)
    if 6 in steps and not args.lenslet:
        steps.remove(6)
    
    if steps == []:
        print("There are no steps to execute. Exiting.")
        sys.exit(0)
    
    # Auto-detect grid dimensions
    grid_t, grid_s = get_grid_dimensions(Path(BASE_DATASETS_PATH) / args.dataset_name / "RAW" / "PPM")
    if grid_t <= 0 or grid_s <= 0:
        print("ERROR: Could not determine grid dimensions from PPM files.")
        sys.exit(1)
    else:
        print(f"Detected grid dimensions: {grid_t}x{grid_s}")
        if args.lenslet:
            grid_t -= 2
            grid_s -= 2
            print(f"Adjusted grid dimensions for lenslet: {grid_t}x{grid_s}")
    
    # Auto-detect image dimensions
    width, height = get_ppm_dimensions(Path(BASE_DATASETS_PATH) / args.dataset_name / "RAW" / "PPM")
    if width is None or height is None:
        print("ERROR: Could not determine image dimensions from PPM files.")
        sys.exit(1)
    else:
        print(f"Detected image dimensions: {width}x{height}")
    
    # Delete old folders if they exist
    raw_shifted_ppm_path = dataset_base_path / "RAW" / "PPM_shifted"
    if args.lenslet and 1 in steps and raw_shifted_ppm_path.exists():
        shutil.rmtree(raw_shifted_ppm_path)
    
    raw_shifted_pgx_path = dataset_base_path / "RAW" / "PGX"
    if 2 in steps and raw_shifted_pgx_path.exists():
        shutil.rmtree(raw_shifted_pgx_path)
    
    encoded_path = dataset_base_path / "encoded" / f"{args.dataset_name}.jpl"
    if 3 in steps and encoded_path.exists():
        encoded_path.unlink()
        
    decoded_pgx_path = dataset_base_path / "decoded" / "PGX"
    if 4 in steps and decoded_pgx_path.exists():
        shutil.rmtree(decoded_pgx_path)
        
    decoded_shifted_ppm_path = dataset_base_path / "decoded" / "PPM" / (f"{args.dataset_name}_shifted" if args.lenslet else args.dataset_name)
    if 5 in steps and decoded_shifted_ppm_path.exists():
        shutil.rmtree(decoded_shifted_ppm_path)
        
    decoded_ppm_path = dataset_base_path / "decoded" / "PPM" / args.dataset_name
    if args.lenslet and 6 in steps and decoded_ppm_path.exists():
        shutil.rmtree(decoded_ppm_path)
    
    try:
        processor = JPEGPlenoProcessor(JPLM_BUILD_PATH, BASE_DATASETS_PATH)
        
        success, step_times = processor.process_dataset(
            args.dataset_name,
            grid_t,
            grid_s, 
            width,
            height,
            steps=steps,
            lambda_value=args.lambda_value,
            lenslet=args.lenslet
        )
        
        if not success:
            print(f"PROCESS FAILED for {args.dataset_name}!")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Initialization error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()