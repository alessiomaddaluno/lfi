import sys
import os
import subprocess
from pathlib import Path
import tempfile

def main():
    if len(sys.argv[1:]) == 4:
        decompressed_path = sys.argv[1]
        original_path = sys.argv[2]
        compressed_jpl_path = sys.argv[3]
        mode = sys.argv[4].upper()
    else:
        print("Specify those parameters:")
        print("\t-Decompressed input path (es. ./datasets/bikes/decoded/PPM/bikes/)")
        print("\t-Original input path (es. ./datasets/bikes/RAW/PPM/)") 
        print("\t-Compressed JPL path (es. ./datasets/bikes/encoded/bikes.jpl)")
        print("\t-Mode (es. LOSSY, LOSSLESS)")
        sys.exit(1)

    print("ANALISI COMPRESSIONE JPEG PLENO")
    print("=" * 50)
    
    original_folder = Path(original_path)
    decompressed_folder = Path(decompressed_path)
    
    # Calcolo dimensioni
    original_size = sum(f.stat().st_size for f in original_folder.glob("*.ppm"))
    decompressed_size = sum(f.stat().st_size for f in decompressed_folder.glob("*.ppm"))
    jpl_size = Path(compressed_jpl_path).stat().st_size
    
    # Metriche semplici e chiare
    compression_ratio = original_size / jpl_size
    savings_percentage = (1 - jpl_size / original_size) * 100
    num_views = len(list(original_folder.glob("*.ppm")))
    
    print(f"Viste nel light field:   {num_views:8d}")
    print(f"Dimensione originale:    {original_size/1000000:8.2f} MB")
    print(f"Dimensione compressa:    {jpl_size/1000000:8.2f} MB") 
    print(f"Dimensione decompressa:  {decompressed_size/1000000:8.2f} MB")
    print(f"Rapporto compressione:   {compression_ratio:8.1f}:1")
    print(f"Riduzione dimensione:    {savings_percentage:8.1f}%")
    
    if mode == "LOSSY":
        print(f"\nANALISI QUALITÀ (LOSSY)")
        print("=" * 50)
        
        # Crea file di lista per FFmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='_original.txt', delete=False) as f_orig:
            orig_list_file = f_orig.name
            for ppm_file in sorted(original_folder.glob("*.ppm")):
                f_orig.write(f"file '{ppm_file.absolute()}'\n")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_decompressed.txt', delete=False) as f_decomp:
            decomp_list_file = f_decomp.name
            for ppm_file in sorted(decompressed_folder.glob("*.ppm")):
                f_decomp.write(f"file '{ppm_file.absolute()}'\n")
        
        try:
            # FFmpeg per SSIM
            print("Calcolo SSIM con FFmpeg...")
            ssim_result = subprocess.run([
                'ffmpeg',
                '-f', 'concat', 
                '-safe', '0',
                '-i', orig_list_file,
                '-f', 'concat', 
                '-safe', '0',
                '-i', decomp_list_file,
                '-lavfi', 'ssim=stats_file=ssim_stats.txt',
                '-f', 'null', '-'
            ], capture_output=True, text=True)
            
            # Analizza risultati SSIM
            ssim_values = []
            if os.path.exists('ssim_stats.txt'):
                with open('ssim_stats.txt', 'r') as f:
                    for line in f:
                        if 'All:' in line:
                            try:
                                parts = line.split('All:')
                                if len(parts) == 2:
                                    ssim_val = float(parts[1].split()[0])
                                    ssim_values.append(ssim_val)
                            except ValueError:
                                continue
            
            # Risultati qualità
            if ssim_values:
                ssim_mean = sum(ssim_values) / len(ssim_values)
                ssim_std = (sum((x - ssim_mean) ** 2 for x in ssim_values) / len(ssim_values)) ** 0.5
                print(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")
                
                # Valutazione semplice
                if ssim_mean > 0.95:
                    qualita = "ECCELLENTE"
                elif ssim_mean > 0.90:
                    qualita = "OTTIMA"
                elif ssim_mean > 0.80:
                    qualita = "BUONA"
                elif ssim_mean > 0.70:
                    qualita = "ACCETTABILE"
                else:
                    qualita = "SCADENTE"
                    
                print(f"Qualità visiva: {qualita}")
            else:
                print("Impossibile calcolare SSIM")
            
        finally:
            # Pulizia
            for temp_file in [orig_list_file, decomp_list_file, 'ssim_stats.txt']:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    print(f"\nRIEPILOGO {mode}:")
    print("=" * 50)
    print(f"Compressione: {compression_ratio:.0f}:1")
    print(f"Riduzione: {savings_percentage:.1f}%")
    if mode == "LOSSY" and ssim_values:
        print(f"Qualità SSIM: {ssim_mean:.4f}")

if __name__ == "__main__":
    main()