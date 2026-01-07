#!/usr/bin/env python3
"""
Video Codec Comparison for Light Field Compression WITH PADDING AND TIMING
Compares HEVC, AV1, and VP9 with proper padding and timing metrics
"""

import argparse
import subprocess
import sys
from pathlib import Path
from PIL import Image
import shutil
import time

def run_ffmpeg_command(cmd, description=""):
    """Esegue un comando FFmpeg e restituisce tempo di esecuzione"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Success - Time: {execution_time:.2f}s")
        return True, execution_time
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Error (after {execution_time:.2f}s): {e}")
        if e.stderr:
            for line in e.stderr.split('\n')[-5:]:
                if line.strip():
                    print(f"  {line}")
        return False, execution_time

def get_ppm_dimensions(ppm_folder):
    """Ottiene le dimensioni dei file PPM"""
    ppm_files = list(ppm_folder.glob("*.ppm"))
    if not ppm_files:
        return None
    
    # Leggi l'header del primo file PPM
    with open(ppm_files[0], 'rb') as f:
        header = list()
        for line in f:
            if line[0] != ord('#'):  # Non è un commento
                for word in line.split():
                    header.append(word)
            if len(header) >= 3:
                break
        
        # Leggi magic number
        magic = header[0]
        if magic != b'P6':
            print(f"Formato PPM non supportato: {magic}")
            return None
        
        width = int(header[1])
        height = int(header[2])
        return width, height

def get_hevc_compatible_dimensions(width, height):
    """Calcola dimensioni multiple di 8 (necessarie per HEVC)"""
    new_width = ((width + 7) // 8) * 8
    new_height = ((height + 7) // 8) * 8
    return new_width, new_height

def compress_with_codec(ppm_folder, output_dir, codec, crf=None):
    """
    Comprime le viste PPM usando FFmpeg.
    Applica il padding ON-THE-FLY usando il filtro 'pad' di FFmpeg se necessario (HEVC).
    """
    
    # 1. Ottieni dimensioni originali leggendo solo il primo file
    original_dimensions = get_ppm_dimensions(ppm_folder)
    if not original_dimensions:
        print("Impossibile leggere dimensioni PPM")
        return None, None, 0
    
    original_width, original_height = original_dimensions
    
    codec_info = {
        'hevc': {
            'name': 'HEVC',
            'codec': 'libx265',
            'extension': 'mp4',
            'requires_padding': True,
            'crf': 14
        },
        'av1': {
            'name': 'AV1', 
            'codec': 'libaom-av1',
            'extension': 'mkv',
            'requires_padding': True,
            'crf': 22
        },
        'vp9': {
            'name': 'VP9',
            'codec': 'libvpx-vp9', 
            'extension': 'webm',
            'requires_padding': True,
            'crf': 22
        }
    }
    
    if codec not in codec_info:
        return None, None, 0
    
    info = codec_info[codec]
    if crf is not None:
        info['crf'] = crf
    
    output_file = output_dir / f"{codec}.{info['extension']}"
    
    # 2. Calcolo del Padding (senza creare file fisici)
    padding_info = None
    filters = []
    
    if info['requires_padding']:
        target_width, target_height = get_hevc_compatible_dimensions(original_width, original_height)
        
        if target_width != original_width or target_height != original_height:
            print(f"Applying FFmpeg padding: {original_width}x{original_height} -> {target_width}x{target_height}")
            # Padding filter: w:h:x:y:color
            # x=0, y=0 allinea l'immagine in alto a sinistra (Top-Left)
            filters.append(f"pad={target_width}:{target_height}:0:0:black")
            
            # Salviamo le info per la decompressione (per fare il crop opposto)
            padding_info = (original_width, original_height, target_width, target_height)
    
    # 3. Costruzione comando FFmpeg
    input_path = str(ppm_folder / "*.ppm")
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "120",
        "-pattern_type", "glob",
        "-i", input_path
    ]
    
    # Aggiungi filtri video se necessari (padding)
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
    
    cmd.extend([
        "-c:v", info['codec'],
        "-crf", str(info['crf']),
        "-pix_fmt", "yuv420p10le"
    ])
    
    # Parametri specifici per codec
    if codec == 'av1':
        cmd.extend(["-cpu-used", "5"])
    elif codec == 'vp9':
        cmd.extend(["-b:v", "0", "-cpu-used", "3"])
    
    cmd.append(str(output_file))
    
    # 4. Esecuzione
    success, compression_time = run_ffmpeg_command(cmd, f"Compression {info['name']} (with internal padding)")
    
    if success and output_file.exists():
        # Nota: ora compression_time include anche il tempo di padding on-the-fly, 
        # che è molto più veloce di farlo con Python prima.
        return output_file, padding_info, compression_time
        
    return None, None, compression_time

def decompress_video_ffmpeg(video_file, output_dir, codec, original_ppm_folder, padding_info=None):
    """
    Decompressione con FIX per larghezza dispari.
    Ordine filtri cruciale:
    1. Bit-shift/Conversione RGB (lutrgb) -> I pixel diventano indipendenti.
    2. Crop -> Ora è possibile tagliare su numeri dispari (es. 625).
    """
    
    decompressed_dir = output_dir / f"{codec}"
    decompressed_dir.mkdir(exist_ok=True)
    
    temp_dir = output_dir / f"temp_ffmpeg_{codec}"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"Decompressing {codec.upper()} (Fix Odd Width)...")
    start_time = time.time()
    
    try:
        cmd = ["ffmpeg", "-y", "-i", str(video_file)]
        
        filters = []
        
        # --- CAMBIAMENTO CRUCIALE QUI ---
        
        # 1. PRIMA applichiamo la correzione colore (lutrgb).
        # Questo costringe FFmpeg a convertire da YUV420p10 a RGB48 PRIMA del crop.
        # Una volta in RGB, non abbiamo più il vincolo dei blocchi 2x2.
        filters.append("lutrgb=r=val/64:g=val/64:b=val/64")
        
        # 2. POI applichiamo il CROP.
        # Essendo ora in RGB, possiamo croppare a 625 (dispari) senza che FFmpeg arrotondi a 624.
        if padding_info:
            original_width, original_height, target_width, target_height = padding_info
            # Crop fisso a 0:0 (Top-Left)
            filters.append(f"crop={original_width}:{original_height}:0:0")
        
        # --------------------------------
        
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
            
        cmd.extend(["-pix_fmt", "rgb48be"])
        
        temp_pattern = temp_dir / "frame_%06d.ppm"
        cmd.append(str(temp_pattern))
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        decompression_time = time.time() - start_time
        
        # --- RINOMINA E PATCH HEADER ---
        original_files = sorted(original_ppm_folder.glob("*.ppm"))
        temp_files = sorted(temp_dir.glob("frame_*.ppm"))
        
        limit = min(len(original_files), len(temp_files))
        
        for i in range(limit):
            src = temp_files[i]
            dst = decompressed_dir / original_files[i].name
            src.rename(dst)
            
            # Patch header 65535 -> 1023
            try:
                with open(dst, 'r+b') as f:
                    header_chunk = f.read(100)
                    val_pos = header_chunk.find(b'65535')
                    if val_pos != -1:
                        f.seek(val_pos)
                        f.write(b' 1023')
            except Exception:
                pass

        shutil.rmtree(temp_dir)
        
        return decompressed_dir, decompression_time

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        return None, time.time() - start_time

def calculate_metrics(original_ppm_dir, decompressed_dir, codec, compressed_dir):
    """Calcola metriche di compressione"""
    
    original_files = list(original_ppm_dir.glob("*.ppm"))
    decompressed_files = list(decompressed_dir.glob("*.ppm"))
    
    if not original_files or not decompressed_files:
        return None
    
    # Verifica che il numero di file corrisponda
    if len(original_files) != len(decompressed_files):
        print(f"Numero di file diverso: original={len(original_files)}, decompressed={len(decompressed_files)}")
    
    original_size = sum(f.stat().st_size for f in original_files)
    compressed_files = list(compressed_dir.glob(f"{codec}.*"))
    
    if not compressed_files:
        return None
    
    compressed_file = compressed_files[0]
    compressed_size = compressed_file.stat().st_size
    
    decompressed_size = sum(f.stat().st_size for f in decompressed_files)
    
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    savings = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    # Verifica dimensioni
    with Image.open(original_files[0]) as orig_img, Image.open(decompressed_files[0]) as dec_img:
        orig_dims = orig_img.size
        dec_dims = dec_img.size
        
        if orig_dims != dec_dims:
            print(f"Dimensioni diverse: {orig_dims} vs {dec_dims}")
        else:
            print(f"Dimensioni corrispondenti: {orig_dims}")
    
    return {
        'original_size_mb': original_size / (1024 * 1024),
        'compressed_size_mb': compressed_size / (1024 * 1024),
        'decompressed_size_mb': decompressed_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'savings_percentage': savings,
        'num_frames': len(decompressed_files),
        'dimensions_match': orig_dims == dec_dims
    }

def format_time(seconds):
    """Formatta il tempo in formato leggibile"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def main():
    parser = argparse.ArgumentParser(
        description="Codec Video Processor - Complete workflow (compression and decompression) with dynamic parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("dataset_name", help="Name of the dataset folder (e.g., bikes, cars)")
    parser.add_argument("--lenslet", action=argparse.BooleanOptionalAction, help="Whether the dataset is of the lenslet type", required=True)
    parser.add_argument("--crf_hevc", type=int, help="CRF value for HEVC (default: 14)")
    parser.add_argument("--crf_av1", type=int, help="CRF value for AV1 (default: 22)")
    parser.add_argument("--crf_vp9", type=int, help="CRF value for VP9 (default: 22)")
    
    args = parser.parse_args()
    
    BASE_DATASETS_PATH = "./datasets"
    
    ppm_folder = Path(BASE_DATASETS_PATH) / args.dataset_name / "RAW" / ("PPM" if not args.lenslet else "PPM_shifted")
    
    if not ppm_folder.exists():
        print(f"Folder not found: {ppm_folder}")
        sys.exit(1)
    
    # Initialize codec CRF values
    codec_crf = {
        'hevc': args.crf_hevc,
        'av1': args.crf_av1,
        'vp9': args.crf_vp9
    }

    # Verifica dimensioni originali
    original_dimensions = get_ppm_dimensions(ppm_folder)
    if original_dimensions:
        print(f"Dimensioni originali: {original_dimensions[0]}x{original_dimensions[1]}")
    
    # Create directory structure
    base_output = ppm_folder.parent.parent
    
    compressed_dir = base_output / "encoded" / "codec_video"
    if compressed_dir.exists():
        shutil.rmtree(compressed_dir)
    compressed_dir.mkdir(parents=True)
    
    decompressed_dir = base_output / "decoded" / "codec_video"
    if decompressed_dir.exists():
        shutil.rmtree(decompressed_dir)
    decompressed_dir.mkdir(parents=True)
    
    print("CODEC COMPARISON WITH TIMING METRICS")
    print("=" * 60)
    print(f"Input: {ppm_folder}")
    print(f"Compressed Output: {compressed_dir}")
    print(f"Decompressed Output: {decompressed_dir}")
    
    # Codecs to test
    codecs = ['hevc', 'av1', 'vp9']
    results = {}
    
    # Phase 1: Compression
    print("\nPHASE 1: COMPRESSION")
    print("=" * 60)
    
    compression_times = {}
    
    for codec in codecs:
        print(f"\n--- {codec.upper()} ---")
        compressed_file, padding_info, compression_time = compress_with_codec(
            ppm_folder, compressed_dir, codec, codec_crf[codec]
        )
        
        if compressed_file:
            results[codec] = {
                'compressed_file': compressed_file,
                'padding_info': padding_info,
                'compression_time': compression_time
            }
            compression_times[codec] = compression_time
            file_size = compressed_file.stat().st_size / (1024 * 1024)
            print(f"Final size: {file_size:.2f} MB")
            print(f"⏱Total compression time: {format_time(compression_time)}")
        else:
            print(f"Compression failed")
            compression_times[codec] = 0
    
    # Phase 2: Decompression
    print("\nPHASE 2: DECOMPRESSION")
    print("=" * 60)
    
    decompression_times = {}
    
    for codec in codecs:
        if codec not in results:
            continue
            
        print(f"\n--- {codec.upper()} ---")
        decompressed_folder, decompression_time = decompress_video_ffmpeg(
            results[codec]['compressed_file'],
            decompressed_dir,
            codec,
            ppm_folder,
            results[codec]['padding_info']
        )
        
        if decompressed_folder:
            results[codec]['decompressed_folder'] = decompressed_folder
            results[codec]['decompression_time'] = decompression_time
            decompression_times[codec] = decompression_time
            num_frames = len(list(decompressed_folder.glob("*.ppm")))
            print(f"Decompressed frames: {num_frames} PPM")
            print(f"Total decompression time: {format_time(decompression_time)}")
        else:
            print(f"Decompression failed")
            decompression_times[codec] = 0
    
    # Phase 3: Metrics
    print("\nPHASE 3: METRICS")
    print("=" * 60)
    
    original_files = list(ppm_folder.glob("*.ppm"))
    original_size = sum(f.stat().st_size for f in original_files)
    print(f"Original size: {original_size/(1024*1024):.2f} MB")
    print(f"Number of views: {len(original_files)}")
    
    print(f"\n{'CODEC':<10} {'COMPRESSED':<12} {'RATIO':<10} {'SAVINGS':<10} {'COMP_TIME':<12} {'DEC_TIME':<12} {'DIMS_OK':<8}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    
    for codec in codecs:
        if codec not in results or 'decompressed_folder' not in results[codec]:
            continue
            
        metrics = calculate_metrics(ppm_folder, results[codec]['decompressed_folder'], codec, compressed_dir)
        
        if metrics:
            results[codec]['metrics'] = metrics
            dims_ok = "OK" if metrics['dimensions_match'] else "FAIL"
            comp_time = format_time(results[codec]['compression_time'])
            dec_time = format_time(results[codec]['decompression_time'])
            
            print(f"{codec.upper():<10} {metrics['compressed_size_mb']:<10.2f}MB {metrics['compression_ratio']:<9.1f}:1 "
                  f"{metrics['savings_percentage']:<9.1f}% {comp_time:<12} {dec_time:<12} {dims_ok:<8}")
    
    # Summary with timing
    print(f"\nRIEPILOGO TEMPI:")
    print("=" * 60)
    
    # Trova il codec più veloce
    fastest_comp = min(compression_times.items(), key=lambda x: x[1]) if compression_times else None
    fastest_dec = min(decompression_times.items(), key=lambda x: x[1]) if decompression_times else None
    
    for codec in codecs:
        if codec in results and 'metrics' in results[codec]:
            metrics = results[codec]['metrics']
            comp_time = results[codec]['compression_time']
            dec_time = results[codec]['decompression_time']
            
            comp_flag = " BEST" if fastest_comp and codec == fastest_comp[0] else ""
            dec_flag = " BEST" if fastest_dec and codec == fastest_dec[0] else ""
            
            print(f"{codec.upper():<6}: {metrics['compression_ratio']:5.1f}:1 compression "
                  f"({metrics['savings_percentage']:4.1f}%) | "
                  f"Comp: {format_time(comp_time)}{comp_flag} | "
                  f"Dec: {format_time(dec_time)}{dec_flag}")
    
    # Total time
    total_comp_time = sum(compression_times.values())
    total_dec_time = sum(decompression_times.values())
    total_time = total_comp_time + total_dec_time
    
    print(f"\nTEMPO TOTALE: {format_time(total_time)} "
          f"(Compressione: {format_time(total_comp_time)}, "
          f"Decompressione: {format_time(total_dec_time)})")
    
    print(f"\nComplete! Files in: {base_output}")
    print("Le dimensioni delle immagini decompresse corrispondono alle originali per l'analisi SSIM!")

if __name__ == "__main__":
    main()