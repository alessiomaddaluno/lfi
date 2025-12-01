#!/usr/bin/env python3
"""
Video Codec Comparison for Light Field Compression WITH PADDING AND TIMING
Compares HEVC, AV1, and VP9 with proper padding and timing metrics
"""

import os
import subprocess
import sys
from pathlib import Path
import av
from PIL import Image
import shutil
import tempfile
import time
from datetime import timedelta

def run_ffmpeg_command(cmd, description=""):
    """Esegue un comando FFmpeg e restituisce tempo di esecuzione"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
        # Leggi magic number
        magic = f.readline().strip()
        if magic != b'P6':
            print(f"Formato PPM non supportato: {magic}")
            return None
        
        # Salta commenti e leggi dimensioni
        while True:
            line = f.readline()
            if line[0] != ord('#'):  # Non è un commento
                try:
                    width, height = map(int, line.split())
                    return width, height
                except ValueError:
                    continue

def get_hevc_compatible_dimensions(width, height):
    """Trova dimensioni compatibili con HEVC (multiple di 8)"""
    new_width = ((width + 7) // 8) * 8  # Arrotonda per eccesso al multiplo di 8
    new_height = ((height + 7) // 8) * 8
    
    if new_width == width and new_height == height:
        print(f"Dimensioni {width}x{height} già compatibili con HEVC")
        return width, height
    
    print(f"Padding necessario: da {width}x{height} a {new_width}x{new_height}")
    return new_width, new_height

def add_padding_to_ppm(ppm_folder, output_dir, target_width, target_height):
    """Aggiunge padding nero ai PPM per raggiungere dimensioni compatibili"""
    padded_dir = output_dir / "padded"
    padded_dir.mkdir(exist_ok=True)
    
    print(f"Aggiungo padding alle immagini...")
    start_time = time.time()
    
    padded_files = []
    
    for ppm_file in sorted(ppm_folder.glob("*.ppm")):
        # Leggi immagine originale
        with Image.open(ppm_file) as img:
            original_width, original_height = img.size
            
            # Calcola padding (centrato)
            pad_width = target_width - original_width
            pad_height = target_height - original_height
            
            left = pad_width // 2
            top = pad_height // 2
            
            # Crea nuova immagine con padding nero
            if img.mode == 'RGB':
                padded_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                padded_img.paste(img, (left, top))
            else:
                # Converti a RGB se necessario
                rgb_img = img.convert('RGB')
                padded_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                padded_img.paste(rgb_img, (left, top))
            
            # Salva
            output_path = padded_dir / ppm_file.name
            padded_img.save(output_path, format='PPM')
            padded_files.append(output_path)
    
    end_time = time.time()
    padding_time = end_time - start_time
    print(f"Padding completato: {len(padded_files)} immagini - Time: {padding_time:.2f}s")
    return padded_dir, padding_time

def remove_padding_from_images(padded_folder, output_dir, original_width, original_height, target_width, target_height):
    """Rimuove il padding dalle immagini decompresse"""
    unpadded_dir = output_dir / "unpadded"
    unpadded_dir.mkdir(exist_ok=True)
    
    print(f"Rimuovo padding dalle immagini...")
    start_time = time.time()
    
    # Calcola coordinate per crop
    pad_width = target_width - original_width
    pad_height = target_height - original_height
    
    left = pad_width // 2
    top = pad_height // 2
    right = left + original_width
    bottom = top + original_height
    
    for ppm_file in sorted(padded_folder.glob("*.ppm")):
        with Image.open(ppm_file) as img:
            # Crop per rimuovere padding
            cropped_img = img.crop((left, top, right, bottom))
            
            # Salva
            output_path = unpadded_dir / ppm_file.name
            cropped_img.save(output_path, format='PPM')
    
    # Sposta i file unpadded nella directory finale
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    
    for unpadded_file in unpadded_dir.glob("*.ppm"):
        final_file = final_dir / unpadded_file.name
        unpadded_file.rename(final_file)
    
    # Pulizia
    shutil.rmtree(unpadded_dir)
    
    end_time = time.time()
    unpadding_time = end_time - start_time
    
    print(f"Padding rimosso: {len(list(final_dir.glob('*.ppm')))} immagini - Time: {unpadding_time:.2f}s")
    return final_dir, unpadding_time

def compress_with_codec(ppm_folder, output_dir, codec, crf=3):
    """Compress le viste PPM con codec specifico e padding per HEVC"""
    
    # Ottieni dimensioni originali
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
            'requires_padding': True
        },
        'av1': {
            'name': 'AV1', 
            'codec': 'libaom-av1',
            'extension': 'mkv',
            'requires_padding': False
        },
        'vp9': {
            'name': 'VP9',
            'codec': 'libvpx-vp9', 
            'extension': 'webm',
            'requires_padding': False
        }
    }
    
    if codec not in codec_info:
        return None, None, 0
    
    info = codec_info[codec]
    output_file = output_dir / f"compressed_{codec}.{info['extension']}"
    
    # Gestione padding per HEVC
    temp_padded_dir = None
    padding_time = 0
    unpadding_time = 0
    
    if info['requires_padding']:
        target_width, target_height = get_hevc_compatible_dimensions(original_width, original_height)
        padded_dir, padding_time = add_padding_to_ppm(ppm_folder, output_dir, target_width, target_height)
        input_path = str(padded_dir / "*.ppm")
        padding_info = (original_width, original_height, target_width, target_height)
        temp_padded_dir = padded_dir
    else:
        input_path = str(ppm_folder / "*.ppm")
        padding_info = None
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "120",
        "-pattern_type", "glob",
        "-i", input_path,
        "-c:v", info['codec'],
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    
    if codec == 'av1':
        cmd.extend(["-cpu-used", "4"])
    elif codec == 'vp9':
        cmd.extend(["-b:v", "0"])
    
    cmd.append(str(output_file))
    
    success, compression_time = run_ffmpeg_command(cmd, f"Compression {info['name']}")
    
    # Pulizia directory temporanea padding
    if temp_padded_dir and temp_padded_dir.exists():
        shutil.rmtree(temp_padded_dir)
    
    if success and output_file.exists():
        total_time = padding_time + compression_time
        return output_file, padding_info, total_time
    return None, None, compression_time

def decompress_video_pyav(video_file, output_dir, codec, original_ppm_folder, padding_info=None):
    """Decompressione video usando PyAV con gestione padding"""
    
    decompressed_dir = output_dir / f"decompressed_{codec}"
    decompressed_dir.mkdir(exist_ok=True)
    
    print(f"Decompressing {codec.upper()} with PyAV")
    start_time = time.time()
    
    try:
        container = av.open(str(video_file), mode="r") 
        stream = container.streams.video[0]
        count = 0
        
        original_files = sorted(original_ppm_folder.glob("*.ppm"))
        
        # Directory temporanea per frame con padding
        temp_frame_dir = output_dir / f"temp_frames_{codec}"
        temp_frame_dir.mkdir(exist_ok=True)
        
        for frame in container.decode(stream):
            if count < len(original_files):
                orig_name = original_files[count].name
                
                # Salva frame temporaneamente (potrebbe avere padding)
                temp_output = temp_frame_dir / f"frame_{count:06d}.ppm"
                img = frame.to_image()
                
                # Converti a RGB se necessario
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(str(temp_output))
                count += 1
            else:
                break
        
        container.close()
        
        decompression_time = time.time() - start_time
        unpadding_time = 0
        
        # Gestione padding se necessario
        if padding_info:
            original_width, original_height, target_width, target_height = padding_info
            final_dir, unpadding_time = remove_padding_from_images(
                temp_frame_dir, output_dir, 
                original_width, original_height, target_width, target_height
            )
            
            # Rinomina i file con i nomi originali
            for i, ppm_file in enumerate(sorted(final_dir.glob("*.ppm"))):
                if i < len(original_files):
                    orig_name = original_files[i].name
                    final_path = decompressed_dir / orig_name
                    ppm_file.rename(final_path)
                    
                    if i < 5:
                        print(f"  Saved: {orig_name}")
        else:
            # Senza padding, rinomina direttamente
            for i, temp_file in enumerate(sorted(temp_frame_dir.glob("*.ppm"))):
                if i < len(original_files):
                    orig_name = original_files[i].name
                    final_path = decompressed_dir / orig_name
                    temp_file.rename(final_path)
                    
                    if i < 5:
                        print(f"  Saved: {orig_name}")
        
        # Pulizia directory temporanee
        if temp_frame_dir.exists():
            shutil.rmtree(temp_frame_dir)
        
        final_files = list(decompressed_dir.glob("*.ppm"))
        total_decompression_time = decompression_time + unpadding_time
        
        print(f"Decompressed frames: {len(final_files)}")
        print(f"Decompression time: {decompression_time:.2f}s + unpadding: {unpadding_time:.2f}s = {total_decompression_time:.2f}s")
        
        # Verifica dimensioni
        if final_files:
            with Image.open(final_files[0]) as img:
                final_width, final_height = img.size
                print(f"Final dimensions: {final_width}x{final_height}")
        
        return decompressed_dir if final_files else None, total_decompression_time
        
    except Exception as e:
        decompression_time = time.time() - start_time
        print(f"PyAV decompression error after {decompression_time:.2f}s: {e}")
        # Pulizia in caso di errore
        for temp_dir in [output_dir / "padded", output_dir / "unpadded", output_dir / "final", output_dir / f"temp_frames_{codec}"]:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        return None, decompression_time

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
    compressed_files = list(compressed_dir.glob(f"compressed_{codec}.*"))
    
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
    if len(sys.argv) < 2:
        print("Usage: python compare_codecs_timing.py <ppm_folder>")
        print("  ppm_folder: Folder with PPM files")
        sys.exit(1)
    
    ppm_folder = Path(sys.argv[1])
    
    if not ppm_folder.exists():
        print(f"Folder not found: {ppm_folder}")
        sys.exit(1)
    
    # Verifica dimensioni originali
    original_dimensions = get_ppm_dimensions(ppm_folder)
    if original_dimensions:
        print(f"Dimensioni originali: {original_dimensions[0]}x{original_dimensions[1]}")
    
    # Create directory structure
    base_output = ppm_folder.parent / "codec_comparison_timing"
    compressed_dir = base_output / "compressed"
    decompressed_dir = base_output / "decompressed"
    
    for dir_path in [base_output, compressed_dir, decompressed_dir]:
        dir_path.mkdir(exist_ok=True)
    
    print("CODEC COMPARISON WITH TIMING METRICS")
    print("=" * 60)
    print(f"Input: {ppm_folder}")
    print(f"Output: {base_output}")
    
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
            ppm_folder, compressed_dir, codec, crf=3
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
    
    # Phase 2: Decompression with PyAV
    print("\nPHASE 2: DECOMPRESSION")
    print("=" * 60)
    
    decompression_times = {}
    
    for codec in codecs:
        if codec not in results:
            continue
            
        print(f"\n--- {codec.upper()} ---")
        decompressed_folder, decompression_time = decompress_video_pyav(
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