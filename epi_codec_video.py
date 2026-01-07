#!/usr/bin/env python3
from math import sqrt
import sys
import numpy as np
import cv2
import os
import glob
import argparse
import subprocess
import time
import shutil
from pathlib import Path

# --- CONFIGURAZIONE ---
TILES_PER_FRAME = 64 

def format_time(seconds):
    if seconds < 60: return f"{seconds:.2f}s"
    return f"{seconds/60:.2f}m"

def get_hevc_compatible_dimensions(width, height):
    """Calcola dimensioni multiple di 8 (necessarie per HEVC/AV1)"""
    new_width = ((width + 7) // 8) * 8
    new_height = ((height + 7) // 8) * 8
    return new_width, new_height

def load_16bit_dataset(folder_path, grid_u, grid_v):
    search_path = os.path.join(folder_path, '*.ppm')
    file_paths = glob.glob(search_path)
    if not file_paths:
        search_path = os.path.join(folder_path, '*.png')
        file_paths = glob.glob(search_path)
    
    file_paths.sort()
    
    total_expected = grid_u * grid_v
    if len(file_paths) < total_expected:
        raise ValueError(f"Trovati {len(file_paths)} file, attesi {total_expected}.")
    
    print(f"Loading {total_expected} images (16-bit)...")
    
    # Leggi prima immagine per dimensioni
    first = cv2.imread(file_paths[0], cv2.IMREAD_UNCHANGED)
    if first is None: raise ValueError("Impossibile leggere le immagini.")
        
    H, W = first.shape[:2]
    C = 3 if len(first.shape) > 2 else 1
    
    lf_data = np.zeros((grid_u, grid_v, H, W, C), dtype=np.uint16)
    names = []
    
    start_load = time.time()
    for idx in range(total_expected):
        path = file_paths[idx]
        names.append(os.path.basename(path))
        u, v = idx % grid_u, idx // grid_u
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H))
        lf_data[u, v] = img
        
    print(f"Dataset caricato in {format_time(time.time() - start_load)}. Shape: {lf_data.shape}")
    return lf_data, names

def transform_to_epi_volume(lf_data):
    start_t = time.time()
    
    U, V, H, W, C = lf_data.shape
    print("Trasformazione in EPI Tiled...")
    
    epi_raw = lf_data.transpose(1, 2, 0, 3, 4).reshape(V * H, U, W, C)
    
    total_epis = epi_raw.shape[0]
    remainder = total_epis % TILES_PER_FRAME
    pad_frames = TILES_PER_FRAME - remainder if remainder != 0 else 0
    
    if pad_frames > 0:
        epi_raw = np.pad(epi_raw, ((0, pad_frames), (0,0), (0,0), (0,0)), mode='edge')
        
    new_frame_count = epi_raw.shape[0] // TILES_PER_FRAME
    final_height = TILES_PER_FRAME * U
    tiled_volume = epi_raw.reshape(new_frame_count, final_height, W, C)
    
    print(f"Volume EPI: {new_frame_count} frames, {W}x{final_height}")
    return tiled_volume, pad_frames, time.time() - start_t

def reconstruct_from_epi(tiled_volume, pad_frames, grid_u, grid_v, original_h, original_w):
    start_t = time.time()
    
    frames, tile_h, w, c = tiled_volume.shape
    total_epis_padded = frames * TILES_PER_FRAME
    epi_raw = tiled_volume.reshape(total_epis_padded, grid_u, w, c)
    
    if pad_frames > 0:
        epi_raw = epi_raw[:-pad_frames]
        
    lf_reconstructed = epi_raw.reshape(grid_v, original_h, grid_u, w, c).transpose(2, 0, 1, 3, 4)
    return lf_reconstructed, time.time() - start_t

def compress_epi_video(volume, output_path, codec, crf=None):
    frames, h, w, c = volume.shape
    
    # --- CALCOLO PADDING (Multipli di 8) ---
    target_w, target_h = get_hevc_compatible_dimensions(w, h)
    pad_w = target_w - w
    pad_h = target_h - h
    
    # Applica padding se necessario
    if pad_w > 0 or pad_h > 0:
        print(f"Padding Codec: {w}x{h} -> {target_w}x{target_h} (Pad W={pad_w}, H={pad_h})")
        # Padding su assi: (Frames, Height, Width, Channels)
        volume = np.pad(volume, ((0,0), (0, pad_h), (0, pad_w), (0,0)), mode='edge')
    
    # Aggiorna dimensioni per FFmpeg
    w, h = target_w, target_h

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr48le',  
        '-s', f'{w}x{h}',
        '-r', '24',
        '-i', '-',
        '-vf', 'lutrgb=r=val*64:g=val*64:b=val*64',
    ]
    
    if codec == 'av1':
        cmd.extend(['-c:v', 'libaom-av1', '-cpu-used', '5', '-crf', ('11' if crf is None else str(crf))])
    elif codec == 'hevc':
        cmd.extend(['-c:v', 'libx265', '-crf', ('14' if crf is None else str(crf))])
    elif codec == 'vp9':
        cmd.extend(['-c:v', 'libvpx-vp9', '-b:v', '0', "-cpu-used", "3", '-crf', ('15' if crf is None else str(crf))])
    
    cmd.extend([
        '-pix_fmt', 'yuv420p10le',
        str(output_path)
    ])
    
    print(f"[{codec.upper()}] Encoding...")
    print(f"Command: {' '.join(cmd)}")
    start_t = time.time()
    
    # IMPORTANTE: Catturiamo stderr per debugging
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=10**8)
    
    try:
        proc.stdin.write(volume.tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception as e:
        print(f"Errore pipe python: {e}")
        if proc.poll() is None: proc.kill()
        return False, 0, 0, 0
    
    if proc.returncode != 0:
        err_msg = proc.stderr.read().decode()
        print(f"FFmpeg Compression Error (Exit {proc.returncode}):\n{err_msg}")
        return False, 0, 0, 0
        
    return True, time.time() - start_t, pad_w, pad_h

def decompress_epi_video(video_path, shape_meta, pad_w, pad_h):
    frames, h_orig, w_orig, c = shape_meta
    
    # Dimensioni del video su disco (che includono il padding)
    w_video = w_orig + pad_w
    h_video = h_orig + pad_h
    
    frame_bytes = w_video * h_video * c * 2 
    total_expected_bytes = frames * frame_bytes

    # Controllo pre-esecuzione
    if not os.path.exists(video_path):
        print(f"ERRORE: File video non trovato: {video_path}")
        return None, 0
    
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        print(f"ERRORE: File video vuoto (0 bytes): {video_path}")
        return None, 0

    print(f"Decoding {os.path.basename(video_path)} (Reading {w_video}x{h_video})...")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-f', 'rawvideo',
        '-vf', 'lutrgb=r=val/64:g=val/64:b=val/64',
        '-pix_fmt', 'bgr48le', 
        '-'
    ]
    
    start_t = time.time()
    # MODIFICA FONDAMENTALE: stderr=subprocess.PIPE per leggere l'errore se fallisce
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    
    try:
        raw_data, err_data = proc.communicate() # Legge tutto stdout e stderr
        
        if len(raw_data) != total_expected_bytes:
            print(f"ERRORE CRITICO FFmpeg:")
            print(f"  Letti: {len(raw_data)} bytes")
            print(f"  Attesi: {total_expected_bytes} bytes")
            print(f"  FFmpeg Stderr:\n{err_data.decode()}") # Stampa il vero motivo dell'errore
            return None, 0
            
        # 1. Reshape con dimensioni paddate
        volume = np.frombuffer(raw_data, dtype=np.uint16).reshape(frames, h_video, w_video, c)
        
        # 2. Rimuovi padding (Crop)
        if pad_w > 0 or pad_h > 0:
            volume = volume[:, :h_orig, :w_orig, :]
        
    except Exception as e:
        print(f"Errore decompressione Python: {e}")
        return None, 0
        
    return volume, time.time() - start_t

def run_comparison(input_folder, grid_u, grid_v, codec_crf):
    output_base = Path(input_folder).parent.parent
    
    comp_dir = output_base / "encoded" / "epi_codec_video"
    if comp_dir.exists(): shutil.rmtree(comp_dir)
    comp_dir.mkdir(parents=True)
    
    decomp_dir = output_base / "decoded" / "epi_codec_video"
    if decomp_dir.exists(): shutil.rmtree(decomp_dir)
    decomp_dir.mkdir(parents=True)
    
    try:
        lf_data, names = load_16bit_dataset(input_folder, grid_u, grid_v)
    except Exception as e:
        print(f"Errore caricamento: {e}")
        return

    original_size_mb = lf_data.nbytes / (1024**2)
    U, V, H, W, C = lf_data.shape
    
    epi_vol, pad_frames, t_transform = transform_to_epi_volume(lf_data)
    epi_shape = epi_vol.shape 
    
    codecs = ['hevc', 'av1', 'vp9']
    results = {}
    
    print("\n" + "="*60)
    print(f"STARTING COMPARISON (10-bit EPI Pipeline)")
    print(f"Input Grid: {grid_u}x{grid_v} | Res: {W}x{H}")
    print("="*60)

    for codec in codecs:
        print(f"\n--- Processing {codec.upper()} ---")
        ext = 'mkv' if codec == 'av1' else 'mp4' if codec == 'hevc' else 'webm'
        vid_path = comp_dir / f"epi_{codec}.{ext}"
        
        # A. Compressione
        success, t_comp, pad_w, pad_h = compress_epi_video(epi_vol, vid_path, codec, codec_crf[codec])
        if not success: continue
        
        size_mb = os.path.getsize(vid_path) / (1024**2)
        ratio = original_size_mb / size_mb
        
        # B. Decompressione
        rec_vol_epi, t_dec = decompress_epi_video(vid_path, epi_shape, pad_w, pad_h)
        if rec_vol_epi is None: continue
        
        print("Ricostruzione Light Field 5D...")
        lf_rec, t_reconstruct = reconstruct_from_epi(rec_vol_epi, pad_frames, grid_u, grid_v, H, W)
        
        # D. Salvataggio Immagini
        out_folder_codec = decomp_dir / codec
        out_folder_codec.mkdir()
        
        print(f"Salvataggio {len(names)} immagini...")
        for i in range(len(names)):
            f_path = out_folder_codec / names[i]
            
            u_idx = i % grid_u
            v_idx = i // grid_u
            
            if v_idx < lf_rec.shape[1] and u_idx < lf_rec.shape[0]:
                cv2.imwrite(str(f_path), lf_rec[u_idx, v_idx])
            
                # Patch header 65535 -> 1023 (solo per PPM non standard)
                try:
                    if str(f_path).endswith('.ppm'):
                        with open(f_path, 'r+b') as f:
                            header_chunk = f.read(100)
                            val_pos = header_chunk.find(b'65535')
                            if val_pos != -1:
                                f.seek(val_pos)
                                f.write(b' 1023')
                except Exception:
                    pass
        
        results[codec] = {
            'size_mb': size_mb,
            'ratio': ratio,
            'time_comp': t_comp + t_transform,
            'time_dec': t_dec + t_reconstruct,
            'savings': (1 - size_mb/original_size_mb)*100
        }
        
    print("\n" + "="*60)
    print(f"{'CODEC':<8} {'SIZE (MB)':<12} {'RATIO':<10} {'SAVINGS':<10} {'T_COMP':<10} {'T_DEC':<10}")
    print("-" * 60)
    print(f"{'ORIG':<8} {original_size_mb:<12.2f} {'1.0:1':<10} {'0.0%':<10} {'-':<10} {'-':<10}")
    
    for codec in codecs:
        if codec in results:
            r = results[codec]
            print(f"{codec.upper():<8} {r['size_mb']:<12.2f} {r['ratio']:<10.1f} {r['savings']:<9.1f}% {format_time(r['time_comp']):<10} {format_time(r['time_dec']):<10}")
            
    print("=" * 60)
    print(f"Output salvati in: {output_base}")

def get_grid_dimensions(folder_path):
    """
    Calcola le dimensioni della griglia (t, s) basate sul numero di file .ppm nella cartella.
    Assumiamo che i file rappresentino una griglia completa.
    """
    
    files = list(folder_path.glob("*.ppm"))
    grid_dim = sqrt(len(files))
    return int(grid_dim), int(grid_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EPI Codec Video Processor - Complete workflow (compression and decompression) with dynamic parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("dataset_name", help="Name of the dataset folder (e.g., bikes, cars)")
    parser.add_argument("--lenslet", action=argparse.BooleanOptionalAction, help="Whether the dataset is of the lenslet type", required=True)
    parser.add_argument("--crf_hevc", type=int, help="CRF value for HEVC (default: 14)")
    parser.add_argument("--crf_av1", type=int, help="CRF value for AV1 (default: 11)")
    parser.add_argument("--crf_vp9", type=int, help="CRF value for VP9 (default: 15)")
    
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
    
    # Auto-detect grid dimensions
    grid_t, grid_s = get_grid_dimensions(ppm_folder)
    if grid_t <= 0 or grid_s <= 0:
        print("ERROR: Could not determine grid dimensions from PPM files.")
        sys.exit(1)
    else:
        print(f"Detected grid dimensions: {grid_t}x{grid_s}")
    
    run_comparison(ppm_folder, grid_t, grid_s, codec_crf)