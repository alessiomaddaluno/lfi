import sys
import numpy as np
import cv2
import os
import glob
import argparse
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def comparison(original_folder, decompressed_folder, compressed_file_path):
    
    # --- FUNZIONI DI UTILITÀ ---
    def get_folder_size(folder, extension=".ppm"):
        """Calcola la dimensione totale dei file con una certa estensione in una cartella."""
        total_size = 0
        files = glob.glob(os.path.join(folder, f"*{extension}"))
        for f in files:
            total_size += os.path.getsize(f)
        return total_size

    def load_frames(folder):
        # Cerca file PPM (case insensitive per estensione)
        paths = sorted([p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith('.ppm')])
        frames = []
        names = []
        
        print(f"Caricamento frame da {folder}...")
        for p in paths:
            # MODIFICA 1: cv2.IMREAD_UNCHANGED è fondamentale per leggere 16-bit/10-bit
            # Senza questo flag, OpenCV converte tutto a 8-bit.
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            
            if img is not None:
                frames.append(img)
                names.append(os.path.basename(p))
            else:
                print(f"Warning: Impossibile leggere {p}")
                
        if frames:
            # Controllo profondità bit del primo frame
            dtype = frames[0].dtype
            max_val = np.max(frames[0])
            print(f"  -> Tipo dati rilevato: {dtype}")
            print(f"  -> Valore massimo rilevato nel primo frame: {max_val}")
            
        print(f"Caricati {len(frames)} frame.")
        return frames, names
    
    # ==========================================
    # 1. ANALISI COMPRESSIONE
    # ==========================================
    print("\n" + "=" * 60)
    print("ANALISI COMPRESSIONE")
    print("=" * 60)

    # Calcolo dimensioni
    original_size_bytes = get_folder_size(original_folder, ".ppm")
    
    if original_size_bytes == 0:
        original_size_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(original_folder, "*")) if os.path.isfile(f))

    try:
        compressed_size_bytes = os.path.getsize(compressed_file_path)
    except OSError:
        print(f"Errore: Impossibile leggere il file compresso: {compressed_file_path}")
        return

    # Calcolo metriche compressione
    original_mb = original_size_bytes / (1024 * 1024)
    compressed_mb = compressed_size_bytes / (1024 * 1024)
    
    if compressed_size_bytes > 0:
        ratio = original_size_bytes / compressed_size_bytes
        savings = (1 - (compressed_size_bytes / original_size_bytes)) * 100
    else:
        ratio = 0
        savings = 0

    print(f"Dimensione Originale (Raw):  {original_mb:8.2f} MB")
    print(f"Dimensione Compressa:        {compressed_mb:8.2f} MB")
    print("-" * 40)
    print(f"Rapporto di Compressione:    {ratio:8.2f}:1")
    print(f"Risparmio di Spazio:         {savings:8.2f}%")
    print("-" * 40)
    
    # Bitrate
    sample_paths = glob.glob(os.path.join(original_folder, "*.ppm"))
    if sample_paths:
        # Legge dimensioni, usa IMREAD_UNCHANGED per sicurezza anche qui
        sample = cv2.imread(sample_paths[0], cv2.IMREAD_UNCHANGED)
        if sample is not None:
            h, w = sample.shape[:2] # Gestisce sia grayscale che RGB
            num_images = len(sample_paths)
            total_pixels = w * h * num_images
            bpp = (compressed_size_bytes * 8) / total_pixels
            print(f"Bitrate (bpp):               {bpp:8.4f} bit/pixel")

    # ==========================================
    # 2. ANALISI QUALITÀ (10-BIT)
    # ==========================================
    print("\n" + "=" * 60)
    print("CONFRONTO QUALITÀ - PSNR & SSIM (10-BIT)")
    print("=" * 60)

    orig_frames, orig_names = load_frames(original_folder)
    decomp_frames, decomp_names = load_frames(decompressed_folder)
    
    num_frames = min(len(orig_frames), len(decomp_frames))
    print(f"Frame da confrontare: {num_frames}")
    
    if num_frames == 0:
        print("Nessun frame da confrontare!")
        return
    
    # MODIFICA 2: Impostiamo il Data Range corretto per 10 bit
    # Poiché abbiamo patchato l'header a 1023, il range è 1023.
    # Se usassimo 65535, il PSNR risulterebbe artificialmente altissimo.
    DATA_RANGE = 1023
    print(f"Utilizzando Data Range per metriche: {DATA_RANGE} (10-bit)")

    # Metriche
    psnr_values = []
    ssim_values = []
    
    for i in range(num_frames):
        # Convertiamo in float32 per i calcoli matematici
        orig = orig_frames[i].astype(np.float32)
        decomp = decomp_frames[i].astype(np.float32)
        
        # Calcola MSE
        current_mse = np.mean((orig - decomp) ** 2)
        
        # Calcola PSNR
        if current_mse == 0:
            current_psnr = float('inf') 
        else:
            # MODIFICA 3: data_range=1023
            current_psnr = psnr(orig, decomp, data_range=DATA_RANGE)
        psnr_values.append(current_psnr)
        
        # Calcola SSIM
        ssim_channels = []
        # Controllo se immagine è a colori (3 canali) o scala di grigi
        if len(orig.shape) == 3 and orig.shape[2] == 3:
            for channel in range(3):  # BGR in OpenCV
                # MODIFICA 4: data_range=1023
                ssim_channel = ssim(orig[:,:,channel], decomp[:,:,channel], 
                                   data_range=DATA_RANGE, win_size=11, channel_axis=None)
                ssim_channels.append(ssim_channel)
            current_ssim = np.mean(ssim_channels)
        else:
            # Caso Grayscale
            current_ssim = ssim(orig, decomp, 
                               data_range=DATA_RANGE, win_size=11, channel_axis=None)
            
        ssim_values.append(current_ssim)
        
        # Debug primo frame
        if i == 0:
            # Prende un pixel centrale per controllo
            mid_h, mid_w = orig.shape[0]//2, orig.shape[1]//2
            # Gestione sicura index se mono-canale
            if len(orig.shape) == 3:
                val_orig = orig[mid_h, mid_w, 0] # Canale Blue
                val_dec = decomp[mid_h, mid_w, 0]
            else:
                val_orig = orig[mid_h, mid_w]
                val_dec = decomp[mid_h, mid_w]
                
            print(f"Check Pixel ({mid_w},{mid_h}) - Orig: {val_orig:.0f}, Dec: {val_dec:.0f} (Max atteso ~1023)")

    # Calcola statistiche
    valid_psnr = [x for x in psnr_values if x != float('inf')]
    if len(valid_psnr) < len(psnr_values):
        print(f"Nota: {len(psnr_values) - len(valid_psnr)} frame erano identici (PSNR infinito)")
    
    # Se tutti infiniti, gestisci il caso
    if not valid_psnr and len(psnr_values) > 0:
        psnr_mean = float('inf')
        psnr_std = 0
        psnr_min = float('inf')
        psnr_max = float('inf')
    elif valid_psnr:
        psnr_mean = np.mean(valid_psnr)
        psnr_std = np.std(valid_psnr)
        psnr_min = np.min(valid_psnr)
        psnr_max = np.max(valid_psnr)
    else:
        psnr_mean = 0 # Fallback vuoto
        
    ssim_mean = np.mean(ssim_values)
    ssim_std = np.std(ssim_values)
    ssim_min = np.min(ssim_values)
    ssim_max = np.max(ssim_values)
    
    # PSNR Report
    print(f"\n PSNR (Peak Signal-to-Noise Ratio) [dB]:")
    if psnr_mean == float('inf'):
         print(f"   Media:    INFINITO (Lossless)")
    else:
        print(f"   Media:    {psnr_mean:6.2f} dB")
        print(f"   Dev Std:  {psnr_std:6.2f} dB")
        print(f"   Min:      {psnr_min:6.2f} dB")
        print(f"   Max:      {psnr_max:6.2f} dB")
    
    # SSIM Report
    print(f"\n SSIM (Structural Similarity):")
    print(f"   Media:    {ssim_mean:8.6f}")
    print(f"   Dev Std:  {ssim_std:8.6f}")
    print(f"   Min:      {ssim_min:8.6f}")
    print(f"   Max:      {ssim_max:8.6f}")
    
    # Valutazione qualità
    print(f"\nVALUTAZIONE SINTETICA (Scala 10-bit):")
    print("=" * 40)
    
    # Valutazione PSNR (Scala leggermente adattata per 10 bit, ma dB sono dB)
    if psnr_mean == float('inf'): psnr_rating = "LOSSLESS"
    elif psnr_mean > 50: psnr_rating = "ECCELLENTE"
    elif psnr_mean > 40: psnr_rating = "OTTIMO"
    elif psnr_mean > 30: psnr_rating = "BUONO"
    elif psnr_mean > 20: psnr_rating = "ACCETTABILE"
    else: psnr_rating = "SCADENTE"
    
    # Valutazione SSIM
    if ssim_mean > 0.99: ssim_rating = "ECCELLENTE" # Alzato asticella per 10 bit
    elif ssim_mean > 0.95: ssim_rating = "OTTIMO"
    elif ssim_mean > 0.85: ssim_rating = "BUONO"
    elif ssim_mean > 0.75: ssim_rating = "DISCRETO"
    else: ssim_rating = "SCADENTE"
    
    print(f"Qualità Compressione (PSNR): {psnr_rating}")
    print(f"Fedeltà Strutturale (SSIM):  {ssim_rating}")
    
    return {
        'compression_ratio': ratio,
        'savings_percent': savings,
        'psnr_mean': psnr_mean,
        'ssim_mean': ssim_mean
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analisi completa compressione e qualità immagini 10-bit',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("dataset_name", help="Name of the dataset folder (e.g., bikes, cars)")
    parser.add_argument("--lenslet", action=argparse.BooleanOptionalAction, help="Whether the dataset is of the lenslet type", required=True)
    parser.add_argument("-c", "--codec", help="Codec used for compression (jpl, av1, hevc, vp9, epi_hevc, epi_av1, epi_vp9)", required=True,
                        choices=["jpl", "av1", "hevc", "vp9", "epi_hevc", "epi_av1", "epi_vp9"])
    
    args = parser.parse_args()
    
    BASE_DATASETS_PATH = "./datasets"
    dataset_base_path = Path(BASE_DATASETS_PATH) / args.dataset_name

    # Original folder
    original_folder = dataset_base_path / "RAW" / ("PPM" if not args.lenslet else "PPM_shifted")
    if not os.path.exists(original_folder):
        print(f"ERRORE: Cartella originale non trovata: {original_folder}")
        sys.exit(1)
    else:
        print(f"Original: {original_folder}")
    
    # Decompressed folder
    if args.codec == "jpl":
        decompressed_folder = dataset_base_path / "decoded" / "PPM" / (args.dataset_name if not args.lenslet else f"{args.dataset_name}_shifted")
    elif args.codec in ["av1", "hevc", "vp9"]:
        decompressed_folder = dataset_base_path / "RAW" / "codec_comparison_timing" / "decompressed" / f"decompressed_{args.codec}"
    else:  # epi_hevc, epi_av1, epi_vp9
        decompressed_folder = dataset_base_path / "RAW" / "epi_comparison" / "decompressed" / args.codec.split('_')[1]

    if not os.path.exists(decompressed_folder):
        print(f"ERRORE: Cartella decompressa non trovata: {decompressed_folder}")
        sys.exit(1)
    else:
        print(f"Decompressed: {decompressed_folder}")
    
    # Compressed file path
    if args.codec == "jpl":
        compressed_file = dataset_base_path / "encoded" / f"{args.dataset_name}.jpl"
    elif args.codec in ["av1", "hevc", "vp9"]:
        codec = args.codec
        compressed_file = dataset_base_path / "RAW" / "codec_comparison_timing" / "compressed" / f"compressed_{codec}.{'mkv' if codec == 'av1' else 'mp4' if codec == 'hevc' else 'webm'}"
    else:  # epi_hevc, epi_av1, epi_vp9
        codec = args.codec.split('_')[1]
        compressed_file = dataset_base_path / "RAW" / "epi_comparison" / "compressed" / f"epi_{codec}.{'mkv' if codec == 'av1' else 'mp4' if codec == 'hevc' else 'webm'}"

    if not os.path.exists(compressed_file):
        print(f"ERRORE: File compresso non trovato: {compressed_file}")
        sys.exit(1)
    else:
        print(f"Compressed file: {compressed_file}")

    comparison(original_folder, decompressed_folder, compressed_file)