import sys
import numpy as np
import cv2
import os
import glob
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def debug_comparison(original_folder, decompressed_folder):
    
    def load_paths(folder):
        # Carica solo i percorsi, non le immagini subito
        paths = sorted([p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith('.ppm')])
        names = [os.path.basename(p) for p in paths]
        return paths, names

    print("=== DEBUG MODE: ANALISI VISIVA DELL'ERRORE ===")
    
    orig_paths, orig_names = load_paths(original_folder)
    decomp_paths, decomp_names = load_paths(decompressed_folder)
    
    num_frames = min(len(orig_paths), len(decomp_paths))
    
    if num_frames == 0:
        print("Errore: Nessun frame trovato.")
        return

    print(f"Confronto i primi frame...")
    
    # Analizziamo solo il primo frame per capire il problema
    i = 0 
    
    # Carica immagini
    img_orig = cv2.imread(orig_paths[i])
    img_decomp = cv2.imread(decomp_paths[i])
    
    print(f"\nCONFRONTO COPPIA {i}:")
    print(f"File Originale:   {orig_names[i]}")
    print(f"File Decompresso: {decomp_names[i]}")
    print(f"Dimensioni Orig:  {img_orig.shape}")
    print(f"Dimensioni Dec:   {img_decomp.shape}")
    
    # 1. CONTROLLO DIMENSIONI
    if img_orig.shape != img_decomp.shape:
        print("!!! ATTENZIONE: LE DIMENSIONI SONO DIVERSE !!!")
        print("Questo rende il calcolo dell'MSE enorme anche se le immagini sembrano uguali.")
        # Resize brutale per permettere il calcolo (solo per debug)
        img_decomp = cv2.resize(img_decomp, (img_orig.shape[1], img_orig.shape[0]))
    
    # Converti per calcoli
    orig_float = img_orig.astype(np.float64)
    decomp_float = img_decomp.astype(np.float64)
    
    # 2. CALCOLO METRICHE MANUALI
    diff = orig_float - decomp_float
    mse = np.mean(diff ** 2)
    psnr_val = 10 * np.log10((255.0**2) / mse) if mse > 0 else float('inf')
    
    print(f"\nRISULTATI MATEMATICI:")
    print(f"MSE:  {mse:.4f} (Dovrebbe essere < 10 per buona qualità)")
    print(f"PSNR: {psnr_val:.2f} dB")
    
    # 3. GENERA IMMAGINE DIFFERENZA (La prova del nove)
    # Calcoliamo il valore assoluto della differenza e lo amplifichiamo per vederlo bene
    diff_abs = np.abs(orig_float - decomp_float)
    
    # Amplifichiamo l'errore x10 per renderlo visibile all'occhio umano
    # I pixel neri significano "identico", i pixel colorati sono l'errore.
    diff_visual = np.clip(diff_abs * 10, 0, 255).astype(np.uint8)
    
    output_filename = "DEBUG_diff_frame_0.jpg"
    cv2.imwrite(output_filename, diff_visual)
    
    print(f"\n Generata immagine di debug: '{output_filename}'")
    print("-" * 50)
    print("ANALISI DELL'IMMAGINE OUTPUT:")
    print("1. Apri l'immagine generata.")
    print("2. È COMPLETAMENTE NERA? -> Le immagini sono identiche, errore nel codice.")
    print("3. Vedi i CONTORNI (Fantasmi)? -> Le immagini sono disallineate (Shift o frame sbagliato).")
    print("4. Vedi RUMORE CASUALE (Neve)? -> Compressione lossy normale.")
    print("5. Vedi COLORI STRANI uniformi? -> Canali invertiti o luminosità sballata.")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--original', required=True)
    parser.add_argument('-d', '--decompressed', required=True)
    args = parser.parse_args()
    
    debug_comparison(args.original, args.decompressed)