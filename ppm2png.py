import argparse
import sys
import time
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Importiamo tqdm per la barra di caricamento
try:
    from tqdm import tqdm
except ImportError:
    print("❌ Errore: La libreria 'tqdm' non è installata.")
    print("   Esegui: pip install tqdm")
    sys.exit(1)

def converti_singolo_file(dati_file):
    """
    Funzione worker eseguita in parallelo.
    """
    path_input, path_output_dir = dati_file
    
    try:
        nome_file = path_input.stem + ".png"
        destinazione = path_output_dir / nome_file
        
        with Image.open(path_input) as img:
            # compress_level=1 per massima velocità
            img.save(destinazione, format="PNG", compress_level=1)
            
        return True # Successo
    except Exception:
        return False # Errore

def main():
    # Setup argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Convertitore PPM -> PNG Parallelo con Status Bar")
    parser.add_argument("input_dir", type=str, help="Cartella con i file .ppm")
    parser.add_argument("output_dir", type=str, help="Cartella dove salvare i .png")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Controlli cartelle
    if not input_path.exists():
        print(f"❌ Errore: La cartella '{args.input_dir}' non esiste.")
        sys.exit(1)
        
    output_path.mkdir(parents=True, exist_ok=True)

    # Trova i file
    files_ppm = list(input_path.glob("*.ppm"))
    total_files = len(files_ppm)

    if total_files == 0:
        print("⚠️  Nessun file .ppm trovato.")
        return

    print(f"🚀 Trovati {total_files} file. Avvio conversione parallela...\n")

    # Preparazione task
    tasks = [(f, output_path) for f in files_ppm]
    
    successi = 0
    errori = 0

    # Avvio del pool di processi
    with ProcessPoolExecutor() as executor:
        # executor.map avvia i processi.
        # tqdm avvolge questo iteratore per mostrare la barra.
        # total=total_files dice a tqdm quanto è lunga la lista.
        
        risultati = list(tqdm(
            executor.map(converti_singolo_file, tasks), 
            total=total_files, 
            unit="img",            # Unità di misura
            desc="Convertendo",    # Testo a sinistra della barra
            colour="green"         # Colore della barra
        ))

    # Conteggio finale risultati
    for esito in risultati:
        if esito:
            successi += 1
        else:
            errori += 1

    print("\n" + "="*40)
    print(f"✅ Completato!")
    print(f"   Successi: {successi}")
    print(f"   Errori:   {errori}")
    print("="*40)

if __name__ == "__main__":
    main()