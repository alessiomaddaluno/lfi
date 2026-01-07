import sys

def analyze_ppm_raw(filepath):
    print(f"--- ANALISI RAW DEL FILE: {filepath} ---")
    
    with open(filepath, 'rb') as f:
        # 1. Lettura Header (byte per byte finché non troviamo l'ultimo newline)
        header = b""
        while True:
            byte = f.read(1)
            header += byte
            # Cerchiamo la fine dell'header (whitespace dopo il maxval)
            # Un header PPM tipico finisce dopo il 4° numero (MaxVal) e un carattere whitespace
            parts = header.split()
            if len(parts) >= 4 and byte.isspace():
                # Magic, W, H, MaxVal
                break
        
        # Pulizia header dai commenti
        header_str = header.decode('ascii', errors='ignore')
        print("HEADER RILEVATO:")
        print(f"'{header_str.strip()}'")
        
        parts = header_str.split()
        magic_number = parts[0]
        width = int(parts[1])
        height = int(parts[2])
        max_val = int(parts[3])
        
        print(f"\nINTERPRETAZIONE:")
        print(f"Format: {magic_number}")
        print(f"Size:   {width}x{height}")
        print(f"MaxVal: {max_val} (Importante!)")
        
        is_16bit = max_val > 255
        print(f"Bit-depth: {'16-bit' if is_16bit else '8-bit'}")

        # 2. Lettura dei primi pixel grezzi
        # Leggiamo il primo pixel (3 canali: R, G, B)
        print("\n--- ANALISI BYTE PRIMO PIXEL ---")
        
        if is_16bit:
            # Legge 6 byte (2 per R, 2 per G, 2 per B)
            raw_bytes = f.read(6)
            print(f"Byte grezzi su disco (Hex): {raw_bytes.hex(' ')}")
            
            # Interpretazione Big Endian (Standard PPM)
            r_be = (raw_bytes[0] << 8) | raw_bytes[1]
            g_be = (raw_bytes[2] << 8) | raw_bytes[3]
            b_be = (raw_bytes[4] << 8) | raw_bytes[5]
            
            # Interpretazione Little Endian (Come potrebbe leggerlo OpenCV sbagliando)
            r_le = (raw_bytes[1] << 8) | raw_bytes[0]
            g_le = (raw_bytes[3] << 8) | raw_bytes[2]
            b_le = (raw_bytes[5] << 8) | raw_bytes[4]
            
            print(f"\nSe letto come BIG ENDIAN (Corretto per PPM):")
            print(f"  R: {r_be:5d} | G: {g_be:5d} | B: {b_be:5d}")
            
            print(f"Se letto come LITTLE ENDIAN (Errore comune):")
            print(f"  R: {r_le:5d} | G: {g_le:5d} | B: {b_le:5d}")
            
        else:
            # 8-bit
            raw_bytes = f.read(3)
            r, g, b = raw_bytes[0], raw_bytes[1], raw_bytes[2]
            print(f"Byte grezzi: {raw_bytes.hex(' ')}")
            print(f"Valori: R={r}, G={g}, B={b}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python raw_check.py immagine.ppm")
    else:
        analyze_ppm_raw(sys.argv[1])