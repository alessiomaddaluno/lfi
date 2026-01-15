# Light Field Images Compression

This repository contains the project developed for the *Data Compression* exam.
The goal is to evaluate the performance of different codecs for **compressing Light Field Images (LFI)** through benchmarks on different datasets and a written report.

## 📄 Report
TODO

## 🖥️ Experimental Setup
- **CPU:** AMD Ryzen 7 6800H (8 cores, SMT, 3.2 GHz base, up to 4.7 GHz boost)  
- **GPU:** NVIDIA GeForce RTX 3060 Mobile (3840 CUDA cores, 900 MHz base, up to 1425 MHz boost)  
- **OS:** Windows 11 with WSL

## 📂 Repository Structure
```text
├── codec_video.py      # Video codec compressor and decompressor (HEVC, AV1, VP9)
├── compare_debug.py    # Visual comparison of images for debugging purposes
├── compare.py          # Evaluate compression performance of every codec for a specific dataset
├── epi_codec_video.py  # Video codec compressor and decompressor which uses EPI images (HEVC, AV1, VP9)
├── jpl_processor.py    # JPEG Pleno compressor and decompressor
├── raw_check.py        # Header Check for PPM files
├── report.pdf          # Final report
└── results.xlsx        # Excel document with benchmarks results
```

## ▶️ How to Run

### Requirements
- Python 3
- [FFMPEG](https://www.ffmpeg.org/)
- [JPEG Pleno](https://gitlab.com/wg1/jpeg-pleno-refsw)

### Datasets
[JPEG Pleno Light Field Datasets](https://plenodb.jpeg.org/lf/pleno_lf)

### Execution
TODO

## 📊 Results Summary
TODO

## 📌 Conclusion
TODO