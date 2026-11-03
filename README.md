# Demo aplikasi eksperimen: In-Place Editing (IPE) vs Conversational Follow-Up (CFU)

Aplikasi web untuk menjalankan sesi eksperimen sesuai proposal "Pengaruh Strategi Revisi Prompt terhadap Jumlah Iterasi pada Pembuatan Gambar Text-to-Image". Satu partisipan mengerjakan dua tugas dengan satu gambar target yang sama, satu tugas per kondisi, dengan urutan kondisi yang di-counterbalance.

## Komponen (sesuai proposal bagian 5.4)

| Komponen | Pilihan | File |
|---|---|---|
| Model gambar | SDXL-Turbo, 1 langkah, guidance 0, 512x512, seed tetap | `app/image_gen.py` |
| LLM penerjemah (mode CFU) | Qwen3-4B lewat Ollama, temperature 0, seed tetap, instruksi sistem perubahan minimal | `app/llm.py` |
| Skor kemiripan | Kosinus embedding gambar CLIP ViT-B/32 (open_clip), ambang 0,85 | `app/similarity.py` |
| Constraint bahasa | Semua teks partisipan wajib bahasa Inggris. Deteksi hanya menandai log dan menampilkan peringatan | `app/lang.py` |
| Logika sesi, counterbalancing, log | | `app/core.py` |
| API dan antarmuka | FastAPI, satu halaman HTML | `app/main.py`, `static/index.html` |
| Analisis | RQ1, DV pendukung, efek urutan, ekspor pengodean, kappa, RQ2 | `scripts/analyze.py` |

## Mencoba tanpa GPU (mode MOCK)

Mode ini memakai gambar kartu warna dan skor tiruan (tumpang tindih kata dengan prompt rahasia). Gunanya hanya untuk mencoba alur dan antarmuka, bukan untuk data penelitian.

```bash
pip install fastapi "uvicorn[standard]" pillow requests numpy pandas scipy statsmodels scikit-learn
MOCK=1 python -m scripts.make_target "a red fox sitting in a snowy forest, golden sunrise light, detailed fur"
MOCK=1 uvicorn app.main:app --port 8000
```

Buka http://localhost:8000. Tambahkan `?order=AB` atau `?order=BA` pada URL untuk memaksa urutan (A = IPE dulu, B = CFU dulu).

## Menjalankan sungguhan

```bash
pip install -r requirements.txt            # torch sebaiknya dipasang sesuai CUDA di mesin Anda
ollama pull qwen3:4b                       # butuh Ollama versi terbaru agar "think": false didukung
python -m scripts.make_target "A tiny fluffy kitten is guarding a very large Golden Retriever that is peacefully sleeping. The kitten sits upright beside the dog with a serious, determined expression, acting like a tiny bodyguard protecting its giant friend. The dramatic size difference between the tiny kitten and the huge dog is clearly emphasized, creating an adorable and funny contrast. Cute, heartwarming, playful atmosphere, highly detailed fluffy fur, expressive face, soft natural lighting, photorealistic."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

1. `make_target` membuat `static/target.png` dari prompt rahasia dengan seed yang sama dengan aplikasi, dan menyimpan prompt itu di `data/secret/target_prompt.txt`. Karena target dibuat dari prompt rahasia, ambang kemiripan pasti bisa dicapai. Jangan beri tahu prompt ini ke partisipan.
2. Unduhan pertama SDXL-Turbo dan CLIP berjalan saat permintaan generate pertama. Panaskan dulu dengan satu sesi latihan.
3. Pengaturan lewat environment variable: `THRESHOLD`, `SEED`, `SDXL_MODEL`, `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_KEEP_ALIVE`, `SHOW_LLM_PROMPT`, `SHOW_SCORE`, `WARMUP`, `DATA_DIR`.
4. Saat server menyala, SDXL-Turbo, CLIP, dan Qwen3 dimuat ke memori dan waktunya dicetak di terminal (`[warmup] ...`). Server baru siap setelah baris itu selesai. Matikan dengan `WARMUP=0`. Ollama menjaga Qwen3 tetap di memori selama 60 menit terakhir penggunaan.

## Melihat skor CLIP

Partisipan tidak melihat skor. Peneliti bisa melihatnya di:

- `data/logs/events.jsonl`, kolom `score` dan `gen_s` (detik untuk satu iterasi) pada baris `iteration`.
- `python -m scripts.pilot_threshold P01`, yang mencetak skor per iterasi.
- Saat pilot, jalankan dengan `SHOW_SCORE=1`. Skor dan waktu generate muncul di bawah gambar pada layar tugas (bukan layar latihan). Jangan aktifkan ini saat mengumpulkan data penelitian.

## Alur sesi per partisipan

`consent` > `practice_1` > `practice_2` > `task_1` > `break` > `task_2` > `done`

- Urutan kondisi ditentukan saat pendaftaran kode partisipan. Sistem menetapkan urutan yang jumlahnya lebih sedikit (AB atau BA), dan memilih acak saat seimbang, sehingga akhirnya mendekati 15 dan 15 untuk N=30.
- Latihan: kedua mode dicoba dengan prompt bebas, tanpa gambar target dan tanpa skor (gambar target baru muncul di tugas). Latihan dicatat sebagai `practice_step`, dan penanda bahasa non-Inggris di dalamnya dapat dipakai peneliti untuk menyaring peserta yang tidak mampu menulis bahasa Inggris.
- Tugas: partisipan melihat gambar target dan tidak melihat skor. Iterasi ke-1 adalah gambar dari prompt awal, tiap revisi menambah satu iterasi. Tugas selesai otomatis saat skor mencapai ambang. Tidak ada batas iterasi. Tombol berhenti mencatat tugas sebagai tidak selesai.
- Halaman bisa dimuat ulang di tengah sesi. Tugas dilanjutkan dari iterasi terakhir.

## Data yang tercatat

`data/logs/events.jsonl`, satu baris per kejadian: `register`, `advance`, `practice_step`, `task_start`, `iteration`, `task_end`.
Setiap `iteration` berisi kode partisipan, kondisi, nomor iterasi, `user_input`, `prompt_before`, `prompt_after`, `added_words`, `removed_words`, `words_typed`, `score`, `reached`, `elapsed_s` (dari awal tugas), penanda bahasa, dan kesalahan LLM. Urutan kondisi tercatat di kejadian `register`. Gambar tiap iterasi disimpan di `data/images/<pid>/`, dan status sesi di `data/sessions/<pid>.json`.

## Analisis

```bash
python -m scripts.pilot_threshold P01 P02 P03          # periksa ambang 0,85 dari log pilot
python -m scripts.analyze                              # RQ1, efek urutan, DV pendukung
python -m scripts.analyze --export-coding coding.csv   # tabel revisi untuk dua pengkode
python -m scripts.analyze --coded coding.csv           # kappa (target 0,70) dan RQ2 dengan koreksi Holm
```

Selisih dihitung IPE dikurangi CFU, jadi nilai positif berarti CFU butuh iterasi lebih sedikit (arah H1). Kode revisi diisi dari `user_input` partisipan, bukan dari prompt keluaran LLM. Kategori: Adding descriptors, Adding themes, Adjusting view, Omitting words, Reordering or Rephrasing, dan Lainnya.

## Keputusan implementasi yang tidak tertulis di proposal

Mohon ditinjau kelompok, karena memengaruhi data.

- Di IPE, menekan Generate tanpa mengubah prompt ditolak dan tidak dihitung sebagai iterasi (seed tetap sehingga gambarnya identik).
- Di CFU, pesan pertama adalah prompt awal (iterasi 1), pesan berikutnya adalah instruksi lanjutan. Prompt hasil LLM tidak ditampilkan ke partisipan tetapi tercatat di log (ubah dengan `SHOW_LLM_PROMPT=1`).
- Jika LLM gagal atau mengembalikan keluaran kosong, prompt lama dipertahankan, iterasi tetap dihitung, dan `llm_error` diisi. Perlu diputuskan apakah iterasi seperti ini dibuang saat analisis.
- `words_typed`: di IPE jumlah kata yang ditambahkan menurut diff terhadap prompt sebelumnya, di CFU jumlah kata pada instruksi.
- Waktu dihitung dari saat tugas dimulai (termasuk waktu membaca), bukan dari generate pertama.
- Partisipan boleh berhenti kapan saja. Datanya tercatat `completed=false` dan dikeluarkan dari analisis berpasangan sesuai proposal.

## Batasan

- Jalur mode asli (SDXL-Turbo, CLIP, Ollama) belum diuji end-to-end karena tidak ada GPU dan akses model di lingkungan penyusunan. Yang sudah diuji hanya alur aplikasi, log, counterbalancing, dan skrip analisis dalam mode MOCK. Jalankan satu sesi latihan dan satu tugas penuh sebelum pilot.
- Lisensi SDXL-Turbo di Hugging Face adalah `sai-nc-community`. Periksa ketentuannya untuk kebutuhan penelitian Anda.
- Tidak ada autentikasi. Jalankan hanya di jaringan lokal selama sesi.
