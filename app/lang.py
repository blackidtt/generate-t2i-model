"""Deteksi sederhana teks yang kemungkinan bukan bahasa Inggris.

Constraint penelitian: semua teks yang diketik partisipan harus berbahasa Inggris. Deteksi ini hanya
memberi peringatan di UI dan menandai log. Sesi tidak dihentikan.
"""
import re

ID_MARKERS = {
    "yang", "dan", "dengan", "dari", "untuk", "tidak", "lebih", "tambah", "tambahkan", "ganti", "hapus",
    "gelap", "terang", "warna", "latar", "belakang", "gambar", "jadi", "saya", "aku", "kamu", "bikin",
    "buat", "buatkan", "sedikit", "banget", "aja", "kucing", "biru", "merah", "hijau", "besar", "kecil",
    "tolong", "agar", "supaya", "pakai", "seperti", "ada", "itu", "ini", "malam", "siang",
}


def non_english_suspected(text: str) -> bool:
    words = re.findall(r"[a-zA-ZÀ-ɏ]+", text.lower())
    if not words:
        return False
    if any(ord(c) > 127 for c in text if c.isalpha()):
        return True
    return any(w in ID_MARKERS for w in words)
