# Template Modul Praktikum Kecerdasan Buatan

Template website modul praktikum menggunakan **MkDocs + Material for MkDocs + GitHub Pages**.

## Struktur

- `docs/` — seluruh materi Markdown
- `mkdocs.yml` — konfigurasi website dan menu
- `.github/workflows/deploy.yml` — deployment otomatis ke GitHub Pages
- `requirements.txt` — dependency Python

## Cara Menggunakan di GitHub

1. Buat repository baru, misalnya `modul-ai`.
2. Upload seluruh isi folder template ini ke repository.
3. Pastikan branch utama bernama `main` atau `master`.
4. Push/commit file.
5. Buka tab **Actions** dan tunggu workflow `Deploy MkDocs to GitHub Pages` berhasil.
6. Buka **Settings > Pages**.
7. Pada **Build and deployment**, pilih **Deploy from a branch**.
8. Pilih branch `gh-pages` dan folder `/ (root)`, lalu Save.
9. Website akan tersedia di `https://USERNAME.github.io/NAMA-REPOSITORY/`.

## Menjalankan di Komputer

```bash
pip install -r requirements.txt
mkdocs serve
```

Kemudian buka `http://127.0.0.1:8000`.

## Mengedit Materi

Edit file seperti:

```text
docs/modul02.md
```

Gunakan Markdown:

```markdown
# Judul
## Subjudul

```python
print("Hello AI")
```
```

Setelah di-commit dan push, website akan diperbarui otomatis.
