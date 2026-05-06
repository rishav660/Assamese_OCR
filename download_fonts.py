import os
import urllib.request

fonts_to_download = [
    ("HindSiliguri-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/hindsiliguri/HindSiliguri-Regular.ttf"),
    ("HindSiliguri-Bold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/hindsiliguri/HindSiliguri-Bold.ttf"),
    ("Galada-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/galada/Galada-Regular.ttf"),
    ("Mina-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/mina/Mina-Regular.ttf"),
    ("TiroBangla-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/tirobangla/TiroBangla-Regular.ttf"),
    ("AnekBangla-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/anekbangla/AnekBangla-Regular.ttf"),
    ("BalooDa2-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/balooda2/BalooDa2-Regular.ttf"),
]

fonts_dir = os.path.join("django_backend", "fonts")
if not os.path.exists(fonts_dir):
    os.makedirs(fonts_dir, exist_ok=True)

new_fonts = []

print("Downloading fonts from GitHub...")

for font_name, url in fonts_to_download:
    dest_path = os.path.join(fonts_dir, font_name)
    print(f"Downloading {font_name}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        new_fonts.append(f"fonts/{font_name}")
        print(f"  -> Saved {font_name}")
    except Exception as e:
        print(f"  -> Failed: {e}")

# Append new fonts to working_fonts.txt
working_fonts_file = os.path.join(fonts_dir, "working_fonts.txt")
existing_fonts = set()
if os.path.exists(working_fonts_file):
    with open(working_fonts_file, "r", encoding="utf-8") as f:
        existing_fonts = set(line.strip() for line in f if line.strip())

added_count = 0
with open(working_fonts_file, "a", encoding="utf-8") as f:
    for font_path in new_fonts:
        if font_path not in existing_fonts:
            f.write(f"{font_path}\n")
            added_count += 1

print(f"\nDone! Added {added_count} new font variations to {working_fonts_file}.")
