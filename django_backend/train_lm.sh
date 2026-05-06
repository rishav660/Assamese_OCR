#!/bin/bash
# Script to build KenLM and train an N-gram Language Model on Colab/Linux

echo "========================================="
echo "1. Installing dependencies..."
echo "========================================="
apt-get update
apt-get install -y cmake g++ libboost-all-dev
pip install pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip

echo "========================================="
echo "2. Downloading and building KenLM binaries..."
echo "========================================="
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
cd ../..

echo "========================================="
echo "3. Preparing corpus..."
echo "========================================="
# Clean the corpus (remove english/numbers, keep only Assamese script and spaces)
python -c "
import re
with open('data/as-wiki-2021.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# Keep only Assamese characters (U+0980-U+09FF) and whitespace
text = re.sub(r'[^\u0980-\u09FF\s]', ' ', text)
text = re.sub(r'\s+', ' ', text)
with open('data/corpus_clean.txt', 'w', encoding='utf-8') as f:
    f.write(text)
"

echo "========================================="
echo "4. Training 4-gram Language Model..."
echo "========================================="
./kenlm/build/bin/lmplz -o 4 < data/corpus_clean.txt > data/assamese_lm.arpa

echo "========================================="
echo "5. Compiling to binary format (for faster loading)..."
echo "========================================="
./kenlm/build/bin/build_binary data/assamese_lm.arpa data/assamese_lm.bin

echo "✅ Language model successfully trained: data/assamese_lm.bin"
