
import os
from collections import Counter
import re

class SpellChecker:
    def __init__(self, corpus_path):
        self.words = Counter()
        self.total_words = 0
        self.load_corpus(corpus_path)

    def load_corpus(self, path):
        print(f"Loading corpus from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Simple tokenization: split by whitespace and strip punctuation
                # Assamese specific: Keep characters, numbers, and common punctuation if needed
                # For now, let's just split by whitespace and strip common non-word chars
                tokens = text.split()
                cleaned_tokens = [self.clean_word(t) for t in tokens]
                self.words.update([t for t in cleaned_tokens if t])
                self.total_words = sum(self.words.values())
            print(f"Loaded {len(self.words)} unique words.")
        except Exception as e:
            print(f"Error loading corpus: {e}")

    def clean_word(self, word):
        # Remove common punctuation from ends
        return word.strip('.,!?।"\'()[]{}')

    def P(self, word):
        "Probability of `word`."
        return self.words[word] / self.total_words

    def correction(self, word):
        "Most probable spelling correction for word."
        # If word is known, return it
        if word in self.words:
            return word
        
        # Get candidates
        candidates = self.candidates(word)
        
        # If no candidates, return original word
        if not candidates:
            return word
            
        # Return candidate with highest probability
        return max(candidates, key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        # 1. Known word (already checked in correction, but good for logic)
        if word in self.words:
            return {word}
            
        # 2. Edit distance 1
        ed1 = self.known(self.edits1(word))
        if ed1:
            return ed1
            
        # 3. Edit distance 2 (only if word is short enough to justify cost, or just skip for speed)
        # For now, let's stick to edits1 for performance, or a very restricted edits2
        # ed2 = self.known(self.edits2(word))
        # if ed2:
        #    return ed2
        
        return {word}

    def known(self, words):
        "The subset of `words` that appear in the dictionary of words."
        return set(w for w in words if w in self.words)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযৰলৱশষসহক্ষড়ঢ়য়ৎংঃঁািীুূৃেৈোৌ্' # Assamese characters
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

# Singleton instance
_spell_checker = None

def get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        corpus_path = os.path.join(os.path.dirname(__file__), 'data', 'as-wiki-2021.txt')
        if os.path.exists(corpus_path):
            _spell_checker = SpellChecker(corpus_path)
        else:
            print("Corpus not found, spell checker disabled.")
    return _spell_checker

def correct_sentence(sentence):
    checker = get_spell_checker()
    if not checker:
        return sentence
        
    words = sentence.split()
    corrected_words = []
    for w in words:
        # Keep punctuation attached
        prefix = ""
        suffix = ""
        clean = w
        
        # Simple punctuation stripping
        if w and w[0] in '(["\'':
            prefix = w[0]
            clean = w[1:]
        if clean and clean[-1] in '.,!?"\')]।':
            suffix = clean[-1]
            clean = clean[:-1]
            
        if clean:
            corrected = checker.correction(clean)
            corrected_words.append(prefix + corrected + suffix)
        else:
            corrected_words.append(w)
            
    return " ".join(corrected_words)
