import os
import re
from symspellpy import SymSpell, Verbosity

class AssameseSpellChecker:
    def __init__(self, corpus_path):
        # max_dictionary_edit_distance=2 allows 2 edits (very slow in Norvig, fast in SymSpell)
        # prefix_length=7 is optimal for performance
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.corpus_path = corpus_path
        self._build_and_load_dict()

    def _build_and_load_dict(self):
        dict_path = self.corpus_path.replace('.txt', '_symspell.txt')
        
        # If the compiled dictionary doesn't exist, we must build it from the corpus
        if not os.path.exists(dict_path):
            print(f"Building SymSpell dictionary from {self.corpus_path}...")
            # symspellpy natively supports creating a dictionary from a corpus file
            self.sym_spell.create_dictionary(self.corpus_path, "utf-8")
            self.sym_spell.save_pickle(dict_path)
            print(f"Saved compiled dictionary to {dict_path}")
        else:
            print(f"Loading existing SymSpell dictionary from {dict_path}...")
            self.sym_spell.load_pickle(dict_path)
            print(f"Loaded dictionary with {len(self.sym_spell.words)} words.")

    def correct(self, word):
        # 1. Skip very short words or non-Assamese words
        if len(word) < 2 or not re.search(r'[\u0980-\u09FF]', word):
            return word

        # 2. Get suggestions (Verbosity.CLOSEST returns the best matches)
        suggestions = self.sym_spell.lookup(
            word, 
            Verbosity.CLOSEST, 
            max_edit_distance=2, 
            include_unknown=True
        )

        if not suggestions:
            return word

        best_suggestion = suggestions[0]

        # 3. Confidence Gating
        # Don't "correct" if the edit distance is large compared to word length,
        # or if the original word is actually in the dictionary (distance 0).
        if best_suggestion.distance == 0:
            return word
        
        # If it requires 2 edits for a 3-letter word, it's likely destroying the word.
        if best_suggestion.distance >= 2 and len(word) <= 4:
            return word
            
        # Return the corrected term
        return best_suggestion.term


# Singleton instance
_spell_checker = None

def get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        corpus_path = os.path.join(os.path.dirname(__file__), 'data', 'as-wiki-2021.txt')
        if os.path.exists(corpus_path):
            try:
                import symspellpy
                _spell_checker = AssameseSpellChecker(corpus_path)
            except ImportError:
                print("[WARNING] symspellpy not installed! Run: pip install symspellpy")
                print("Spell checking disabled.")
                return None
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
            corrected = checker.correct(clean)
            corrected_words.append(prefix + corrected + suffix)
        else:
            corrected_words.append(w)
            
    return " ".join(corrected_words)
