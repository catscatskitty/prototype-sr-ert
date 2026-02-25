import re
import numpy as np

class LinguisticAnalyzer:
    def __init__(self):
        self.vowels = set('аеёиоуыэюяaeiou')
        self.consonants = set('бвгджзйклмнпрстфхцчшщbcdfghjklmnpqrstvwxyz')
        
    def analyze(self, text):
        if not text or not isinstance(text, str):
            return {
                'word_count': 0,
                'char_count': 0,
                'vowel_count': 0,
                'consonant_count': 0,
                'avg_word_length': 0,
                'has_cyrillic': False,
                'has_latin': False
            }
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        char_count = len(text)
        
        vowel_count = sum(1 for c in text.lower() if c in self.vowels)
        consonant_count = sum(1 for c in text.lower() if c in self.consonants)
        
        has_cyrillic = bool(re.search('[а-яё]', text.lower()))
        has_latin = bool(re.search('[a-z]', text.lower()))
        
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'vowel_count': vowel_count,
            'consonant_count': consonant_count,
            'avg_word_length': avg_word_length,
            'has_cyrillic': has_cyrillic,
            'has_latin': has_latin
        }
    
    def extract_features(self, text):
        features = self.analyze(text)
        return np.array([[
            features['word_count'],
            features['char_count'],
            features['vowel_count'],
            features['consonant_count'],
            features['avg_word_length'],
            1 if features['has_cyrillic'] else 0,
            1 if features['has_latin'] else 0
        ]])

LinguisticFeatureExtractor = LinguisticAnalyzer