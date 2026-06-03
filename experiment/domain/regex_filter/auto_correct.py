"""
auto_correct.py — Auto-corrector for OCR plate strings based on Turkish plate regex formats.

Corrects common OCR mistakes strictly in positions that require a specific character class (digit or letter).
"""

import re

class OCRAutoCorrector:
    """
    Intelligently auto-corrects plate strings by assessing all valid Turkish plate partitions
    (2 digits, 1-3 letters, 2-4 digits) and choosing the one that requires the minimum number 
    of common character-level corrections (e.g., '0' <-> 'O', '8' <-> 'B').
    """

    def __init__(self):
        # Letters that commonly look like digits
        self.letter_to_digit = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1',
            'Z': '2',
            'E': '3',
            'A': '4',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8'
        }
        
        # Digits that commonly look like letters
        self.digit_to_letter = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '3': 'E',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B'
        }

    def _cost_and_fix_digits(self, text: str) -> tuple[int, str]:
        cost = 0
        fixed = []
        for char in text:
            if char.isdigit():
                fixed.append(char)
            elif char in self.letter_to_digit:
                fixed.append(self.letter_to_digit[char])
                cost += 1
            else:
                fixed.append(char)
                cost += 100  # heavy penalty for impossible/unlikely conversion
        return cost, "".join(fixed)

    def _cost_and_fix_letters(self, text: str) -> tuple[int, str]:
        cost = 0
        fixed = []
        for char in text:
            if char.isalpha():
                fixed.append(char)
            elif char in self.digit_to_letter:
                fixed.append(self.digit_to_letter[char])
                cost += 1
            else:
                fixed.append(char)
                cost += 100  # heavy penalty for impossible/unlikely conversion
        return cost, "".join(fixed)

    def apply(self, text: str) -> str:
        if not text:
            return ""

        # Remove whitespaces and upper
        text = re.sub(r'\s+', '', text.upper())
        
        # A valid pure Turkish plate length is between 5 and 9 characters
        if len(text) < 5 or len(text) > 9:
            return text

        best_cost = float('inf')
        best_result = text

        # Partition into (2 digits, 1..3 letters, 2..4 digits)
        for l_len in [1, 2, 3]:
            for d_len in [2, 3, 4]:
                if 2 + l_len + d_len == len(text):
                    part1 = text[0:2]
                    part2 = text[2:2+l_len]
                    part3 = text[2+l_len:]

                    cost1, fix1 = self._cost_and_fix_digits(part1)
                    cost2, fix2 = self._cost_and_fix_letters(part2)
                    cost3, fix3 = self._cost_and_fix_digits(part3)

                    total_cost = cost1 + cost2 + cost3
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        # We return it continuously without spaces to let regex filter do the spacing
                        best_result = f"{fix1}{fix2}{fix3}"

        # If it's absolutely impossible to correct without hard violations, just return original
        if best_cost >= 100:
            return text

        return best_result
