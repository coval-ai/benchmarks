import string
import re

def dehyphenate(text):
    words = text.split()
    result = []
    
    for word in words:
        dehyphenated_word = ""
        i = 0
        while i < len(word):
            if (i > 0 and i < len(word) - 1 and 
                word[i] == '-' and 
                word[i-1].isalnum() and 
                word[i+1].isalnum()):
                dehyphenated_word += ' '
            else:
                dehyphenated_word += word[i]
            i += 1
        
        result.append(dehyphenated_word)
    
    return ' '.join(result)

def sentence_to_numbers(sentence):
    # Dictionary mapping words to numbers
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'hundred': '100', 'thousand': '1000', 'million': '1000000', 'billion': '1000000000'
    }
    
    # Months for date recognition
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    
    # Split the sentence into words
    words = sentence.split()
    result_words = []
    
    i = 0
    while i < len(words):
        current_word = words[i]
        current = current_word.lower().rstrip(',.;:!?')
        
        # Check for date contexts
        is_date_context = False
        if i > 0 and words[i-1].lower().rstrip(',.;:!?') in months:
            is_date_context = True
        elif i > 1 and words[i-2].lower().rstrip(',.;:!?') in months and words[i-1].lower() in ['the']:
            is_date_context = True
            
        # Special handling for year patterns (e.g., "twenty twenty-four" -> "2024")
        if current in word_to_num and i+1 < len(words):
            first_num = int(word_to_num[current])
            next_word = words[i+1].lower().rstrip(',.;:!?')
            
            # Check if this might be a year pattern (twenty twenty-four)
            is_year_pattern = False
            year_value = ""
            
            # Check for decade words (twenty, thirty, etc.)
            if first_num >= 20 and first_num % 10 == 0 and first_num < 100:
                # Check for patterns like "twenty twenty-four"
                if '-' in next_word:
                    parts = next_word.split('-')
                    if len(parts) == 2 and all(part in word_to_num for part in parts):
                        if word_to_num[parts[0]].endswith('0') and int(word_to_num[parts[1]]) < 10:
                            second_num = int(word_to_num[parts[0]]) + int(word_to_num[parts[1]])
                            if second_num < 100:
                                year_value = f"{first_num}{second_num:02d}"
                                is_year_pattern = True
                
                # Check for patterns like "twenty four" (non-hyphenated)
                elif next_word in word_to_num:
                    second_num = int(word_to_num[next_word])
                    
                    # If it looks like "twenty twenty" (for 2020)
                    if second_num >= 20 and second_num % 10 == 0:
                        year_value = f"{first_num}{second_num//10}0"
                        is_year_pattern = True
                    
                    # If it looks like "twenty four" (for year like 2004)
                    elif second_num < 20:
                        year_value = f"{first_num}{second_num:02d}"
                        is_year_pattern = True
            
            # Always use year pattern in date context, or use it if it looks like a valid year
            if (is_date_context or is_year_pattern) and year_value:
                # Preserve any trailing punctuation from the second word
                punctuation = ''.join(c for c in words[i+1] if c in '.,:;!?')
                result_words.append(year_value + punctuation)
                i += 2  # Skip the next word
                continue
        
        # Check if the current word is a hyphenated number (e.g., "twenty-four")
        if '-' in current and any(part in word_to_num for part in current.split('-')):
            parts = current.split('-')
            if all(part in word_to_num for part in parts):
                # Handle cases like "twenty-four"
                if word_to_num[parts[0]].endswith('0') and int(word_to_num[parts[1]]) < 10:
                    num_value = int(word_to_num[parts[0]]) + int(word_to_num[parts[1]])
                    # Preserve any punctuation at the end
                    punctuation = ''.join(c for c in current_word if c in '.,:;!?')
                    result_words.append(str(num_value) + punctuation)
                else:
                    # Just append if we can't process it properly
                    result_words.append(current_word)
            else:
                result_words.append(current_word)
        
        # Check for consecutive number words (e.g., "twenty four")
        elif current in word_to_num:
            num_value = int(word_to_num[current])
            
            # Look ahead for compound numbers
            j = i + 1
            compound_found = False
            
            while j < len(words) and words[j].lower().rstrip(',.;:!?') in word_to_num:
                next_num = int(word_to_num[words[j].lower().rstrip(',.;:!?')])
                
                # Handle different number combinations
                if num_value > 0 and next_num in [100, 1000, 1000000, 1000000000]:
                    num_value *= next_num
                    compound_found = True
                elif num_value % 10 == 0 and next_num < 10:
                    num_value += next_num
                    compound_found = True
                else:
                    break
                
                j += 1
            
            if compound_found:
                # Preserve any trailing punctuation
                punctuation = ''.join(c for c in words[j-1] if c in '.,:;!?')
                result_words.append(str(num_value) + punctuation)
                i = j - 1
            else:
                # Preserve any punctuation at the end
                punctuation = ''.join(c for c in current_word if c in '.,:;!?')
                result_words.append(str(num_value) + punctuation)
        else:
            result_words.append(current_word)
        
        i += 1
    
    result = ' '.join(result_words)
    result = convert_ordinals(result)
    result = clean_time_expressions(result)
    
    return result

def convert_ordinals(sentence):
    # Dictionary mapping ordinal words to their numerical form
    ordinal_map = {
        'first': '1st',
        'second': '2nd',
        'third': '3rd',
        'fourth': '4th',
        'fifth': '5th',
        'sixth': '6th',
        'seventh': '7th',
        'eighth': '8th',
        'ninth': '9th',
        'tenth': '10th',
        'eleventh': '11th',
        'twelfth': '12th',
        'thirteenth': '13th',
        'fourteenth': '14th',
        'fifteenth': '15th',
        'sixteenth': '16th',
        'seventeenth': '17th',
        'eighteenth': '18th',
        'nineteenth': '19th',
        'twentieth': '20th',
        'thirtieth': '30th',
        'fortieth': '40th',
        'fiftieth': '50th',
        'sixtieth': '60th',
        'seventieth': '70th',
        'eightieth': '80th',
        'ninetieth': '90th',
        'hundredth': '100th',
        'thousandth': '1000th',
        'millionth': '1000000th',
        'billionth': '1000000000th'
    }
    
    # Also handle compound ordinals like "twenty-first", "twenty-second", etc.
    tens = {
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90
    }
    
    units = {
        'first': (1, 'st'),
        'second': (2, 'nd'),
        'third': (3, 'rd'),
        'fourth': (4, 'th'),
        'fifth': (5, 'th'),
        'sixth': (6, 'th'),
        'seventh': (7, 'th'),
        'eighth': (8, 'th'),
        'ninth': (9, 'th')
    }
    
    # Split the sentence into words
    words = sentence.split()
    result_words = []
    
    for i, current_word in enumerate(words):
        current = current_word.lower().rstrip(',.;:!?')
        punctuation = ''.join(c for c in current_word if c in '.,:;!?')
        
        # Check for hyphenated ordinals like "twenty-first"
        if '-' in current:
            parts = current.split('-')
            if len(parts) == 2 and parts[0] in tens and parts[1] in units:
                num_value = tens[parts[0]] + units[parts[1]][0]
                suffix = units[parts[1]][1]
                # Handle exceptions for 11th, 12th, 13th
                if num_value % 100 in [11, 12, 13]:
                    suffix = 'th'
                result_words.append(f"{num_value}{suffix}{punctuation}")
                continue
        
        # Check for simple ordinals like "seventh"
        if current in ordinal_map:
            # Preserve any capitalization
            if current_word[0].isupper():
                result = ordinal_map[current].capitalize() + punctuation
            else:
                result = ordinal_map[current] + punctuation
            result_words.append(result)
        else:
            result_words.append(current_word)
    
    return ' '.join(result_words)

def clean_time_expressions(sentence):
    import re
    
    # Pattern for time units (am/pm)
    time_pattern = r'(\d+)(\s+)([aApP][mM])'
    sentence = re.sub(time_pattern, r'\1\3', sentence)
    
    # Pattern for "o'clock"
    oclock_pattern = r'(\d+)(\s+)(o\'clock|o\'Clock|O\'clock|O\'Clock)'
    sentence = re.sub(oclock_pattern, r'\1\3', sentence)
    
    # Pattern for time with colon (3 : 30 PM -> 3:30PM)
    colon_time_pattern = r'(\d+)(\s*):(\s*)(\d+)(\s+)([aApP][mM])'
    sentence = re.sub(colon_time_pattern, r'\1:\4\6', sentence)
    
    # Pattern for just hours and minutes without am/pm (3 : 30 -> 3:30)
    hours_mins_pattern = r'(\d+)(\s*):(\s*)(\d+)'
    sentence = re.sub(hours_mins_pattern, r'\1:\4', sentence)
    
    return sentence

def remove_punctuation(text):
    # Step 1: Replace decimal points in numbers with spaces
    text = re.sub(r'(\d+)\.(\d+)', r'\1 \2', text)
    
    # Step 2: Create a custom punctuation set that doesn't include decimal points
    # (though at this point we've already handled the decimal points in numbers)
    punctuation = string.punctuation
    
    # Step 3: Create a translation table and remove all punctuation
    translator = str.maketrans('', '', punctuation)
    cleaned_text = text.translate(translator)
    
    # Step 4: Normalize spaces
    return ' '.join(cleaned_text.split())

def squish_numbers(sentence):
    characters = list(sentence)
    if len(characters) < 3:
        return sentence

    filtered_characters = []
    for i in range(len(characters)):
        if i == 0:
            filtered_characters.append(characters[i])
            continue

        if i == len(characters) - 1:
            filtered_characters.append(characters[i])
            continue

        previous_character = characters[i-1]
        current_character = characters[i]
        next_character = characters[i+1]

        should_skip = previous_character.isdigit() and current_character == " " and next_character.isdigit()
        if should_skip:
            continue
        else:
            filtered_characters.append(current_character)

    return ''.join(filtered_characters)

def format_money(sentence):
    # Dictionary mapping currency symbols to their word forms
    currency_words = {
        "$": "dollars",
        "€": "euros",
        "£": "pounds",
        "¥": "yen",
        "₹": "rupees",
        "₽": "rubles",
        "₩": "won",
        "₿": "bitcoin",
        "₺": "lira",
        "₴": "hryvnia",
        "₼": "manat",
        "₾": "lari",
        "฿": "baht",
        "₫": "dong",
        "₱": "pesos",
        "₦": "naira"
    }
    
    # Regular expression to match monetary values
    # Matches patterns like $123, $123.45, €1,234.56, etc.
    pattern = r'([€$£¥₹₽₩₿₺₴₼₾฿₫₱₦])([0-9,]+)(?:\.([0-9]+))?'
    
    def replace_money(match):
        symbol = match.group(1)
        whole_part = match.group(2)
        decimal_part = match.group(3)
        
        currency_word = currency_words.get(symbol, symbol)
        
        if decimal_part:
            return f"{whole_part} {currency_word} and {decimal_part} cents"
        else:
            return f"{whole_part} {currency_word}"
    
    # Also handle cases where the currency symbol follows the amount (e.g., 100€)
    pattern_trailing = r'([0-9,]+)(?:\.([0-9]+))?([€$£¥₹₽₩₿₺₴₼₾฿₫₱₦])'
    
    def replace_trailing_money(match):
        whole_part = match.group(1)
        decimal_part = match.group(2)
        symbol = match.group(3)
        
        currency_word = currency_words.get(symbol, symbol)
        
        if decimal_part:
            return f"{whole_part} {currency_word} and {decimal_part} cents"
        else:
            return f"{whole_part} {currency_word}"
    
    # Handle currency codes (USD, EUR, etc.)
    pattern_code = r'([0-9,]+)(?:\.([0-9]+))?\s*(USD|EUR|GBP|JPY|INR|RUB|KRW|BTC|TRY|UAH|AZN|GEL|THB|VND|PHP|NGN)'
    
    currency_code_words = {
        "USD": "dollars",
        "EUR": "euros",
        "GBP": "pounds",
        "JPY": "yen",
        "INR": "rupees",
        "RUB": "rubles",
        "KRW": "won",
        "BTC": "bitcoin",
        "TRY": "lira",
        "UAH": "hryvnia",
        "AZN": "manat",
        "GEL": "lari",
        "THB": "baht",
        "VND": "dong",
        "PHP": "pesos",
        "NGN": "naira"
    }
    
    def replace_code_money(match):
        whole_part = match.group(1)
        decimal_part = match.group(2)
        code = match.group(3)
        
        currency_word = currency_code_words.get(code, code.lower())
        
        if decimal_part:
            return f"{whole_part} {currency_word} and {decimal_part} cents"
        else:
            return f"{whole_part} {currency_word}"
    
    # Apply all patterns
    result = re.sub(pattern, replace_money, sentence)
    result = re.sub(pattern_trailing, replace_trailing_money, result)
    result = re.sub(pattern_code, replace_code_money, result)
    
    return result               

def normalize_text(text):
    normalized_text = squish_numbers(remove_punctuation(format_money(convert_ordinals(sentence_to_numbers(dehyphenate(text.lower()))))))
    return normalized_text

def calculate_word_error_rate(reference, hypothesis):
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    # Normalize the text.
    ref_words = normalized_reference.split()
    hyp_words = normalized_hypothesis.split()
    
    # Create dynamic programming matrix for edit distance calculation
    # Matrix size: (len(ref_words) + 1) x (len(hyp_words) + 1)
    d = [[0 for x in range(len(hyp_words) + 1)] for y in range(len(ref_words) + 1)]
    
    # Initialize first row and column with incremental values
    # This represents the cost of transforming empty string to target string
    for i in range(len(ref_words) + 1):
        d[i][0] = i  # Cost of deleting i words from reference
    for j in range(len(hyp_words) + 1):
        d[0][j] = j  # Cost of inserting j words into empty string
    
    # Fill the matrix using dynamic programming
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                # Words match - no cost
                d[i][j] = d[i-1][j-1]
            else:
                # Calculate costs for different operations
                substitution = d[i-1][j-1] + 1  # Replace word
                insertion = d[i][j-1] + 1      # Insert word
                deletion = d[i-1][j] + 1       # Delete word
                d[i][j] = min(substitution, insertion, deletion)
    
    # Calculate WER as ratio of errors to reference length
    error_count = d[len(ref_words)][len(hyp_words)]
    wer = float(error_count) / len(ref_words) if len(ref_words) > 0 else 0
    
    # Find the specific words that caused errors
    incorrect_words = find_incorrect_words(d, ref_words, hyp_words)
    
    return wer, incorrect_words, normalized_reference, normalized_hypothesis

def find_incorrect_words(d, ref_words, hyp_words):
    incorrect = []
    i, j = len(ref_words), len(hyp_words)
    
    # Backtrack from bottom-right to top-left of matrix
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            # Words match - move diagonally
            i -= 1
            j -= 1
        else:
            # Determine which operation was performed
            if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
                # Substitution: replace word
                incorrect.append({
                    'type': 'substitution',
                    'reference': ref_words[i-1],
                    'hypothesis': hyp_words[j-1]
                })
                i -= 1
                j -= 1
            elif j > 0 and d[i][j] == d[i][j-1] + 1:
                # Insertion: extra word in hypothesis
                incorrect.append({
                    'type': 'insertion',
                    'reference': None,
                    'hypothesis': hyp_words[j-1]
                })
                j -= 1
            elif i > 0 and d[i][j] == d[i-1][j] + 1:
                # Deletion: missing word in hypothesis
                incorrect.append({
                    'type': 'deletion',
                    'reference': ref_words[i-1],
                    'hypothesis': None
                })
                i -= 1
    
    # Reverse to get errors in original order
    incorrect.reverse()
    return incorrect

def compare_transcription(original_text, transcription):
    # Calculate WER and errors for the transcription
    wer, incorrect, normalized_reference, normalized_hypothesis = calculate_word_error_rate(original_text, transcription)
    
    # Format results
    results = {
        'wer': wer,
        'incorrect_words': incorrect,
        'original_text': original_text,
        'transcription': transcription,
        'normalized_original_text': normalized_reference,
        'normalized_transcription': normalized_hypothesis
    }
    
    return results