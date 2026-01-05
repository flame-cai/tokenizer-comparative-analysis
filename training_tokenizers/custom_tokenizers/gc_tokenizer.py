import unicodedata
import pandas as pd


# def get_gc_tokens(devanagari_string):
#     # Pre-compile set lookups for better performance
#     # combining_marks = frozenset({
#     #     '_','्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै','ॆ', 'ो', 'ौ','ॊ', '़','ॅ','ॉ',
#     #     '_', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ৢ', 'ৣ','ে', 'ৈ', 'ো', 'ৌ', 'ঁ', 'ং', 'ঃ', '্', '়', 'ৗ', '্র', '‍',
#     #     '্', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़', 'ॅ', 'ॉ', 'ँ','्'
#     # })
    
#     combining_marks = frozenset({
#         '_','्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै','ॆ', 'ो', 'ौ','ॊ', '़','ॅ','ॉ',
#         '_', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़', 'ॅ', 'ॉ', 'ँ','्',
#     })

#     # digits = frozenset('०१२३४५६७८९০১২৩৪৫৬৭৮৯0123456789')
#     digits = frozenset('०१२३४५६७८९0123456789')
#     end_markers = frozenset(['্', '्'])
    
#     # Pre-allocate list with estimated capacity
#     graphemes = []
#     temp_grapheme = []
    
#     # Replace spaces once instead of checking each time
#     devanagari_string = devanagari_string.replace(" ", "_")
    
#     for char in devanagari_string:
#         if char not in combining_marks:
#             if temp_grapheme and not (temp_grapheme[-1] in end_markers):
#                 if char in digits:
#                     graphemes.append(''.join(temp_grapheme))
#                     temp_grapheme = []
#                     graphemes.append(char)
#                 else:
#                     graphemes.append(''.join(temp_grapheme))
#                     temp_grapheme = [char]
#             else:
#                 temp_grapheme.append(char)
#         else:
#             temp_grapheme.append(char)
    
#     if temp_grapheme:
#         graphemes.append(''.join(temp_grapheme))
    
#     # Process all cleanups in one pass
#     if '\u200c' in devanagari_string or '\u200d' in devanagari_string or '\u200b' in devanagari_string:
#         graphemes = [g.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '') for g in graphemes]
    
#     return graphemes

def get_gc_tokens(devanagari_string):
    # Pre-compile set lookups for better performance
    # combining_marks = frozenset({
    #     '_','्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै','ॆ', 'ो', 'ौ','ॊ', '़','ॅ','ॉ',
    #     '_', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ৄ', 'ৢ', 'ৣ','ে', 'ৈ', 'ো', 'ৌ', 'ঁ', 'ং', 'ঃ', '্', '়', 'ৗ', '্র', '‍',
    #     '্', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़', 'ॅ', 'ॉ', 'ँ','्'
    # })
    
    combining_marks = frozenset({
        '_','्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै','ॆ', 'ो', 'ौ','ॊ', '़','ॅ','ॉ',
        '_', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़', 'ॅ', 'ॉ', 'ँ','्',
        '̪', 'ː', 'ʱ', 'ᵊ','ʰ', '̃', '̩', '̯', '̤', '̥', '̬', '̰', '̱', '̲', '̳', '̴', '̵', '̶',
        '̷', '̸', '̹', '̺', '̻', '̼', '̽', '̾', '̿', '̀', '́', '͆', '͇', '͈', '͉',
        '͊', '͋', '͌', '͍', '͎', '͐', '͑', '͒', '͓', '͔', '͕', '͖', '͗', '͘', '͙',
        '͚', '͛', '͜', '͝', '͞', '͟', '͠', '͡', '͢', 'ͣ', 'ͤ', 'ͥ', 'ͦ', 'ͧ', 'ͨ',
        'ͩ', 'ͪ', 'ͫ', 'ͬ', 'ͭ', 'ͮ', 'ͯ', '̚', '̣', '̇', '̈', '̊', '̋', '̌', '̍', '̎',
        '̏', '̓', '̔', '̕', '̖', '̗', '̘', '̙', '̜', '̝', '̞', '̟', '̠', '̡', '̢',
        '̦', '̨', 
        # '.'
        # 'ĩː','ɑː',
    })

    # digits = frozenset('०१२३४५६७८९০১২৩৪৫৬৭৮৯0123456789')
    digits = frozenset('०१२३४५६७८९0123456789')
    end_markers = frozenset(['্', '्','͡',])
    
    # Pre-allocate list with estimated capacity
    graphemes = []
    temp_grapheme = []
    
    # Replace spaces once instead of checking each time
    devanagari_string = devanagari_string.replace(" ", "_")
    
    for char in devanagari_string:
        if char not in combining_marks:
            if temp_grapheme and not (temp_grapheme[-1] in end_markers):
                if char in digits:
                    graphemes.append(''.join(temp_grapheme))
                    temp_grapheme = []
                    graphemes.append(char)
                else:
                    graphemes.append(''.join(temp_grapheme))
                    temp_grapheme = [char]
            else:
                temp_grapheme.append(char)
        else:
            temp_grapheme.append(char)
    
    if temp_grapheme:
        graphemes.append(''.join(temp_grapheme))
    
    # Process all cleanups in one pass
    if '\u200c' in devanagari_string or '\u200d' in devanagari_string or '\u200b' in devanagari_string:
        graphemes = [g.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '') for g in graphemes]
    
    return graphemes

def get_gc_tokens_(devanagari_string):
    combining_marks = {'_', '्', 'ँ', 'ं', 'ः', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ', 'े', 'ै', 'ो', 'ौ', '़','ॅ','ॉ'}
    graphemes = []
    temp_grapheme = ""
    devanagari_string = devanagari_string.replace(" ","_")
    # print(devanagari_string)
    
    for char in devanagari_string:
        if unicodedata.combining(char) == 0 and char not in combining_marks:
            # If temp_grapheme has something and last char was not virama, append to graphemes
            if temp_grapheme and not temp_grapheme.endswith('्'):
                graphemes.append(temp_grapheme)
                temp_grapheme = char
            else:
                temp_grapheme += char
        else:
            temp_grapheme += char  # Add combining mark or virama to current grapheme

    # Append the last grapheme if exists
    if temp_grapheme:
        graphemes.append(temp_grapheme)

    graphemes = [i.replace('\u200c', '').replace('\u200d', '').replace('\u200b', '') for i in graphemes]

    return graphemes

# print(get_gc_tokens('meːɾɑː n̪ɑːm gʰəʈoːt̪kət͡ʃ'))
