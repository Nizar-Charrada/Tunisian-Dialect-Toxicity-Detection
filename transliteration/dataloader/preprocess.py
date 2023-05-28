import re
import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(PARENT_DIR)

import pyarabic.araby as araby
import pyarabic.trans as trans
from unidecode import unidecode
import itertools
from transliteration.dataloader.mapping import mapping
import pandas as pd


def remove_emoji(text):
    # remove emoji from text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    return text


def clean_tweet(text):
    text = re.sub("#\d+K\d+", " ", text)  # years like 2K19
    text = re.sub("http\S+\s*", " ", text)  # remove URLs
    text = re.sub("RT|cc", " ", text)  # remove RT and cc
    text = re.sub("@[^\s]+", " ", text)
    text = clean_hashtag(text)

    return text


def split_hashtag_to_words(tag):
    # split tag to words exemple : #i_like_you -> ['i', 'like', 'you']
    tag = tag.replace("#", "")
    tags = tag.split("_")
    return tags


def clean_hashtag(text):
    words = text.split()
    text = list()
    for word in words:
        if is_hashtag(word):
            text.extend(extract_hashtag(word))
        else:
            text.append(word)
    return " ".join(text)


def is_hashtag(word):
    if word.startswith("#"):
        return True
    else:
        return False


def extract_hashtag(text):
    hash_list = [re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")]
    word_list = []
    for word in hash_list:
        word_list.extend(split_hashtag_to_words(word))
    return word_list


def remove_num(text):
    text_list = re.findall(r"[\w']+|[?!.,]", text)

    for i in range(len(text_list)):
        if text_list[i].isnumeric():
            text_list[i] = ""

    return " ".join(text_list).strip()


def preprocess(text):
    text = split_arabic_latin(text)
    ## Remove punctuations
    text = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), " ", text
    )  # remove punctuation
    ## remove extra whitespace
    text = re.sub("\s+", " ", text)
    ## Remove Emojis
    text = remove_emoji(text)
    ## Convert text to lowercases
    text = text.lower()

    text = clean_tweet(text)
    text = remove_num(text)

    text = text.replace("x000d", "")
    return text


def preprocess_arabizi(word):
    word = word.replace("\n", "")
    word = word.replace("′", "'")
    word = word.replace("ß", "b")
    word = word.replace("$", "s")
    word = word.replace("1", "")
    word = word.replace("0", "")
    word = word.replace("6", "")
    word = word.lower()
    # word = my_unidecode(word)  # Remove accents
    # Remove '@name'
    word = re.sub(r"(@.*?)[\s]", " ", word)
    # Replace '&amp;' with '&'
    word = re.sub(r"&amp;", "&", word)
    # Remove trailing whitespace
    word = re.sub(
        r"\s+", " ", word
    ).strip()  # Remove trailing whitespace exemple: 'hello    world' -> 'hello world'
    word = re.sub(r"([ha][ha])\1+", r"\1", word)  # exemple: 'hahaha' -> 'haha'
    word = re.sub(r"([h][h][h][h])\1+", r"\1", word)  # exemple: 'hahahaha' -> 'hahaha'
    word = re.sub(
        r"([b-g-i-z])\1+", r"\1", word
    )  # Remove repeating characters exemple: 'hellooooo' -> 'hello'
    # text = re.sub(r'(.)\1+', r'\1', text)
    word = re.sub(r" [0-9]+ ", " ", word)  # exemple: 'hello 123 world' -> 'hello world'
    word = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), " ", word
    )  # remove punctuation exemple 'hello! world' -> 'hello world'
    return word


def preprocess_arabic(word):
    word = araby.strip_harakat(
        word
    )  # Remove diacritics exemple: 'مَرْحَباً' -> 'مرحبا'
    word = araby.strip_tashkeel(word)
    word = araby.strip_lastharaka(word)
    word = araby.strip_diacritics(word)
    word = araby.strip_tatweel(word)
    word = araby.normalize_hamza(
        word, method="tasheel"
    )  # Normalize hamza exemple: 'ء' -> 'ا'
    word = trans.normalize_digits(
        word, source="all", out="west"
    )  # Normalize digits exemple: '١٢٣' -> '123'
    word = "".join(
        mapping.get(c, c) for c in word
    )  # Replace arabic non general characters exemple: 'ء' -> 'ا'
    # word = "".join(
    #    char for char, _ in itertools.groupby(word)
    # )  # Remove repeating characters

    return word


def preprocess_data(data):
    data["arabic"] = data["arabic"].apply(preprocess_arabic)
    data["arabizi"] = data["arabizi"].apply(preprocess_arabizi)

    return data


def get_data():
    """Get data from csv and excel files and preprocess it.
    Yes, it is a bad idea to put the data in the code but im lazy and it works.
    We will change it later.
    """

    data = pd.read_csv("transliteration\data\dataset.csv")
    data = data[["arabizi", "arabic"]]
    add_data = pd.read_excel(
        r"transliteration/data/TACA-TA transliteration corpus.xlsx"
    )
    add_data = add_data.drop(["id", "GraphicalWord"], axis=1)
    add_data = add_data.rename(
        columns={"TACAword": "arabizi", "manual_transliteration": "arabic"}
    )
    data = pd.concat([data, add_data], axis=0)
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data = preprocess_data(data)

    return data


def my_unidecode(text):
    # Split the text into words using whitespace as the delimiter
    words = re.split(r"\s+", text)

    # Process each word
    result_words = []
    for word in words:
        # Check if the word contains Arabic characters
        arabic_chars = re.compile(r"[\u0600-\u06FF]+")
        if arabic_chars.search(word):
            # If the word contains Arabic characters, replace them with placeholders
            placeholders = {}
            for match in arabic_chars.finditer(word):
                placeholder = f"_ARABIC_{len(placeholders)}_"
                placeholders[placeholder] = match.group()
                word = word[: match.start()] + placeholder + word[match.end() :]

            # Transliterate the word using unidecode
            word = unidecode(word)

            # Replace the placeholders with the original Arabic characters
            for placeholder, char in placeholders.items():
                word = word.replace(placeholder, char)
        else:
            # If the word doesn't contain Arabic characters, just transliterate it using unidecode
            word = unidecode(word)

        result_words.append(word)

    # Join the words back together with spaces
    return " ".join(result_words)


def split_arabic_latin(text):
    # Split the text into groups of Arabic and non-Arabic characters
    arabic_chars = re.compile(r"[\u0600-\u06FF]+")
    latin_chars = re.compile(r"[^\u0600-\u06FF]+")
    groups = []
    while text:
        match = arabic_chars.match(text) or latin_chars.match(text)
        if match:
            groups.append(match.group())
            text = text[len(match.group()) :]
        else:
            raise ValueError(f"Invalid character: {text[0]}")

    # Join the groups and return the result
    return " ".join(groups)
