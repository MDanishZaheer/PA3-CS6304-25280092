# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""VQAv2 answer normalization and local soft-consensus scoring.

Punctuation, article, digit, and contraction rules follow the official
GT-Vision-Lab/VQA vqaEval.py evaluator (see Task_2/README.md).
"""

import re
from collections import Counter


CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id've": "i'd've", "idve": "i'd've", "im": "i'm", "ive": "i've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebodyd": "somebody'd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "theydve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "wedve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'alldve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "youdve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
           "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
           "nine": "9", "ten": "10"}
PUNCTUATION = (';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
               '_', '-', '>', '<', '@', '`', ',', '?', '!')
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_answer(answer):
    """Apply VQAv2 punctuation, digit, article, and contraction rules."""
    original = str(answer).replace("\n", " ").replace("\t", " ").strip()
    text = original
    for punctuation in PUNCTUATION:
        if punctuation + " " in original or " " + punctuation in original or COMMA_STRIP.search(original):
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = PERIOD_STRIP.sub("", text)
    tokens = [NUMBERS.get(word, word) for word in text.lower().split()]
    return " ".join(CONTRACTIONS.get(word, word) for word in tokens if word not in ("a", "an", "the"))


def consensus(answers):
    """Return normalized consensus, an original target, and its vote count."""
    raw = [item["answer"] if isinstance(item, dict) else str(item) for item in answers]
    if len(raw) != 10:
        raise ValueError("VQAv2 examples require ten human answers.")
    normalized = [normalize_answer(answer) for answer in raw]
    counts = Counter(normalized)
    winner = sorted(counts, key=lambda value: (-counts[value], value))[0]
    target = next(answer for answer, value in zip(raw, normalized) if value == winner)
    return winner, target, counts[winner]


def vqa_soft_score(prediction, references):
    """Compute the assigned ten-reference leave-one-out VQA score."""
    predicted = normalize_answer(prediction)
    normalized = [normalize_answer(answer) for answer in references]
    if len(normalized) != 10:
        raise ValueError("VQAv2 scoring requires ten references.")
    return sum(min(1.0, sum(predicted == answer for j, answer in enumerate(normalized) if j != i) / 3.0)
               for i in range(10)) / 10.0
