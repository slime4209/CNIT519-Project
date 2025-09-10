import nltk
from nltk.corpus import wordnet as wn

# You may need these once:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# THIS IS A TEST COMMENT!!!!
# ---------------------------
# Mock AoA dictionary (replace with real data if you have it)
# Values = approximate age of acquisition
aoa_dict = {
    "trunk_elephant": 5,
    "trunk_luggage": 8,
    "guts_organs": 7,
    "guts_courage": 9,
    "shingles_disease": 12,
    "shingles_roof": 10,
}

# ---------------------------
def get_candidate_words(text):
    """Extract candidate words that have multiple WordNet senses."""
    words = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    candidates = []
    for w in words:
        senses = wn.synsets(w)
        if len(senses) >= 2:
            candidates.append(w)
    return list(set(candidates))

# ---------------------------
def explain_wordplay(word, text, age):
    """Check if a word has multiple senses relevant to the text."""
    senses = wn.synsets(word)
    explanations = []
    
    for i, s in enumerate(senses[:4]):  # limit to first few senses
        explanations.append((s.name(), s.definition()))
    
    # Mock logic: pick two senses from AoA dict if available
    sense_keys = [k for k in aoa_dict if word in k]
    if len(sense_keys) < 2:
        return None  # no valid wordplay
    
    sense1, sense2 = sense_keys[:2]
    age1, age2 = aoa_dict[sense1], aoa_dict[sense2]
    
    result = {
        "word": word,
        "senses": {
            sense1: f"({age1}+ yrs) {sense1.split('_')[-1]}",
            sense2: f"({age2}+ yrs) {sense2.split('_')[-1]}",
        },
        "text": text,
        "age_ok": age >= max(age1, age2),
        "explanation": f"The word '{word}' has multiple meanings ({sense1.split('_')[-1]} vs {sense2.split('_')[-1]}). "
                       f"Both meanings are triggered in the text, creating humor."
    }
    return result

# ---------------------------
def analyze_text(text, age=10):
    """Main pipeline."""
    candidates = get_candidate_words(text)
    results = []
    
    if not candidates:
        return {"joke": False, "reason": "No words with multiple senses found."}
    
    for w in candidates:
        exp = explain_wordplay(w, text, age)
        if exp:
            results.append(exp)
    
    if results:
        return {"joke": True, "details": results}
    else:
        return {"joke": False, "reason": "No wordplay fits context."}

# ---------------------------
# Example usage
if __name__ == "__main__":
    examples = [
        "Why don't skeletons fight? Because they have no guts.",
        "Why do skeletons fight? Because they have no guts.",
        "I saw a bat in the cave.",
        "I've got a bad case of shingles."
    ]
    
    for ex in examples:
        print("\nINPUT:", ex)
        output = analyze_text(ex, age=10)
        print("OUTPUT:", output)
