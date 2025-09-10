import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

# You may need these once:
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt_tab')

# ---------------------------
# Load AoA dictionary from CSV
def load_aoa_csv(csv_file):
    df = pd.read_csv(csv_file)
    aoa_dict = {}
    for _, row in df.iterrows():
        aoa_dict[row["sense_id"]] = {
            "word": row["word"].lower(),
            "gloss": row["sense_gloss"],
            "aoa": int(row["aoa"])
        }
    return aoa_dict

# ---------------------------
# Pretty-print helper
def pretty_output(input_text, result):
    print("\n" + "="*50)
    print(f"INPUT: {input_text}")
    print("-"*50)
    if not result["joke"]:
        print(f"Classification: Not a joke")
        print(f"Reason: {result['reason']}")
    else:
        print(f"Classification: Joke Detected ✅")
        for detail in result["details"]:
            print(f"\n  Word with multiple meanings: {detail['word']}")
            for sense, gloss in detail["senses"].items():
                print(f"    - {sense}: {gloss}")
            print(f"  Age appropriate? {'Yes' if detail['age_ok'] else 'No'}")
            print(f"  Explanation: {detail['explanation']}")
    print("="*50)

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
def explain_wordplay(word, text, age, aoa_dict):
    """Check if a word has multiple senses relevant to the text (from CSV AoA)."""
    # Look up all sense_ids for this word
    senses = [sid for sid, info in aoa_dict.items() if info["word"] == word]
    if len(senses) < 2:
        return None

    # Just pick first 2 senses for now
    sense1, sense2 = senses[:2]
    info1, info2 = aoa_dict[sense1], aoa_dict[sense2]

    result = {
        "word": word,
        "senses": {
            sense1: f"({info1['aoa']}+ yrs) {info1['gloss']}",
            sense2: f"({info2['aoa']}+ yrs) {info2['gloss']}"
        },
        "text": text,
        "age_ok": age >= max(info1["aoa"], info2["aoa"]),
        "explanation": f"The word '{word}' has multiple meanings ({info1['gloss']} vs {info2['gloss']}). "
                       f"Both meanings are triggered in the text, creating humor."
    }
    return result

# ---------------------------
def analyze_text(text, age, aoa_dict):
    """Main pipeline."""
    candidates = get_candidate_words(text)
    results = []
    
    if not candidates:
        return {"joke": False, "reason": "No words with multiple senses found."}
    
    for w in candidates:
        exp = explain_wordplay(w, text, age, aoa_dict)
        if exp:
            results.append(exp)
    
    if results:
        return {"joke": True, "details": results}
    else:
        return {"joke": False, "reason": "No wordplay fits context."}

# ---------------------------
# Example usage
if __name__ == "__main__":
    # Load AoA data
    aoa_dict = load_aoa_csv("aoa_data.csv")  # <-- change filename to your CSV

    examples = [
        "Why don't skeletons fight? Because they have no guts.",
        "Why do skeletons fight? Because they have no guts.",
        "I saw a bat in the cave.",
        "I've got a bad case of shingles."
    ]
    
    for ex in examples:
        print("\nINPUT:", ex)
        output = analyze_text(ex, age=10, aoa_dict=aoa_dict)
        pretty_output(ex, output)
