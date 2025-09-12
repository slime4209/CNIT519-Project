import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Step 1: Load dataset ===
csv_path = "Jokes.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Encode label
y = df["Jokes"].apply(lambda t: 1 if str(t).strip().lower() in ["funny", "yes", "1", "true"] else 0)

# === Step 2: Create features for training ===
def create_features(row):
    text = str(row["Jokes"]).lower()
    homograph = str(row["Homograph"]).lower() if pd.notna(row["Homograph"]) else ""
    meaning1 = str(row["Meaning 1"]).lower() if pd.notna(row["Meaning 1"]) else ""
    meaning2 = str(row["Meaning 2"]).lower() if pd.notna(row["Meaning 2"]) else ""
    
    return pd.Series({
        "homograph_present": 1 if homograph in text else 0,
        "meaning1_in_text": 1 if meaning1 and meaning1 in text else 0,
        "meaning2_in_text": 1 if meaning2 and meaning2 in text else 0,
    })

X = df.apply(create_features, axis=1)

# === Step 3: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a small decision tree
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Step 4: Automatic prediction on user input ===
def classify_input(user_text, df_homographs=df):
    user_text_lower = user_text.lower()
    
    # For each homograph in dataset, check if it and its meanings are in the text
    features = []
    for idx, row in df_homographs.iterrows():
        homograph_present = 1 if str(row["Homograph"]).lower() in user_text_lower else 0
        meaning1_in_text = 1 if str(row["Meaning 1"]).lower() in user_text_lower else 0
        meaning2_in_text = 1 if str(row["Meaning 2"]).lower() in user_text_lower else 0
        features.append([homograph_present, meaning1_in_text, meaning2_in_text])
    
    # Aggregate features across all homographs: if any is present, use 1
    agg_features = [
        1 if any(f[i] == 1 for f in features) else 0
        for i in range(3)
    ]
    
    pred = clf.predict([agg_features])[0]
    label = "Funny" if pred == 1 else "Not Funny"
    return {
        "input": user_text,
        "classification": label
    }

# Interactive test
if __name__ == "__main__":
    while True:
        user_input = input("Enter a sentence (or 'quit'): ")
        if user_input.lower() == "quit":
            break
        result = classify_input(user_input)
        print(f"\nClassification: {result['classification']}\n")
