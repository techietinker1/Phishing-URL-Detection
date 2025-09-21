import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV_PATH = 'PhiUSIIL_Phishing_URL_Dataset.csv'
MODEL_OUT = 'phishing_model.pkl'

def load_dataset(path: str):
    # Some CSVs may contain BOM in first column name; fix it.
    df = pd.read_csv(path)
    if '\ufeffURL' in df.columns:
        df = df.rename(columns={'\ufeffURL': 'URL'})
    return df

def build_feature_frame(df: pd.DataFrame):
    # Keep a conservative set of numeric / binary features that are already in dataset
    # Exclude text fields like Title to avoid extra preprocessing for now.
    candidate_cols = [
        'URLLength','DomainLength','IsDomainIP','URLSimilarityIndex','CharContinuationRate',
        'TLDLegitimateProb','URLCharProb','TLDLength','NoOfSubDomain','HasObfuscation',
        'NoOfObfuscatedChar','ObfuscationRatio','NoOfLettersInURL','LetterRatioInURL',
        'NoOfDegitsInURL','DegitRatioInURL','NoOfEqualsInURL','NoOfQMarkInURL',
        'NoOfAmpersandInURL','NoOfOtherSpecialCharsInURL','SpacialCharRatioInURL','IsHTTPS',
        'LineOfCode','LargestLineLength','HasTitle','DomainTitleMatchScore','URLTitleMatchScore',
        'HasFavicon','Robots','IsResponsive','NoOfURLRedirect','NoOfSelfRedirect','HasDescription',
        'NoOfPopup','NoOfiFrame','HasExternalFormSubmit','HasSocialNet','HasSubmitButton',
        'HasHiddenFields','HasPasswordField','Bank','Pay','Crypto','HasCopyrightInfo',
        'NoOfImage','NoOfCSS','NoOfJS','NoOfSelfRef','NoOfEmptyRef','NoOfExternalRef'
    ]
    # Some columns might be missing depending on dataset version; keep only existing
    existing = [c for c in candidate_cols if c in df.columns]
    X = df[existing].copy()
    y = df['label']
    return X, y, existing

def train():
    df = load_dataset(CSV_PATH)
    X, y, used_cols = build_feature_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    # Persist model plus column order metadata
    payload = {
        'model': model,
        'feature_columns': used_cols
    }
    joblib.dump(payload, MODEL_OUT)
    print(f'Saved enhanced model with {len(used_cols)} features to {MODEL_OUT}')

if __name__ == '__main__':
    if not Path(CSV_PATH).exists():
        raise SystemExit(f'Dataset not found: {CSV_PATH}')
    train()
