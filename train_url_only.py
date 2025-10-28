import pandas as pd
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV_PATH = 'PhiUSIIL_Phishing_URL_Dataset.csv'
MODEL_OUT = 'phishing_model.pkl'

def extract_url_features(url: str):
    """Extract features that can be computed directly from URL string only."""
    url_lower = url.lower()
    
    # Basic length and character counts
    url_length = len(url)
    count_dot = url.count('.')
    count_at = url.count('@')
    count_hyphen = url.count('-')
    count_underscore = url.count('_')
    count_slash = url.count('/')
    count_question = url.count('?')
    count_equal = url.count('=')
    count_ampersand = url.count('&')
    
    # Digit analysis
    digit_count = sum(c.isdigit() for c in url)
    digit_ratio = digit_count / len(url) if len(url) > 0 else 0
    
    # Letter analysis
    letter_count = sum(c.isalpha() for c in url)
    letter_ratio = letter_count / len(url) if len(url) > 0 else 0
    
    # Special character analysis
    special_chars = '@?=&%#'
    special_count = sum(url.count(c) for c in special_chars)
    special_ratio = special_count / len(url) if len(url) > 0 else 0
    
    # Protocol analysis
    is_https = 1 if url_lower.startswith('https://') else 0
    is_http = 1 if url_lower.startswith('http://') else 0
    
    # Domain analysis
    domain_match = re.search(r'https?://([^/]+)', url_lower)
    if domain_match:
        domain = domain_match.group(1)
        domain_length = len(domain)
        subdomain_count = domain.count('.') - 1 if domain.count('.') > 0 else 0
        # Check if domain is IP address
        is_ip = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0
    else:
        domain_length = 0
        subdomain_count = 0
        is_ip = 0
    
    # Suspicious keywords (common in phishing)
    suspicious_keywords = [
        'login', 'signin', 'account', 'verify', 'confirm', 'update', 'secure',
        'bank', 'paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook',
        'security', 'suspended', 'limited', 'expire', 'urgent'
    ]
    keyword_count = sum(1 for keyword in suspicious_keywords if keyword in url_lower)
    
    # Suspicious TLDs
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.ru', '.cn']
    has_suspicious_tld = 1 if any(tld in url_lower for tld in suspicious_tlds) else 0
    
    # URL path analysis
    path_depth = url.count('/') - 2 if url.startswith('http') else url.count('/')
    path_depth = max(0, path_depth)
    
    return {
        'url_length': url_length,
        'count_dot': count_dot,
        'count_at': count_at,
        'count_hyphen': count_hyphen,
        'count_underscore': count_underscore,
        'count_slash': count_slash,
        'count_question': count_question,
        'count_equal': count_equal,
        'count_ampersand': count_ampersand,
        'digit_count': digit_count,
        'digit_ratio': digit_ratio,
        'letter_count': letter_count,
        'letter_ratio': letter_ratio,
        'special_count': special_count,
        'special_ratio': special_ratio,
        'is_https': is_https,
        'is_http': is_http,
        'domain_length': domain_length,
        'subdomain_count': subdomain_count,
        'is_ip': is_ip,
        'keyword_count': keyword_count,
        'has_suspicious_tld': has_suspicious_tld,
        'path_depth': path_depth
    }

def train():
    """Train model using only URL-extractable features."""
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    if '\ufeffURL' in df.columns:
        df = df.rename(columns={'\ufeffURL': 'URL'})
    
    print(f"Dataset loaded: {len(df)} URLs")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Extract features for all URLs
    print("Extracting features from URLs...")
    features_list = []
    for i, url in enumerate(df['URL']):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(df)} URLs")
        try:
            features = extract_url_features(str(url))
            features_list.append(features)
        except Exception as e:
            print(f"Error processing URL {i}: {url} - {e}")
            # Use default features for failed URLs
            features_list.append({k: 0 for k in extract_url_features('http://example.com').keys()})
    
    # Create feature dataframe
    X = pd.DataFrame(features_list)
    y = df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    print("Feature names:", list(X.columns))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=20, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    
    # Show feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save model with metadata
    payload = {
        'model': model,
        'feature_columns': list(X.columns),
        'label_mapping': {0: 'phishing', 1: 'legitimate'}
    }
    joblib.dump(payload, MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == '__main__':
    train()