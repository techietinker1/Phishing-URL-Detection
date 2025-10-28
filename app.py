from flask import Flask, request, render_template, jsonify
import joblib
from pathlib import Path
import re
import os
import pandas as pd

app = Flask(__name__)

MODEL_PATH = Path('phishing_model.pkl')
model = None  # can be a sklearn model or dict with {'model':..., 'feature_columns': [...]}
PHISHING_CLASS = int(os.getenv('PHISHING_CLASS', '0'))  # default assumption: 0 = phishing, 1 = legitimate
PHISHING_PROB_THRESHOLD = float(os.getenv('PHISHING_PROB_THRESHOLD', '0.70'))  # stricter threshold now that model stronger
PHISHING_PROB_MARGIN = float(os.getenv('PHISHING_PROB_MARGIN', '0.10'))  # phishing prob must exceed next highest by this margin

def load_model():
    """Load raw model or payload dict with metadata."""
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}. Please train the model by running train_enhanced.py")
        loaded = joblib.load(MODEL_PATH)
        # Backwards compatibility: older pickle was just model
        if isinstance(loaded, dict) and 'model' in loaded:
            model = loaded
        else:
            model = {'model': loaded, 'feature_columns': ['url_length','count_dot','count_at','count_hyphen']}
    return model

def extract_features(url: str, feature_columns):
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
    
    feature_map = {
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
    
    # Return only the features that the model expects
    result = {}
    for col in feature_columns:
        result[col] = feature_map.get(col, 0)
    return result

# --- Heuristic / rule-based risk analysis (simple illustrative) ---
RISK_TLDS = {"zip","mov","country","click","link","work","fit","tokyo","rest"}
SUSPICIOUS_KEYWORDS = ["login","verify","update","secure","account","webscr","banking","confirm","limited","suspend","unlock","password"]
IP_REGEX = re.compile(r"^https?://(?:\d{1,3}\.){3}\d{1,3}")

def analyze_url(url: str):
    """Return (is_risky, reasons list) based on simple handcrafted heuristics.
    These are NOT exhaustive but help catch obvious fakes the tiny ML model may miss.
    """
    reasons = []
    lower = url.lower()
    # IP address instead of domain
    if IP_REGEX.search(lower):
        reasons.append("Uses raw IP address")
    # Excessive length
    if len(url) > 120:
        reasons.append("Very long URL (>120 chars)")
    # Many hyphens
    if url.count('-') >= 4:
        reasons.append("Many hyphens (>=4)")
    # Many special chars that can obfuscate
    special = sum(url.count(c) for c in ['@','?','=','%'])
    if special >= 5:
        reasons.append("High special character count")
    # Suspicious keywords
    kw_hits = [k for k in SUSPICIOUS_KEYWORDS if k in lower]
    if kw_hits:
        reasons.append(f"Suspicious keywords: {', '.join(kw_hits[:4])}")
    # Suspicious TLD
    # Extract TLD crudely
    m = re.search(r"https?://[^/]+", lower)
    if m:
        host = m.group(0).split('//',1)[1]
        if ':' in host:
            host = host.split(':',1)[0]
        host_parts = host.rsplit('.',2)
        if len(host_parts) >= 2:
            tld = host_parts[-1]
            if tld in RISK_TLDS:
                reasons.append(f"Potentially risky TLD: .{tld}")
    # At sign presence
    if '@' in url:
        reasons.append("Contains '@' (can hide real host)")
    # Mixed http login path
    if lower.startswith('http://') and any(k in lower for k in ("login","secure","account")):
        reasons.append("Unencrypted (http) with credential keyword")
    # Final decision
    return (len(reasons) > 0, reasons)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    ok = MODEL_PATH.exists()
    return {'status': 'ok' if ok else 'missing-model', 'model_exists': ok}

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '').strip()
    if not url:
        return render_template('index.html', prediction="Please enter a URL.")
    payload = load_model()
    mdl = payload['model']
    feature_columns = payload.get('feature_columns', ['url_length','count_dot','count_at','count_hyphen'])
    feats_dict = extract_features(url, feature_columns)
    df = pd.DataFrame([feats_dict])[feature_columns]
    pred = mdl.predict(df)[0]
    # Probability-based decision for model portion
    proba = None
    model_unsafe = False
    if hasattr(mdl, 'predict_proba'):
        proba = mdl.predict_proba(df)[0]
        # find index of phishing class
        classes = list(mdl.classes_)
        phish_index = classes.index(PHISHING_CLASS)
        phish_prob = proba[phish_index]
        # margin check
        sorted_probs = sorted(proba, reverse=True)
        second = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin_ok = (phish_prob - second) >= PHISHING_PROB_MARGIN
        model_unsafe = (pred == PHISHING_CLASS) and (phish_prob >= PHISHING_PROB_THRESHOLD) and margin_ok
    else:
        model_unsafe = (pred == PHISHING_CLASS)
    heuristic_unsafe, reasons = analyze_url(url)
    unsafe = heuristic_unsafe or model_unsafe
    output = "URL is Unsafe" if unsafe else "URL is Safe"
    # Always show reasons (even if empty) for transparency; if safe and no reasons, user just sees the message
    return render_template('index.html', prediction=output, url=url, reasons=reasons)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'url is required'}), 400
    try:
        payload = load_model()
        mdl = payload['model']
        feature_columns = payload.get('feature_columns', ['url_length','count_dot','count_at','count_hyphen'])
        feats_dict = extract_features(url, feature_columns)
        df = pd.DataFrame([feats_dict])[feature_columns]
        pred = int(mdl.predict(df)[0])
        model_unsafe = False
        model_phish_prob = None
        if hasattr(mdl, 'predict_proba'):
            proba = mdl.predict_proba(df)[0]
            classes = list(mdl.classes_)
            phish_index = classes.index(PHISHING_CLASS)
            model_phish_prob = float(proba[phish_index])
            sorted_probs = sorted(proba, reverse=True)
            second = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            margin_ok = (model_phish_prob - second) >= PHISHING_PROB_MARGIN
            model_unsafe = (pred == PHISHING_CLASS) and (model_phish_prob >= PHISHING_PROB_THRESHOLD) and margin_ok
        else:
            model_unsafe = (pred == PHISHING_CLASS)
        heuristic_unsafe, reasons = analyze_url(url)
        unsafe = heuristic_unsafe or model_unsafe
        label = 'unsafe' if unsafe else 'safe'
        message = 'URL is Unsafe' if unsafe else 'URL is Safe'
        return jsonify({
            'url': url,
            'prediction': pred,
            'phishing_class_config': PHISHING_CLASS,
            'phishing_prob_threshold': PHISHING_PROB_THRESHOLD,
            'phishing_prob_margin': PHISHING_PROB_MARGIN,
            'model_phish_prob': model_phish_prob,
            'model_raw': pred,
            'heuristic_risky': heuristic_unsafe,
            'reasons': reasons,
            'label': label,
            'message': message
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug', methods=['POST'])
def api_debug():
    """Return detailed feature breakdown and probabilities for a URL."""
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'url is required'}), 400
    payload = load_model()
    mdl = payload['model']
    feature_columns = payload.get('feature_columns', ['url_length','count_dot','count_at','count_hyphen'])
    feats_dict = extract_features(url, feature_columns)
    df = pd.DataFrame([feats_dict])[feature_columns]
    pred = int(mdl.predict(df)[0])
    probabilities = None
    phishing_probability = None
    if hasattr(mdl, 'predict_proba'):
        proba = mdl.predict_proba(df)[0]
        probabilities = {str(cls): float(p) for cls, p in zip(mdl.classes_, proba)}
        phishing_probability = probabilities.get(str(PHISHING_CLASS))
    heuristic_unsafe, reasons = analyze_url(url)
    model_unsafe = False
    if probabilities is not None and phishing_probability is not None:
        # margin logic
        proba_list = [probabilities[str(c)] for c in mdl.classes_]
        phish_index = list(mdl.classes_).index(PHISHING_CLASS)
        phish_prob = proba_list[phish_index]
        sorted_probs = sorted(proba_list, reverse=True)
        second = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin_ok = (phish_prob - second) >= PHISHING_PROB_MARGIN
        model_unsafe = (pred == PHISHING_CLASS) and (phish_prob >= PHISHING_PROB_THRESHOLD) and margin_ok
    else:
        model_unsafe = (pred == PHISHING_CLASS)
    unsafe = heuristic_unsafe or model_unsafe
    return jsonify({
        'url': url,
        'features': feats_dict,
        'classes': list(map(int, mdl.classes_)),
        'probabilities': probabilities,
        'phishing_probability': phishing_probability,
        'phishing_class_config': PHISHING_CLASS,
        'phishing_prob_threshold': PHISHING_PROB_THRESHOLD,
        'phishing_prob_margin': PHISHING_PROB_MARGIN,
        'model_pred': pred,
        'model_unsafe_component': model_unsafe,
        'heuristic_unsafe_component': heuristic_unsafe,
        'unsafe_final': unsafe,
        'heuristic_reasons': reasons
    })

if __name__ == '__main__':
    # For development only; in production use waitress (see wsgi.py)
    app.run(host='0.0.0.0', port=5000, debug=True)
