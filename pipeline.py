# pipeline.py
# -----------------------------------------------------------------------------
# Sunday ETL Pipeline: Scraping, Cleansing, Gemini Sentiment Analysis, and Cloud DB Upload
# BPSDM Jatim Data Science Project
# -----------------------------------------------------------------------------

import os
import re
import json
import time
import urllib.parse
import requests
import pandas as pd
import scrapetube

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
scrapetube.scrapetube.type_property_map["streams"] = "lockupViewModel"

CHANNEL_ID = "UCS9pHfAZEeelx1tYtDh_ZAg"
LIMIT_VIDEOS = 3
CSV_OUTPUT_PATH = "bpsdm_gemini_sentiment_results.csv"
SPREADSHEET_NAME = "BPSDM Jatim Live Chat Sentiments" # Target Google Sheet name

# -----------------------------------------------------------------------------
# DATA CLEANSING
# -----------------------------------------------------------------------------
def is_spam_attendance(text):
    text_clean = text.strip().lower()
    text_normalized = text_clean.replace('_', ' ').replace('-', ' ')
    
    if not text_normalized:
        return True
        
    greetings = [
        r'as+a+l+a+m+u+a+l+a+i+k+u+m',
        r'as+a+l+a+m+u\s*a+l+a+i+k+u+m',
        r'as+a+l+a+m\s*a+l+a+i+k+u+m',
        r'as+a+l+a+m+u\'*a+l+a+i+k+u+m',
        r'\baskum\b',
        r'w[a\']*l+a+i+k+u+m',
        r'w+a+[\'\s]*a+l+a+i+k+u+m',
        r'w+a+l+a+i+k+u+m\s*s+a+l+a+m',
        r'w+a+l+a+i+k+u+m+s+a+l+a+m',
        r'w+a+l+a+i+k+u+m+s+l+a+m',
        r'w+a+l+a+i+k+u+m+s+l+m',
        r'wa*laik*us*alam',
        r'shalom', r'om\s+swastiastu', r'namo\s+buddhaya', r'salam\s+kebajikan',
        r'^salam$', r'^salam\s+pancasila$', r'^salam\s+kenal$', r'\bsalam\s+kenal\b', r'\bwr\s*wb\b'
    ]
    for pattern in greetings:
        if re.search(pattern, text_normalized):
            return True
            
    nyimak_patterns = [
        r'\bnyimak[k]*\b', r'\bmenyimak\b', r'\bsimak\b', r'\bmenonton\b', r'\bnonton\b'
    ]
    for pattern in nyimak_patterns:
        if re.search(pattern, text_normalized):
            return True
    
    if re.search(r'\b(hadi+r+|hadr|hadl[ir]+|hadiroh|hadiro|hadirat|hafir)\b', text_normalized):
        short_spam_patterns = [
            r'^hadir$',
            r'^hadir+[ \.]*(pak|bu|bos|min|admin|dan menyimak|menyimak|dan mengikuti|mengikuti|selalu|ikut)?[\.]*$',
            r'^(siap|izin|ijin|saya|kami)\s+hadir',
            r'^hadir\s+(di|dari|prov|kab|kec|dinas|instansi|sd|smp|sma|smk|rs|rssa|puskesmas|upt|sekolah|badan|biro|setda|bpkad|dlh|rsud)',
            r'(dinas|instansi|badan|biro|setda|bpkad|dlh|rs|smk|smp|sd|rsud|kab|prov|kec|upt|puskesmas|sekolah|rs|rssa)\b.*\bhadir',
            r'\bhadir\b.*\b(dinas|instansi|badan|biro|setda|bpkad|dlh|rs|smk|smp|sd|rsud|kab|prov|kec|upt|puskesmas|sekolah|rs|rssa)\b'
        ]
        for pattern in short_spam_patterns:
            if re.search(pattern, text_normalized):
                return True
        if len(text_normalized) < 50:
            return True

    spam_keywords = [
        r'\bnip\b', r'\babsensi\b', r'\babsen\b', r'\bpresensi\b', r'\bdaftar\s+hadir\b', 
        r'\blink\s+(presensi|absen|daftar\s+hadir|sertifikat)\b',
        r'\bnama\s*:', r'\bnip\s*:', r'\binstansi\s*:', r'\bunit\s*kerja\s*:', r'\bjabatan\s*:'
    ]
    for pattern in spam_keywords:
        if re.search(pattern, text_normalized):
            return True
            
    if re.search(r'\b\d{9,18}\b', text_normalized):
        return True
        
    return False

# -----------------------------------------------------------------------------
# GEMINI DYNAMIC MODEL DISCOVERY & CLASSIFIER
# -----------------------------------------------------------------------------
def get_available_flash_model(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models_list = models_data.get("models", [])
            flash_models = []
            for m in models_list:
                name = m.get("name", "")
                methods = m.get("supportedGenerationMethods", [])
                if "flash" in name.lower() and "generateContent" in methods:
                    model_id = name.split("/")[-1]
                    flash_models.append(model_id)
            if flash_models:
                flash_models.sort(reverse=True)
                return flash_models[0]
    except Exception as e:
        print(f"[Gemini Warning] Model discovery failed: {e}")
    return "gemini-1.5-flash"

def analyze_sentiment_gemini_batch(messages, api_key, model_name):
    if not messages:
        return []
        
    batch_data = [{"id": idx, "message": msg} for idx, msg in enumerate(messages)]
    
    prompt = f"""
    You are an expert Indonesian sentiment analysis model.
    Classify the sentiment of the following list of YouTube live chat comments into one of these categories: "Positive", "Negative", or "Neutral".
    
    Guidelines:
    - "Positive": Praise for the speaker/host, webinar contents, or clear stream quality.
    - "Negative": Technical issues (sound cuts, lag, pixelated video), complaints about registration links, or general dissatisfaction.
    - "Neutral": General questions, topical discussions, or general statements unrelated to stream quality or speaker appreciation.
    
    Input data:
    {json.dumps(batch_data, ensure_ascii=False)}
    
    Response Format:
    Return your response strictly as a JSON array of objects matching the input IDs. Example:
    [
      {{"id": 0, "sentiment": "Positive"}},
      {{"id": 1, "sentiment": "Negative"}}
    ]
    Do not add any markdown, code blocks (e.g. ```json), or preamble outside the JSON array.
    """
    
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            return ["Neutral"] * len(messages)
            
        res_json = response.json()
        candidate_text = res_json['candidates'][0]['content']['parts'][0]['text']
        results = json.loads(candidate_text.strip())
        
        sentiment_map = {item['id']: item['sentiment'] for item in results}
        
        labels = []
        for idx in range(len(messages)):
            label = sentiment_map.get(idx, "Neutral").strip().title()
            if label not in ["Positive", "Negative", "Neutral"]:
                label = "Neutral"
            labels.append(label)
        return labels
    except Exception:
        return ["Neutral"] * len(messages)

# -----------------------------------------------------------------------------
# YOUTUBE SCRAPING CORE
# -----------------------------------------------------------------------------
def fetch_video_metadata(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    
    response = session.get(url, timeout=15)
    if response.status_code != 200:
        raise Exception(f"HTTP Error {response.status_code}")
        
    html = response.text
    
    def extract_json(regex_pattern):
        match = re.search(regex_pattern, html)
        return json.loads(match.group(1)) if match else None

    ytcfg_json = extract_json(r'ytcfg\.set\(({.+?})\);')
    yt_initial_data = extract_json(r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;')
    
    if not ytcfg_json or not yt_initial_data:
        raise Exception("Failed to parse page keys")
        
    api_key = ytcfg_json.get("INNERTUBE_API_KEY")
    context = ytcfg_json.get("INNERTUBE_CONTEXT")
    
    try:
        conversation_bar = yt_initial_data['contents']['twoColumnWatchNextResults']['conversationBar']
        live_chat_renderer = conversation_bar['liveChatRenderer']
        continuation_token = live_chat_renderer['continuations'][0]['reloadContinuationData']['continuation']
    except KeyError:
        continuation_token = None
        
    return session, api_key, context, continuation_token

def download_chat_replay(video_id):
    try:
        session, api_key, context, token = fetch_video_metadata(video_id)
    except Exception:
        return []
        
    if not token:
        return []
        
    session.headers.update({
        "Content-Type": "application/json",
        "Referer": f"https://www.youtube.com/live_chat_replay?continuation={token}"
    })
    
    messages = []
    while token:
        api_url = f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat_replay?key={api_key}"
        payload = {"context": context, "continuation": urllib.parse.unquote(token)}
        
        response = session.post(api_url, json=payload, timeout=15)
        if response.status_code != 200:
            break
            
        res_json = response.json()
        continuation_contents = res_json.get("continuationContents", {})
        live_chat_continuation = continuation_contents.get("liveChatContinuation", {})
        actions = live_chat_continuation.get("actions", [])
        
        if not actions:
            break
            
        for action in actions:
            replay_action = action.get("replayChatItemAction", {})
            inner_actions = replay_action.get("actions", [])
            for inner_action in inner_actions:
                item = inner_action.get("addChatItemAction", {}).get("item", {})
                text_renderer = item.get("liveChatTextMessageRenderer", {})
                if text_renderer:
                    author = text_renderer.get("authorName", {}).get("simpleText", "Unknown")
                    message_runs = text_renderer.get("message", {}).get("runs", [])
                    message_content = "".join([run.get("text", "") for run in message_runs])
                    time_offset = replay_action.get("videoOffsetTimeMsec")
                    try:
                        time_seconds = int(round(float(time_offset) / 1000)) if time_offset else 0
                    except ValueError:
                        time_seconds = 0
                        
                    messages.append({
                        "Author": author,
                        "Message": message_content,
                        "Time Offset (Seconds)": time_seconds
                    })
                    
        continuations = live_chat_continuation.get("continuations", [])
        if continuations:
            con_data = continuations[0].get("liveChatReplayContinuationData") or continuations[0].get("reloadContinuationData")
            token = con_data.get("continuation") if con_data else None
        else:
            token = None
            
        time.sleep(0.5)
        
    return messages

# -----------------------------------------------------------------------------
# GOOGLE SHEETS CLOUD INTEGRATION
# -----------------------------------------------------------------------------
def update_google_sheet(df, spreadsheet_name):
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        print("[Sheet Warning] GOOGLE_SHEETS_CREDENTIALS not set. Skipping sheet upload.")
        return False
        
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # Parse Credentials JSON
        creds_dict = json.loads(creds_json)
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open / Create spreadsheet
        try:
            sh = client.open(spreadsheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            print(f"[Sheet Info] Sheet '{spreadsheet_name}' not found. Creating it...")
            sh = client.create(spreadsheet_name)
            print(f"[Sheet Warning] Created new sheet. You must share it with your personal email address from the Google Drive UI to see it.")
            
        # Get first worksheet
        worksheet = sh.get_worksheet(0)
        worksheet.clear()
        
        # Format the df headers and values
        data = [df.columns.values.tolist()] + df.values.tolist()
        worksheet.update('A1', data)
        print(f"[Sheet Info] Successfully wrote {len(df)} rows to Google Sheet: {spreadsheet_name}")
        return True
    except Exception as e:
        print(f"[Sheet Error] Failed to update Google Sheet: {e}")
        return False

# -----------------------------------------------------------------------------
# PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("Starting Sunday ETL Pipeline...")
    
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY environment variable is required.")
        return
        
    print("1. Discovering Gemini model...")
    model_name = get_available_flash_model(gemini_key)
    print(f"Using model: {model_name}")
    
    print("2. Fetching recent videos...")
    try:
        videos = scrapetube.get_channel(channel_id=CHANNEL_ID, content_type="streams")
        recent_videos = []
        for v in videos:
            recent_videos.append(v)
            if len(recent_videos) >= LIMIT_VIDEOS:
                break
    except Exception as e:
        print(f"Error fetching channel stream: {e}")
        return
        
    video_details = []
    for v in recent_videos:
        video_id = v.get("contentId")
        title = v.get("metadata", {}).get("lockupMetadataViewModel", {}).get("title", {}).get("content", "Unknown Title")
        video_details.append({"id": video_id, "title": title})
        
    all_chats = []
    print(f"3. Processing chats for {len(video_details)} videos...")
    for video in video_details:
        print(f"   Downloading: {video['title']} ({video['id']})")
        chats = download_chat_replay(video["id"])
        
        for chat in chats:
            if is_spam_attendance(chat["Message"]):
                continue
            all_chats.append({
                "Waktu": chat["Time Offset (Seconds)"],
                "Username": chat["Author"],
                "Pesan": chat["Message"],
                "Video ID": video["id"],
                "Video Title": video["title"]
            })
            
    if not all_chats:
        print("No valid chats extracted.")
        return
        
    df = pd.DataFrame(all_chats)
    df = df.dropna(subset=['Pesan'])
    df = df[df['Pesan'].str.strip() != '']
    df = df.drop_duplicates(subset=['Username', 'Pesan']).copy()
    
    print(f"4. Running sentiment analysis on {len(df)} comments...")
    comments = df["Pesan"].tolist()
    sentiments = []
    batch_size = 50
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        batch_sentiments = analyze_sentiment_gemini_batch(batch, gemini_key, model_name)
        sentiments.extend(batch_sentiments)
        time.sleep(1.0)
        
    df["Sentiment"] = sentiments
    
    # Save a backup local CSV
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"5. Backup CSV saved to: {CSV_OUTPUT_PATH}")
    
    # Upload/overwrite Google Sheet
    update_google_sheet(df, SPREADSHEET_NAME)
    print("ETL Pipeline completed successfully.")

if __name__ == "__main__":
    main()
