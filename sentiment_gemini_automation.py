# sentiment_gemini_automation.py
# -----------------------------------------------------------------------------
# Gemini-Powered Automation Script for YouTube Chat Sentiment Analysis
# BPSDM Jatim Data Science Project (Free-Tier API Batching Method)
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
# CONFIGURATION & MONKEYPATCH
# -----------------------------------------------------------------------------

# YouTube's recent transition to "lockupViewModel" schema broke scrapetube's 
# native scraper. We monkeypatch scrapetube to correctly read stream items.
scrapetube.scrapetube.type_property_map["streams"] = "lockupViewModel"

# Target YouTube Channel Info for BPSDM Jatim
CHANNEL_ID = "UCS9pHfAZEeelx1tYtDh_ZAg"
LIMIT_VIDEOS = 3
CSV_OUTPUT_PATH = "bpsdm_gemini_sentiment_results.csv"

# -----------------------------------------------------------------------------
# DATA CLEANSING FUNCTION
# -----------------------------------------------------------------------------

def is_spam_attendance(text):
    """
    Cleanses the message by identifying if it matches Indonesian or English 
    attendance logs/spam (e.g. 'Hadir', 'Nama/NIP', 'Instansi', 'nyimak', 
    'assalamualaikum', etc.).
    """
    text_clean = text.strip().lower()
    text_normalized = text_clean.replace('_', ' ').replace('-', ' ')
    
    if not text_normalized:
        return True # Treat empty/whitespace-only messages as spam
        
    # 1. Filter out common Islamic/general greetings and responses (e.g., 'assalamualaikum', 'waalaikumsalam')
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
            
    # 2. Filter out 'nyimak' (paying attention) or 'menyimak' or 'simak' variations
    nyimak_patterns = [
        r'\bnyimak[k]*\b', r'\bmenyimak\b', r'\bsimak\b', r'\bmenonton\b', r'\bnonton\b'
    ]
    for pattern in nyimak_patterns:
        if re.search(pattern, text_normalized):
            return True
    
    # 3. Check for 'hadir' word variations (with custom word boundaries)
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

    # 4. Check for typical registration/attendance metadata labels
    spam_keywords = [
        r'\bnip\b', r'\babsensi\b', r'\babsen\b', r'\bpresensi\b', r'\bdaftar\s+hadir\b', 
        r'\blink\s+(presensi|absen|daftar\s+hadir|sertifikat)\b',
        r'\bnama\s*:', r'\bnip\s*:', r'\binstansi\s*:', r'\bunit\s*kerja\s*:', r'\bjabatan\s*:'
    ]
    for pattern in spam_keywords:
        if re.search(pattern, text_normalized):
            return True
            
    # 5. Check for numeric segments representing NIP (Employee ID numbers are 9-18 digits)
    if re.search(r'\b\d{9,18}\b', text_normalized):
        return True
        
    return False

# -----------------------------------------------------------------------------
# GEMINI BATCH SENTIMENT CLASSIFIER (REST API)
# -----------------------------------------------------------------------------

def get_available_flash_model(api_key):
    """
    Queries Google AI API to find the best available Flash model.
    Falls back to 'gemini-1.5-flash' if it fails or if no models are returned.
    """
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
                selected = flash_models[0]
                print(f"    [Gemini Info] Found available Flash models: {flash_models}. Selected: {selected}")
                return selected
    except Exception as e:
        print(f"    [Gemini Warning] Could not retrieve available models: {e}")
        
    print("    [Gemini Info] Falling back to default: gemini-1.5-flash")
    return "gemini-1.5-flash"

def analyze_sentiment_gemini_batch(messages, api_key, model_name):
    """
    Classifies a batch of chat messages using Gemini via Google's REST API.
    Enforces structured JSON output matching input IDs to maintain alignment.
    """
    if not messages:
        return []
        
    # Map input messages to a list of dicts with unique IDs
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
    # Use the discovered model name dynamically with v1beta endpoint for responseMimeType support
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"    [Gemini Error] API failed (Code: {response.status_code}): {response.text}")
            return ["Neutral"] * len(messages)
            
        res_json = response.json()
        candidate_text = res_json['candidates'][0]['content']['parts'][0]['text']
        results = json.loads(candidate_text.strip())
        
        # Build index mapping for sentiment values
        sentiment_map = {item['id']: item['sentiment'] for item in results}
        
        # Generate aligned output list
        labels = []
        for idx in range(len(messages)):
            label = sentiment_map.get(idx, "Neutral")
            # Clean and normalize label case
            label = label.strip().title()
            if label not in ["Positive", "Negative", "Neutral"]:
                label = "Neutral"
            labels.append(label)
        return labels
        
    except Exception as e:
        print(f"    [Gemini Error] Exception during classification batch: {e}")
        return ["Neutral"] * len(messages)

# -----------------------------------------------------------------------------
# CORE YOUTUBE INTERNAL API INTERACTION
# -----------------------------------------------------------------------------

def extract_json_from_html(html, regex_pattern):
    match = re.search(regex_pattern, html)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None

def fetch_video_metadata(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    
    print(f"  [GET] Connecting to video page watch?v={video_id}...")
    response = session.get(url, timeout=15)
    if response.status_code != 200:
        raise Exception(f"HTTP Error {response.status_code} requesting video watch page")
        
    html = response.text
    ytcfg_json = extract_json_from_html(html, r'ytcfg\.set\(({.+?})\);')
    yt_initial_data = extract_json_from_html(html, r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;')
    
    if not ytcfg_json or not yt_initial_data:
        raise Exception("Failed to parse page configuration keys (ytcfg/ytInitialData)")
        
    api_key = ytcfg_json.get("INNERTUBE_API_KEY")
    context = ytcfg_json.get("INNERTUBE_CONTEXT")
    
    try:
        conversation_bar = yt_initial_data['contents']['twoColumnWatchNextResults']['conversationBar']
        live_chat_renderer = conversation_bar['liveChatRenderer']
        continuations = live_chat_renderer['continuations']
        continuation_token = continuations[0]['reloadContinuationData']['continuation']
    except KeyError:
        continuation_token = None
        
    return session, api_key, context, continuation_token

def download_chat_replay(video_id):
    try:
        session, api_key, context, token = fetch_video_metadata(video_id)
    except Exception as e:
        print(f"  [Error] Failed to initialize video metadata: {e}")
        return []
        
    if not token:
        print("  [Warning] No chat replay continuation token found.")
        return []
        
    session.headers.update({
        "Content-Type": "application/json",
        "Referer": f"https://www.youtube.com/live_chat_replay?continuation={token}"
    })
    
    messages = []
    batch_index = 0
    
    while token:
        batch_index += 1
        api_url = f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat_replay?key={api_key}"
        payload = {
            "context": context,
            "continuation": urllib.parse.unquote(token)
        }
        
        response = session.post(api_url, json=payload, timeout=15)
        if response.status_code != 200:
            break
            
        res_json = response.json()
        continuation_contents = res_json.get("continuationContents", {})
        live_chat_continuation = continuation_contents.get("liveChatContinuation", {})
        actions = live_chat_continuation.get("actions", [])
        
        if not actions:
            break
            
        new_batch_messages = 0
        for action in actions:
            replay_action = action.get("replayChatItemAction", {})
            inner_actions = replay_action.get("actions", [])
            if not inner_actions:
                continue
            for inner_action in inner_actions:
                item = inner_action.get("addChatItemAction", {}).get("item", {})
                if not item:
                    continue
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
                    new_batch_messages += 1
                    
        print(f"  Batch {batch_index}: Retrieved {new_batch_messages} chat messages.")
        
        continuations = live_chat_continuation.get("continuations", [])
        if continuations:
            con_data = continuations[0].get("liveChatReplayContinuationData") or continuations[0].get("reloadContinuationData")
            if con_data:
                token = con_data.get("continuation")
            else:
                token = None
        else:
            token = None
            
        time.sleep(0.5)
        
    return messages

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATION PIPELINE
# -----------------------------------------------------------------------------

def main():
    print("==========================================================")
    print(" BPSDM JATIM YOUTUBE LIVE CHAT SENTIMENT ANALYZER (GEMINI)")
    print("==========================================================")
    
    # 1. Retrieve Gemini API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please enter your Gemini API Key: ").strip()
        if not api_key:
            print("ERROR: Gemini API Key is required to run this script.")
            return
            
    # Discover best available model dynamically
    print("\nDetecting available models on your API key...")
    model_name = get_available_flash_model(api_key)
            
    # 2. Fetch Stream Video IDs and Titles
    print(f"\n1. Fetching latest {LIMIT_VIDEOS} stream uploads from channel ID: {CHANNEL_ID}...")
    try:
        videos = scrapetube.get_channel(channel_id=CHANNEL_ID, content_type="streams")
        recent_videos = []
        for v in videos:
            recent_videos.append(v)
            if len(recent_videos) >= LIMIT_VIDEOS:
                break
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to retrieve video list: {e}")
        return
        
    if not recent_videos:
        print("No livestream videos found on the channel.")
        return
        
    video_details = []
    for idx, v in enumerate(recent_videos):
        video_id = v.get("contentId")
        title = v.get("metadata", {}).get("lockupMetadataViewModel", {}).get("title", {}).get("content", "Unknown Title")
        video_details.append({"id": video_id, "title": title})
        print(f"   [{idx+1}] ID: {video_id} | Title: {title}")
        
    all_records = []
    
    # 3. Extract and Clean Chats per Video
    print("\n2. Scraping and cleansing chat history for each video...")
    for idx, video in enumerate(video_details):
        v_id = video["id"]
        print(f"\nProcessing Video [{idx+1}/{LIMIT_VIDEOS}] ID: {v_id}...")
        
        raw_chats = download_chat_replay(v_id)
        print(f"Total raw chat messages retrieved: {len(raw_chats)}")
        
        clean_count = 0
        video_cleansed_chats = []
        
        for chat in raw_chats:
            msg = chat["Message"]
            
            # Skip if classified as spam attendance, greeting, or observational feedback
            if is_spam_attendance(msg):
                continue
                
            clean_count += 1
            video_cleansed_chats.append({
                "Waktu": chat["Time Offset (Seconds)"],
                "Username": chat["Author"],
                "Pesan": msg
            })
            
        print(f"Cleansing finished: {clean_count} valid messages retained ({len(raw_chats) - clean_count} spam logs removed).")
        all_records.extend(video_cleansed_chats)
        
    # 4. Consolidate results & Remove Duplicates before sentiment analysis (saves API usage)
    print("\n3. Consolidating records and removing duplicate rows...")
    if not all_records:
        print("No valid chat records retrieved after cleansing.")
        return
        
    df = pd.DataFrame(all_records)
    
    # Remove empty/whitespace rows in 'Pesan'
    df = df.dropna(subset=['Pesan'])
    df = df[df['Pesan'].str.strip() != '']
    
    # Deduplicate unique user messages to keep clean datasets and minimize API load
    df = df.drop_duplicates(subset=['Username', 'Pesan']).copy()
    print(f"Total unique records to analyze: {len(df)}")
    
    # 5. Perform Batch Sentiment Analysis using Gemini API
    print(f"\n4. Analyzing sentiments in batches of 50 using {model_name}...")
    pesan_list = df['Pesan'].tolist()
    sentiments = []
    batch_size = 50
    
    for i in range(0, len(pesan_list), batch_size):
        batch = pesan_list[i:i+batch_size]
        print(f"   Processing batch {i // batch_size + 1} ({len(batch)} items)...")
        
        # Fetch sentiment results for this batch
        batch_results = analyze_sentiment_gemini_batch(batch, api_key, model_name)
        sentiments.extend(batch_results)
        
        # Brief sleep to stay safe from the 15 Requests Per Minute limit
        time.sleep(1.0)
        
    # Assign the results back to the DataFrame
    df['Sentiment'] = sentiments
    
    # 6. Export to CSV file
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSuccess! Cleansed and analyzed results exported to '{CSV_OUTPUT_PATH}'")
    print(f"Total unique data entries generated: {len(df)}")
    
    # Print statistics summary
    print("\nSummary of overall sentiments:")
    print(df["Sentiment"].value_counts().to_string())

if __name__ == "__main__":
    main()
