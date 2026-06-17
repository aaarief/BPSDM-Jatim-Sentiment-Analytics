# sentiment_automation.py
# -----------------------------------------------------------------------------
# Automation Script for YouTube Live Chat Sentiment Analysis at BPSDM Jatim
# -----------------------------------------------------------------------------

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

# YouTube recent transition to "lockupViewModel" schema broke scrapetube's 
# native scraper. We monkeypatch scrapetube to correctly read stream items.
scrapetube.scrapetube.type_property_map["streams"] = "lockupViewModel"

# Target YouTube Channel Info for BPSDM Jatim
CHANNEL_ID = "UCS9pHfAZEeelx1tYtDh_ZAg"  # The unique channel ID for BPSDM Jatim
LIMIT_VIDEOS = 3                         # Number of recent live stream videos to process
CSV_OUTPUT_PATH = "bpsdm_sentiment_results.csv" # Output file path for CSV

# -----------------------------------------------------------------------------
# SENTIMENT ANALYSIS DICTIONARY (INDONESIAN LEXICON)
# -----------------------------------------------------------------------------

# Curated list of Indonesian positive terms frequently seen in webinar sessions
POSITIVE_WORDS = {
    "bagus", "baik", "mantap", "hebat", "keren", "luar biasa", "menarik", "bermanfaat", 
    "sangat bermanfaat", "ilmu baru", "terima kasih", "makasih", "suwun", "matur nuwun",
    "jelas", "sangat jelas", "paham", "mengerti", "senang", "gembira", "puas", "top",
    "sukses", "jaya", "maju", "semangat", "inspirasi", "menginspirasi", "luar-biasa",
    "memuaskan", "luarbiasa", "kreatif", "inovatif", "setuju", "mendukung", "applause",
    "salut", "apresiasi", "tertarik", "suka", "cinta", "bangga", "terbantu", "membantu",
    "lancar", "alhamdulillah", "puji tuhan", "barakallah", "berkah"
}

# Curated list of Indonesian negative terms highlighting audio/video issues or user dissatisfaction
NEGATIVE_WORDS = {
    "jelek", "buruk", "kurang", "lambat", "lemot", "putus", "terputus", "macet",
    "lag", "buffer", "buffering", "error", "salah", "kecewa", "sedih", "marah",
    "sulit", "susah", "bingung", "pusing", "rumit", "gagal", "tidak jelas", "kurang jelas",
    "samar", "hilang", "suara hilang", "audio buruk", "video macet", "tidak mengerti",
    "tidak paham", "bosan", "membosankan", "tidak setuju", "menolak", "keberatan",
    "lamban", "lemah", "kacau", "parah", "rugi", "mengecewakan", "kesulitan", "terganggu",
    "tidak kedengaran", "tidak terdengar", "kecil suaranya", "suara kecil",
    "kresek", "berisik", "no signal", "blank", "gelap", "buram", "kabur"
}

# -----------------------------------------------------------------------------
# DATA CLEANSING FUNCTION
# -----------------------------------------------------------------------------

def is_spam_attendance(text):
    """
    Cleanses the message by identifying if it matches Indonesian or English 
    attendance logs/spam (e.g. 'Hadir', 'Nama/NIP', 'Instansi', 'nyimak', 
    'assalamualaikum', etc.).
    Returns True if the message is spam, False if it is a valid comment.
    """
    # Convert text to lowercase and strip whitespaces
    text_clean = text.strip().lower()
    
    # Normalize underscores and hyphens to spaces to handle patterns like 'BPKAD_HADIR'
    text_normalized = text_clean.replace('_', ' ').replace('-', ' ')
    
    if not text_normalized:
        return True # Treat empty/whitespace-only messages as spam/junk
        
    # 1. Filter out common Islamic/general greetings and responses (e.g., 'assalamualaikum', 'waalaikumsalam')
    greetings = [
        # Match variations of assalamualaikum
        r'as+a+l+a+m+u+a+l+a+i+k+u+m',
        r'as+a+l+a+m+u\s*a+l+a+i+k+u+m',
        r'as+a+l+a+m\s*a+l+a+i+k+u+m',
        r'as+a+l+a+m+u\'*a+l+a+i+k+u+m',
        r'\baskum\b',
        
        # Match variations of waalaikumsalam / walaikumsalam
        r'w[a\']*l+a+i+k+u+m',
        r'w+a+[\'\s]*a+l+a+i+k+u+m',
        r'w+a+l+a+i+k+u+m\s*s+a+l+a+m',
        r'w+a+l+a+i+k+u+m+s+a+l+a+m',
        r'w+a+l+a+i+k+u+m+s+l+a+m',
        r'w+a+l+a+i+k+u+m+s+l+m',
        r'wa*laik*us*alam',
        
        # Other religion greetings
        r'shalom', r'om\s+swastiastu', r'namo\s+buddhaya', r'salam\s+kebajikan',
        
        # Standalone short greetings/salutations or signatures
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
    # Matches 'hadir', 'hadiir', 'hadirrr', 'hadr', 'hadiroh', 'hafir', but excludes sub-words like 'kehadiran'
    if re.search(r'\b(hadi+r+|hadr|hadl[ir]+|hadiroh|hadiro|hadirat|hafir)\b', text_normalized):
        # Specific patterns indicating common attendance formulas
        short_spam_patterns = [
            r'^hadir$',  # Matches exact word 'hadir'
            r'^hadir+[ \.]*(pak|bu|bos|min|admin|dan menyimak|menyimak|dan mengikuti|mengikuti|selalu|ikut)?[\.]*$',
            r'^(siap|izin|ijin|saya|kami)\s+hadir', # Matches 'siap hadir', 'izin hadir', etc.
            # Matches 'hadir di/dari/prov/kab/dinas/sekolah/puskesmas/upt...'
            r'^hadir\s+(di|dari|prov|kab|kec|dinas|instansi|sd|smp|sma|smk|rs|rssa|puskesmas|upt|sekolah|badan|biro|setda|bpkad|dlh|rsud)',
            # Matches messages containing agency details followed by/preceded by 'hadir'
            r'(dinas|instansi|badan|biro|setda|bpkad|dlh|rs|smk|smp|sd|rsud|kab|prov|kec|upt|puskesmas|sekolah|rs|rssa)\b.*\bhadir',
            r'\bhadir\b.*\b(dinas|instansi|badan|biro|setda|bpkad|dlh|rs|smk|smp|sd|rsud|kab|prov|kec|upt|puskesmas|sekolah|rs|rssa)\b'
        ]
        # Iterate over each regex pattern
        for pattern in short_spam_patterns:
            if re.search(pattern, text_normalized):
                return True
                
        # If the message is very short (less than 50 characters) and contains "hadir", 
        # it is almost certainly a simple attendance log.
        if len(text_normalized) < 50:
            return True

    # 4. Check for typical registration/attendance metadata labels
    spam_keywords = [
        r'\bnip\b', r'\babsensi\b', r'\babsen\b', r'\bpresensi\b', r'\bdaftar\s+hadir\b', 
        r'\blink\s+(presensi|absen|daftar\s+hadir|sertifikat)\b', # Filter admin link messages
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
# SENTIMENT CLASSIFICATION FUNCTION
# -----------------------------------------------------------------------------

def analyze_sentiment(text):
    """
    Performs custom lexicon-based Indonesian sentiment analysis with negation detection.
    Classifies the text into 'Positive', 'Negative', or 'Neutral'.
    """
    # Lowercase and clean input text
    text_clean = text.lower().strip()
    
    # Tokenize text into words (removing punctuation)
    words = re.findall(r'\b\w+\b', text_clean)
    
    # If no valid words found, classify as Neutral
    if not words:
        return "Neutral"
        
    pos_score = 0  # Initialize positive score accumulator
    neg_score = 0  # Initialize negative score accumulator
    
    # Standard negation words in Indonesian
    negations = {"tidak", "bukan", "belum", "ga", "gak", "ndak", "kurang"}
    
    i = 0
    # Iterate through the list of tokenized words
    while i < len(words):
        word = words[i]
        
        # Look ahead for bigrams and trigrams
        phrase_2 = " ".join(words[i:i+2]) if i + 1 < len(words) else ""
        phrase_3 = " ".join(words[i:i+3]) if i + 2 < len(words) else ""
        
        is_phrase_matched = False
        
        # 1. Match Trigram
        if phrase_3 in POSITIVE_WORDS:
            # If negation precedes the phrase, flip sentiment score to negative
            if i > 0 and words[i-1] in negations:
                neg_score += 1
            else:
                pos_score += 1
            i += 3
            is_phrase_matched = True
        elif phrase_3 in NEGATIVE_WORDS:
            # If negation precedes a negative phrase, double negation flips it positive
            if i > 0 and words[i-1] in negations:
                pos_score += 1
            else:
                neg_score += 1
            i += 3
            is_phrase_matched = True
            
        # 2. Match Bigram
        if not is_phrase_matched and phrase_2:
            if phrase_2 in POSITIVE_WORDS:
                if i > 0 and words[i-1] in negations:
                    neg_score += 1
                else:
                    pos_score += 1
                i += 2
                is_phrase_matched = True
            elif phrase_2 in NEGATIVE_WORDS:
                if i > 0 and words[i-1] in negations:
                    pos_score += 1
                else:
                    neg_score += 1
                i += 2
                is_phrase_matched = True
                
        # 3. Match Monogram (Single word)
        if not is_phrase_matched:
            if word in POSITIVE_WORDS:
                if i > 0 and words[i-1] in negations:
                    neg_score += 1
                else:
                    pos_score += 1
            elif word in NEGATIVE_WORDS:
                if i > 0 and words[i-1] in negations:
                    pos_score += 1
                else:
                    neg_score += 1
            i += 1
            
    # Assign classification based on final score counts
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

# -----------------------------------------------------------------------------
# CORE YOUTUBE INTERNAL API INTERACTION
# -----------------------------------------------------------------------------

def extract_json_from_html(html, regex_pattern):
    """
    Utility helper to extract structured JSON data embedded in YouTube HTML scripts.
    """
    match = re.search(regex_pattern, html)
    if not match:
        return None
    try:
        # Load the matched JSON text segment into a python dictionary
        return json.loads(match.group(1))
    except Exception:
        return None

def fetch_video_metadata(video_id):
    """
    Simulates a browser loading a YouTube video page to extract client context parameters,
    the InnerTube API Key, and the initial live chat reload continuation token.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Establish a persistent requests session
    session = requests.Session()
    # Mask headers to represent a typical desktop Chrome browser request
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    # Accept cookies consent to bypass region redirects
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    
    print(f"  [GET] Connecting to video page watch?v={video_id}...")
    response = session.get(url, timeout=15)
    
    if response.status_code != 200:
        raise Exception(f"HTTP Error {response.status_code} while requesting video page")
        
    html = response.text
    
    # Regex pattern to locate the main configuration JSON ytcfg
    ytcfg_json = extract_json_from_html(html, r'ytcfg\.set\(({.+?})\);')
    # Regex pattern to locate the initial page rendering JSON ytInitialData
    yt_initial_data = extract_json_from_html(html, r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;')
    
    if not ytcfg_json or not yt_initial_data:
        raise Exception("Failed to parse page variables (ytcfg or ytInitialData) from video HTML")
        
    # Extract the internal API Key required for InnerTube API queries
    api_key = ytcfg_json.get("INNERTUBE_API_KEY")
    # Extract client context dictionary (details like client name, version, platform, etc.)
    context = ytcfg_json.get("INNERTUBE_CONTEXT")
    
    # Locate the initial live chat reload continuation token in the page DOM structure
    try:
        conversation_bar = yt_initial_data['contents']['twoColumnWatchNextResults']['conversationBar']
        live_chat_renderer = conversation_bar['liveChatRenderer']
        continuations = live_chat_renderer['continuations']
        continuation_token = continuations[0]['reloadContinuationData']['continuation']
    except KeyError:
        # If there is no live chat replays on this video (e.g. disabled or stream didn't finish processing)
        continuation_token = None
        
    return session, api_key, context, continuation_token

def download_chat_replay(video_id):
    """
    Queries YouTube's InnerTube POST endpoint repeatedly in a pagination loop to retrieve
    all chat replay actions/messages for the specified video ID.
    """
    try:
        # Load watch page to fetch context keys
        session, api_key, context, token = fetch_video_metadata(video_id)
    except Exception as e:
        print(f"  [Error] Failed to initialize video metadata: {e}")
        return []
        
    if not token:
        print("  [Warning] No chat replay continuation token found. Replay might be disabled/unavailable.")
        return []
        
    # Ensure correct content-type header for InnerTube payload delivery
    session.headers.update({
        "Content-Type": "application/json",
        "Referer": f"https://www.youtube.com/live_chat_replay?continuation={token}"
    })
    
    messages = []
    batch_index = 0
    
    # Loop pagination requests as long as a continuation token is returned
    while token:
        batch_index += 1
        # InnerTube Endpoint URL for Live Chat Replay
        api_url = f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat_replay?key={api_key}"
        
        # Assemble payload
        payload = {
            "context": context,
            "continuation": urllib.parse.unquote(token)
        }
        
        # Send HTTP POST request
        response = session.post(api_url, json=payload, timeout=15)
        
        if response.status_code != 200:
            print(f"  [Error] API POST failed (Code: {response.status_code}). Exiting loop.")
            break
            
        res_json = response.json()
        
        # Navigate through response paths to find actions
        continuation_contents = res_json.get("continuationContents", {})
        live_chat_continuation = continuation_contents.get("liveChatContinuation", {})
        actions = live_chat_continuation.get("actions", [])
        
        if not actions:
            break
            
        new_batch_messages = 0
        for action in actions:
            # Extract underlying message details
            replay_action = action.get("replayChatItemAction", {})
            inner_actions = replay_action.get("actions", [])
            if not inner_actions:
                continue
            for inner_action in inner_actions:
                item = inner_action.get("addChatItemAction", {}).get("item", {})
                if not item:
                    continue
                # Extract text renderer info
                text_renderer = item.get("liveChatTextMessageRenderer", {})
                if text_renderer:
                    # Extract author handle / name
                    author = text_renderer.get("authorName", {}).get("simpleText", "Unknown")
                    # Combine all message runs to form the complete string
                    message_runs = text_renderer.get("message", {}).get("runs", [])
                    message_content = "".join([run.get("text", "") for run in message_runs])
                    # Extract timestamp offset (milliseconds from start of video)
                    time_offset = replay_action.get("videoOffsetTimeMsec")
                    try:
                        time_seconds = float(time_offset) / 1000 if time_offset else 0
                    except ValueError:
                        time_seconds = 0
                        
                    # Save dictionary entry
                    messages.append({
                        "Author": author,
                        "Message": message_content,
                        "Time Offset (Seconds)": time_seconds
                    })
                    new_batch_messages += 1
                    
        print(f"  Batch {batch_index}: Retrieved {new_batch_messages} chat messages.")
        
        # Find next continuation token from response
        continuations = live_chat_continuation.get("continuations", [])
        if continuations:
            con_data = continuations[0].get("liveChatReplayContinuationData") or continuations[0].get("reloadContinuationData")
            if con_data:
                token = con_data.get("continuation")
            else:
                token = None
        else:
            token = None
            
        # Respectful pause to avoid rate-limiting triggers
        time.sleep(0.5)
        
    return messages

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATION PIPELINE
# -----------------------------------------------------------------------------

def main():
    print("==========================================================")
    print(" BPSDM JATIM YOUTUBE LIVE CHAT SENTIMENT ANALYZER SYSTEM")
    print("==========================================================")
    
    # 1. Fetching Stream Video IDs and Titles
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
        
    # Extract metadata properties from scrapetube models
    video_details = []
    for idx, v in enumerate(recent_videos):
        video_id = v.get("contentId")
        # Navigate through the complex lockupViewModel hierarchy to get title
        title = v.get("metadata", {}).get("lockupMetadataViewModel", {}).get("title", {}).get("content", "Unknown Title")
        video_details.append({"id": video_id, "title": title})
        print(f"   [{idx+1}] ID: {video_id} | Title: {title}")
        
    all_records = []
    
    # 2. Extract and Process Chats per Video
    print("\n2. Scraping, cleansing, and analyzing chat history for each video...")
    for idx, video in enumerate(video_details):
        v_id = video["id"]
        print(f"\nProcessing Video [{idx+1}/{LIMIT_VIDEOS}] ID: {v_id}...")
        
        # Download raw messages
        raw_chats = download_chat_replay(v_id)
        print(f"Total raw chat messages retrieved: {len(raw_chats)}")
        
        clean_count = 0
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        # Process and cleanse each chat message
        for chat in raw_chats:
            msg = chat["Message"]
            
            # Skip if classified as spam attendance, greeting, or observational feedback
            if is_spam_attendance(msg):
                continue
                
            clean_count += 1
            
            # Run lexicon sentiment analysis
            sentiment_label = analyze_sentiment(msg)
            sentiment_counts[sentiment_label] += 1
            
            # Save record with columns: Waktu, Username, Pesan, and Sentiment
            all_records.append({
                "Waktu": chat["Time Offset (Seconds)"],
                "Username": chat["Author"],
                "Pesan": msg,
                "Sentiment": sentiment_label
            })
            
        print(f"Cleansing finished: {clean_count} valid messages retained ({len(raw_chats) - clean_count} spam logs removed).")
        print(f"Sentiment distribution: {json.dumps(sentiment_counts)}")
        
    # 3. Assemble Pandas DataFrame & Export
    print("\n3. Consolidating results into Pandas DataFrame...")
    if not all_records:
        print("No valid chat records retrieved after cleansing.")
        return
        
    df = pd.DataFrame(all_records)
    
    # Remove any empty or whitespace-only rows in the 'Pesan' column
    df = df.dropna(subset=['Pesan'])
    df = df[df['Pesan'].str.strip() != '']
    
    # Remove duplicate rows (same Username and Pesan) to keep messages unique
    df = df.drop_duplicates(subset=['Username', 'Pesan'])
    
    # Export DataFrame containing Waktu, Username, Pesan, and Sentiment
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Success! Cleansed and analyzed results exported to '{CSV_OUTPUT_PATH}'")
    print(f"Total unique data entries generated: {len(df)}")
    
    # Print statistics summary
    print("\nSummary of overall sentiments:")
    print(df["Sentiment"].value_counts().to_string())

# Entry point of script
if __name__ == "__main__":
    main()
