# clean_step1.py
import os
import re
import time
from glob import glob
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd

# ========= 配置 =========
SOURCE_FOLDER = "raw_parquet_data/cc_index_parquet"
WORK_FOLDER = "datacleanworkfolder"
os.makedirs(WORK_FOLDER, exist_ok=True)

SKIP_IF_OUTPUT_EXISTS = False
WATCH_MODE = True
POLL_SECONDS = 5
FILE_STABLE_SECONDS = 20

MAX_ID_ITEMS_PER_FOLDER = 2
MIN_ROWS_PER_DOMAIN = 20
MAX_SEGMENT_LEN = 30

# ========= ✅ A. 特定 domain 直接排除 =========
EXCLUDE_DOMAINS = {
    # 1. 成人與搜尋雜訊類
    "zh.wow-xxx-videos.com",
    "zingiberaceae.godfatherxxx.com",
    "yala5a.maxsp0orts.com",
    "yala6a.maxsp0orts.com",
    "x.wow-xxx-videos.com",
    "porn-channels.com",

    # 2. SEO 垃圾站與內容採集站
    "0888988.0888988adh2.sbs",
    "126795dhy.wqias.com",
    "13bjc.mazinjy.com",
    "mazinjy.com",
    "wqias.com",
    "0888988.sbs",

    # 3. 隨機字串/短網址類
    "0-g-0.ru",
    "bit.ly",
    "t.co",

    # 4. 靜態資料查詢/雜訊站
    "zipcode.etoile.edu.gr",
    "planet-beruf.de",
    "h5.wlbps.com",
    "hi.wn.com",
    "hire.withgoogle.com",
    "m.wsccsm.com",
    "movies.wowgirls.com",
    "on.wsj.com",
    "parts.wirtgen-group.com",
    "photos.wowgirls.com",
    "pireaus.wixsite.com",
    "www.wordfind.com",
    "fr.wn.com",
    "shoutout.wix.com",
    "trends.withgoogle.com",
    "tyszkiewiczjr.wixstudio.com",
    "winwaysoft.com",
    "worldofthegrishaverse.com",
    "worldofpotter.com",
    "worldofolympians.com",
    "pde.gov.gr",
    "depod.bioss.uni-freiburg.de",
    "lba.hist.uni-marburg.de",
    "uchinadi-mannheim.de",
    "urkundenrepositorium.uni-marburg.de",
    "www.tvminton.de",
    "www.uniturm.de",
    "podologie-ramsen.de",
    "podologie-perk-meisterernst.de",
    "rederberch.de",
    "functions.wolfram.com",
    "afisha.yandex.ru",
    "praxis-fuer-loesungsorientierte-arbeit.de",
    "praxis-hernandez.de",
    "praxis-laemmerhirt.de",
    "bildungsprogramm.pi-muenchen.de",
    "biomanufaktur.schlosshamborn.de",
    "www.shisha-ssm.de",
    "www.shishagalaxy.de",
    "apply.workable.com",
    "api.world-airport-codes.com",
    "annuaire.woufipedia.com",
    "bacmap.wishartlab.com",
    "link.wpbuilds.com",
    "m.wjpyyy.com"
}

# ========= ✅ B. 排除不適合的頂級域名 (TLD) =========
BLACKLIST_TLDS = {
    "ir", "sbs", "xyz", "icu", "top", "work", "biz",
    "today", "monster", "fit", "tk", "gq", "ml", "ga", "cf"
}

# ========= ✅ C. 排除特定關鍵字網域 =========
BAD_DOMAIN_KEYWORDS = ["xxx", "porn", "video", "sex", "tube", "adult", "gamble", "casino", "produkt", "produktkatalog"]

# ========= ✅ D. 排除特定關鍵字 path（命中就丟，參數化） =========
BAD_PATH_KEYWORDS = ["casino", "gamble", "slot", "bet", "bonus", "german", "korean", "turkish", "de-DE", "pt-BR"]
BAD_PATH_SEPARATORS = r"[\/\-_]"
BAD_PATH_MATCH_MODE = "token"

MAX_DASH_PER_SEGMENT = 1
MAX_DASH_TOTAL = 4
MAX_PERCENT_ENCODED_BYTES = 4
SHORTENER_DOMAINS = {"onl.sc"}
MAX_BUCKET_ITEMS = 3
GERMAN_HIT_THRESHOLD = 3
MAX_DOTS_PER_SEGMENT = 1

KEEP_HIGH_PER_DOMAIN = None
KEEP_MID_PER_DOMAIN = 40
KEEP_LOW_PER_DOMAIN = 3
KEEP_OTHER_PER_DOMAIN = None

ENABLE_PRIORITIZE_SECURITY_PATHS = True

# ✅ 新增：同時輸出「other 價值」清單
OUTPUT_OTHER_LIST = False
OTHER_SUFFIX = ".other.csv"  # 例：xxx.csv -> xxx.other.csv

# ✅ 新增：硬性裁剪「低價值 segment」(不依賴 tier 判斷)
ENFORCE_LOW_SEGMENTS_LIMIT = True
LOW_SEGMENTS_KEEP_PER_DOMAIN = 3

# ========= Regex =========
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
LONG_HEX_RE = re.compile(r"/[a-fA-F0-9]{40,}")
ONLY_SLASHES_RE = re.compile(r"^/+$")
DOUBLE_HASH_RE = re.compile(r"/<HASH>/<HASH>(/|$)")

IMAGE_FILE_RE = re.compile(r"/[^/]+\.(jpg|jpeg|png|gif|bmp|webp|svg)$", re.I)
PDF_FILE_RE = re.compile(r"/[^/]+\.pdf$", re.I)

COMMON_EXT = r"(?:html?|php|phtml|aspx|asp|jsp|jspx|shtml|shtm|htm)"

NUMERIC_SUFFIX_EXTRACT_RE = re.compile(rf"^(?P<folder>.*/)(?P<num>\d+)(?:\.(?:{COMMON_EXT}))?$", re.I)
HEX_SUFFIX_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<hex>[0-9A-Fa-f]{8,16})$", re.I)
SLUG_NUM_SUFFIX_EXTRACT_RE = re.compile(
    rf"^(?P<folder>.*/)(?P<prefix>[A-Za-z]{{2,}}[A-Za-z0-9_-]{{0,30}})[-_](?P<num>\d{{1,10}})(?:\.(?:{COMMON_EXT}))?$",
    re.I
)
NESTED_NUM_FOLDER_ID_RE = re.compile(rf"^(?P<folder>.*/\d+/)(?P<num>\d{{1,10}})/(?:[^/]+\.(?:{COMMON_EXT}))$", re.I)
PREFIX_HEX_SUFFIX_EXTRACT_RE = re.compile(rf"^(?P<folder>.*/)(?P<prefix>[A-Za-z]{{1,5}})(?P<hex>[0-9A-Fa-f]{{8,16}})(?:\.(?:{COMMON_EXT}))?$", re.I)
PREFIX_DIGITS_SUFFIX_EXTRACT_RE = re.compile(rf"^(?P<folder>.*/)(?P<prefix>[A-Za-z]{{1,10}})(?P<num>\d{{2,14}})(?:\.(?:{COMMON_EXT}))?$", re.I)
PREFIX_SPACE_NUM_SUFFIX_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<prefix>[A-Za-z0-9]{1,20})%20(?P<num>\d{2,6})$", re.I)
ALNUM_MIXED_TOKEN_8_24_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<id>(?=[A-Za-z0-9]{8,24}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{8,24})$", re.I)
ALNUM_CODE3_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<id>(?=[A-Za-z0-9]{3}$)(?=.*\d)[A-Za-z0-9]{3})$", re.I)
FIXEDNAME_CODE3_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<name>[A-Za-z][A-Za-z0-9]{2,30})-(?P<id>[A-Za-z0-9]{3})$", re.I)
TWO_LAYER_NUMDOTHEX_EXTRACT_RE = re.compile(
    r"^(?P<folder>.*/)"
    r"(?P<num1>\d+)\.(?P<hex1>[0-9A-Fa-f]{8,40})/"
    r"(?P<num2>\d+)\.(?P<hex2>[0-9A-Fa-f]{8,40})/"
    r"(?P<tail>[^/]+)$",
    re.I
)
DATE_SUFFIX_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<date>(?:19|20)\d{2}-\d{2}-\d{2}|(?:19|20)\d{6})$")
NUM_PARENT_CHILD_EXTRACT_RE = re.compile(r"^(?P<folder>.*/\d+/)(?P<num>\d+)$")
NUM_UNDERSCORE_NUM_EXTRACT_RE = re.compile(rf"^(?P<folder>.*/)(?P<a>\d+)_(?P<b>\d+)(?:\.(?:{COMMON_EXT}))?$", re.I)
NUM_UNDERSCORE_MULTI_EXTRACT_RE = re.compile(rf"^(?P<folder>.*/)(?P<seq>\d+(?:_\d+){{2,}})(?:\.(?:{COMMON_EXT}))?$", re.I)
TRIPLE_NUM_SEGMENT_EXTRACT_RE = re.compile(r"^(?P<folder>.*/)(?P<n1>\d{1,6})/(?P<n2>\d{1,6})/(?P<n3>\d{1,6})(?:/.*)?$")
WIX_SO_CODE_C_EXTRACT_RE = re.compile(r"^(?P<folder>.*/so/)(?P<code>[A-Za-z0-9_-]{6,32})/(?P<tail>c)$", re.I)
NEW_LOWER_ALPHA_10_EXTRACT_RE = re.compile(r"^(?P<folder>.*/new/)(?P<slug>[a-z]{10})$")
PRODUCTS_SLUG_EXTRACT_RE = re.compile(r"^(?P<folder>.*/products/)(?P<slug>[^/]{1,80})$")
LONG_MIXED_CODE_AFTER_NUMFOLDER_RE = re.compile(r"^(?P<folder>.*/\d+/)(?P<code>(?=.{12,80}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]{12,80})$")
YANDEX_DASH_FOLDER_CODE_RE = re.compile(r"^(?P<folder>.*/-/)(?P<code>[A-Za-z0-9~_-]{4,32})$", re.I)
PARENTDIR_LONGCODE_RE = re.compile(r"^(?P<folder>.*/[A-Za-z0-9]{6,24}/)(?P<code>(?=.{12,100}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]{12,100})$")
SINGLE_SEG_SHORTCODE_RE = re.compile(r"^(?P<folder>/)(?P<code>[A-Za-z0-9]{5,12})$")
WIZWID_PRODUCT_DETAIL_RE = re.compile(r"^(?P<folder>.*/product/)(?P<code>[A-Za-z]{1,6}\d{8,20})/detail$", re.I)
BASE64URL_SINGLESEG_RE = re.compile(r"^(?P<folder>/)(?P<code>(?=.{12,64}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]+)$", re.I)
BASE64URL_UNDER_S_RE = re.compile(r"^(?P<folder>.*/s/)(?P<code>(?=.{6,64}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]+)$", re.I)
BASE64URL_UNDER_SHAARE_RE = re.compile(r"^(?P<folder>.*/shaare/)(?P<code>(?=.{5,64}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]+)$", re.I)
STELLENANGEBOTE_APPLY_RE = re.compile(r"^(?P<folder>.*/stellenangebote/)(?P<num>\d{4,12})/bewerbung$", re.I)

PCT_BYTE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
EMBEDDED_DOMAIN_RE = re.compile(r"^[A-Za-z0-9-]+\.[A-Za-z]{2,}$")
COMMON_EXT_SET = {
    "htm", "html", "php", "phtml", "asp", "aspx", "jsp", "jspx", "shtml", "shtm",
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "pdf", "js", "css", "map", "ico",
    "woff", "woff2", "ttf", "eot", "mp4", "mp3", "webm", "avi", "mov", "mkv", "zip"
}

DIGITS_DIR_DIGITS_HTML_RE = re.compile(r"/\d{6,10}/\d+\.(?:html?|shtml|htm)$", re.I)
TRIPLE_NUM_ANYWHERE_RE = re.compile(r"(?:^|/)\d{1,10}/\d{1,10}/\d{1,10}(?:/|$)")

HEURISTIC_JUNK_RE = re.compile(
    r"(?:^|/)(?:19|20)\d{2}/(?:0?[1-9]|1[0-2])(?:/|$)|"
    r"/(?:[a-z0-9]{32,})(?:/|$)|"
    r"/\d+\.html$|"
    r"/_wp_link_placeholder(?:/|$)|"
    r"/wp-json/oembed(?:/|$)",
    re.I
)

# ========= ✅ 你新增的三種「只留三筆」格式 =========
PATTERN_NUMNUMNUM_WORD_RE = re.compile(
    r"^(?P<folder>.*/)(?P<a>\d{1,10})_(?P<b>\d{1,10})_(?P<c>\d{1,10})_(?P<w>[A-Za-z]+)\.(?P<ext>html|htm)$",
    re.I
)
PATTERN_ENEN_LONGDIGITS_RE = re.compile(
    r"^(?P<folder>.*/)(?P<prefix>[A-Za-z]+-[A-Za-z]+)_(?P<num>\d{8,})\.(?P<ext>html|htm)$",
    re.I
)
PATTERN_NUMDIR_NUMDOTNUM_RE = re.compile(
    r"^(?P<folder>.*/)(?P<n1>\d{1,10})/(?P<n2>\d{1,10})\.(?P<n3>\d{1,10})\.(?P<ext>html|htm)$",
    re.I
)

# ========= 分桶 Regex =========
CONTENT_PAGE_RE = re.compile(
    r"^/(?P<section>news|post|article)/(?P<slug>(?:\d{4}-\d{2}-\d{2}|[A-Za-z0-9_-]{1,120}))(?:/|$)",
    re.I
)
LIST_PAGE_RE = re.compile(
    r"^/(?P<section>list)/(?P<a>\d{1,10})/(?P<b>\d{1,12})/(?P<file>[A-Za-z0-9_-]{1,40}\.(?:html?|php|aspx|jsp|jspx|shtml|htm))$",
    re.I
)
SHARE_S_RE = re.compile(r"^/s/(?P<code>[A-Za-z0-9~_-]{4,120})(?:/|$)", re.I)
SHARE_SHARE_RE = re.compile(r"^/share/(?P<code>[A-Za-z0-9~_-]{4,160})(?:/|$)", re.I)
SHARE_DASH_RE = re.compile(r"^/-/(?P<code>[A-Za-z0-9~_-]{2,120})(?:/|$)", re.I)
MEDIA_MEDIAS_RE = re.compile(r"^/medias/(?P<id>[A-Za-z0-9_-]{4,160})(?:/|$)", re.I)
MEDIA_ASSETS_RE = re.compile(r"^/assets/(?P<rest>.+)$", re.I)
PRODUCT_DETAIL_BUCKET_RE = re.compile(r"^/product/(?P<id>[A-Za-z0-9_-]{3,120})/detail(?:/|$)", re.I)

AUTHORS_BUCKET_RE = re.compile(
    r"^/(?P<section>aiia-authors|authors?|author|people|staff|team|contributors?|implementers)/(?P<slug>[A-Za-z0-9_-]{1,160})(?:/|$)",
    re.I
)
SPECIALIST_BUCKET_RE = re.compile(
    r"^/(?P<section>specialist|experts?)/(?P<slug>[A-Za-z0-9,_-]{1,160})(?:/|$)",
    re.I
)

# ========= ✅ Path 黑名單 regex（參數化） =========
def compile_bad_path_re(keywords, mode="token", seps=r"[\/\-_]"):
    kws = [k.strip() for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not kws:
        return None
    union = "|".join(map(re.escape, kws))

    if str(mode).lower() == "substring":
        return re.compile(rf"(?i)({union})")

    return re.compile(rf"(?i)(?:^|{seps})(?:{union})(?:$|{seps})")

BAD_PATH_RE = compile_bad_path_re(BAD_PATH_KEYWORDS, mode=BAD_PATH_MATCH_MODE, seps=BAD_PATH_SEPARATORS)

# ========= German heuristic =========
GERMAN_TOKENS = [
    "landestheater", "spielstaetten", "stuecke", "startseite",
    "buehne", "theater", "sinfoniekonzert", "theaternacht",
    "soehne", "zaehmung", "raeuber", "zauberfloete", "zimmerschlacht",
    "kreidekreis", "kirschgarten", "menschenfeind", "widerspenstigen",
    "sparschwein", "goldene", "brunnen", "ronny", "kredit", "fiskus", "biberpelz",
    "kaukasische", "domschule", "duenen", "husum", "schleswig", "flensburg", "itzehoe",
]
GERMAN_TOKEN_RE = re.compile(r"(?i)(?:^|/|-|_)(%s)(?:$|/|-|_)" % "|".join(map(re.escape, GERMAN_TOKENS)))

def german_token_hits(path: str) -> int:
    if not isinstance(path, str) or not path:
        return 0
    p = path.strip().lower().rstrip("/")
    hits = set(m.group(1).lower() for m in GERMAN_TOKEN_RE.finditer(p))
    return len(hits)

# ========= helper =========
def normalize_domain(d: str) -> str:
    if not isinstance(d, str):
        return ""
    d = d.strip().lower()
    if not d:
        return ""
    d = d.split(":")[0]
    d = d.rstrip(".")
    d = d.strip("[]")
    return d

def get_tld(domain: str) -> str:
    if not isinstance(domain, str):
        return ""
    d = domain.strip().lower()
    if not d:
        return ""
    d = d.split(":")[0]
    parts = [p for p in d.split(".") if p]
    return parts[-1] if parts else ""

def is_bad_domain(domain: str) -> bool:
    d = normalize_domain(domain)
    if not d:
        return True

    if d in EXCLUDE_DOMAINS:
        return True
    for base in EXCLUDE_DOMAINS:
        if d.endswith("." + base):
            return True

    for kw in BAD_DOMAIN_KEYWORDS:
        if kw in d:
            return True

    if get_tld(d) in BLACKLIST_TLDS:
        return True

    return False

def has_too_long_segment(path: str, max_len: int) -> bool:
    if not isinstance(path, str):
        return True
    segs = [s for s in path.split("/") if s]
    for s in segs:
        if s in ("<HASH>", "<IMAGE>", "<PDF>"):
            continue
        if len(s) > max_len:
            return True
    return False

def has_too_many_dots_per_segment(path: str, max_dots: int) -> bool:
    if not isinstance(path, str):
        return True
    segs = [s for s in path.split("/") if s]
    for s in segs:
        if s in ("<HASH>", "<IMAGE>", "<PDF>"):
            continue
        if s.count(".") > max_dots:
            return True
    return False

def has_too_many_dashes_per_segment(path: str, max_dashes: int) -> bool:
    if not isinstance(path, str):
        return True
    segs = [s for s in path.split("/") if s]
    for s in segs:
        if s in ("<HASH>", "<IMAGE>", "<PDF>"):
            continue
        if s.count("-") > max_dashes:
            return True
    return False

def has_too_many_dashes_total(path: str, max_total: int) -> bool:
    if not isinstance(path, str):
        return True
    return path.count("-") > max_total

def has_too_many_pct_bytes(path: str, max_cnt: int) -> bool:
    if not isinstance(path, str):
        return True
    return len(PCT_BYTE_RE.findall(path)) > max_cnt

def _base36_to_int(s: str) -> float:
    if not isinstance(s, str) or not s:
        return np.nan
    s = s.strip().lower()
    try:
        return float(int(s, 36))
    except Exception:
        return np.nan

def _hashish_to_float(s: str) -> float:
    if not isinstance(s, str) or not s:
        return np.nan
    t = s.strip().lower()
    t = re.sub(r"[~_\-]", "", t)
    if not t:
        return np.nan
    t = t[:16]
    t = re.sub(r"[^a-z0-9]", "", t)
    if not t:
        return np.nan
    try:
        return float(int(t, 36))
    except Exception:
        return np.nan

def _has_embedded_domain_tail(path: str) -> bool:
    segs = [s for s in str(path).split("/") if s]
    if not segs:
        return False

    last_raw = segs[-1].strip()
    last = unquote(last_raw).strip().strip("\u00a0")

    if "." not in last:
        return False

    ext = last.rsplit(".", 1)[-1].lower()
    if ext in COMMON_EXT_SET:
        return False

    return EMBEDDED_DOMAIN_RE.fullmatch(last) is not None

# ========= 清洗核心 =========
def clean_url_core(url: str):
    """回傳 (domain, cleaned_path) 或 (None, None)"""
    try:
        if not isinstance(url, str):
            return None, None

        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        if not domain or not path:
            return None, None

        domain = normalize_domain(domain)
        if not domain:
            return None, None

        if is_bad_domain(domain):
            return None, None

        path = str(path).strip()

        if not path or path.strip("/") == "":
            return None, None
        if ONLY_SLASHES_RE.match(path):
            return None, None

        # ✅ path 關鍵字命中就刪（/ - _ 分隔，參數化）
        if BAD_PATH_RE is not None and BAD_PATH_RE.search(path):
            return None, None

        # ✅ '+' 超過 1 個才刪
        if path.count("+") > 1:
            return None, None

        # 你之前要求刪除的符號類
        if '"' in path or "," in path or "'" in path or "@" in path or "=" in path:
            return None, None

        if german_token_hits(path) >= GERMAN_HIT_THRESHOLD:
            return None, None

        if has_too_many_pct_bytes(path, MAX_PERCENT_ENCODED_BYTES):
            return None, None

        if has_too_many_dashes_per_segment(path, MAX_DASH_PER_SEGMENT):
            return None, None
        if has_too_many_dashes_total(path, MAX_DASH_TOTAL):
            return None, None

        low = path.lower()
        if "http:" in low or "https:" in low:
            return None, None

        # ✅ .../數字/數字/數字/... 任何位置命中就丟
        if TRIPLE_NUM_ANYWHERE_RE.search(path):
            return None, None

        # ✅ 垃圾路徑特徵
        if HEURISTIC_JUNK_RE.search(path):
            return None, None

        # 指定要刪：.../<digits>/<digits>.html
        if DIGITS_DIR_DIGITS_HTML_RE.search(path):
            return None, None

        # 正規化長 hash / 圖片 / PDF
        path = LONG_HEX_RE.sub("/<HASH>", path)
        path = IMAGE_FILE_RE.sub("/<IMAGE>", path)
        path = PDF_FILE_RE.sub("/<PDF>", path)

        if DOUBLE_HASH_RE.search(path):
            return None, None

        # 資源型標記直接丟掉
        if "<IMAGE>" in path or "<PDF>" in path or "<HASH>" in path:
            return None, None

        # ✅ 任一 segment '.' 超過門檻就丟
        if has_too_many_dots_per_segment(path, MAX_DOTS_PER_SEGMENT):
            return None, None

        # 去除非 ASCII（允許 %）
        safe_path = path.replace("%", "")
        if NON_ASCII_RE.search(safe_path):
            return None, None

        # segment 過長丟掉
        if has_too_long_segment(path, MAX_SEGMENT_LEN):
            return None, None

        # embedded-domain tail 丟掉
        if _has_embedded_domain_tail(path):
            return None, None

        if not path or path.strip("/<>") == "":
            return None, None

        return domain, path

    except Exception:
        return None, None

# ========= 舊版：ID/序號限縮 =========
def limit_id_suffix_per_folder(df: pd.DataFrame) -> pd.DataFrame:
    # （保留你的原邏輯）
    df = df.copy()
    df["cleaned_path"] = df["cleaned_path"].astype(str).str.strip()

    trimmed = df["cleaned_path"].str.rstrip("/")
    trimmed = trimmed.where(trimmed.str.len() > 0, np.nan)

    # --- extract ---
    num_ex = trimmed.str.extract(NUMERIC_SUFFIX_EXTRACT_RE)
    hex_ex = trimmed.str.extract(HEX_SUFFIX_EXTRACT_RE)
    slug_ex = trimmed.str.extract(SLUG_NUM_SUFFIX_EXTRACT_RE)
    nested_ex = trimmed.str.extract(NESTED_NUM_FOLDER_ID_RE)
    phex_ex = trimmed.str.extract(PREFIX_HEX_SUFFIX_EXTRACT_RE)
    pdig_ex = trimmed.str.extract(PREFIX_DIGITS_SUFFIX_EXTRACT_RE)
    sp_ex = trimmed.str.extract(PREFIX_SPACE_NUM_SUFFIX_EXTRACT_RE)

    token_ex = trimmed.str.extract(ALNUM_MIXED_TOKEN_8_24_EXTRACT_RE)
    code3_ex = trimmed.str.extract(ALNUM_CODE3_EXTRACT_RE)
    fixed3_ex = trimmed.str.extract(FIXEDNAME_CODE3_EXTRACT_RE)
    twohex_ex = trimmed.str.extract(TWO_LAYER_NUMDOTHEX_EXTRACT_RE)
    date_ex = trimmed.str.extract(DATE_SUFFIX_EXTRACT_RE)
    numpc_ex = trimmed.str.extract(NUM_PARENT_CHILD_EXTRACT_RE)

    nn_ex = trimmed.str.extract(NUM_UNDERSCORE_NUM_EXTRACT_RE)
    nnn_ex = trimmed.str.extract(NUM_UNDERSCORE_MULTI_EXTRACT_RE)
    tri_ex = trimmed.str.extract(TRIPLE_NUM_SEGMENT_EXTRACT_RE)

    wix_ex = trimmed.str.extract(WIX_SO_CODE_C_EXTRACT_RE)
    new10_ex = trimmed.str.extract(NEW_LOWER_ALPHA_10_EXTRACT_RE)
    prod_ex = trimmed.str.extract(PRODUCTS_SLUG_EXTRACT_RE)

    longcode_ex = trimmed.str.extract(LONG_MIXED_CODE_AFTER_NUMFOLDER_RE)
    yandex_ex = trimmed.str.extract(YANDEX_DASH_FOLDER_CODE_RE)
    parent_long_ex = trimmed.str.extract(PARENTDIR_LONGCODE_RE)

    short_ex = trimmed.str.extract(SINGLE_SEG_SHORTCODE_RE)
    wizwid_ex = trimmed.str.extract(WIZWID_PRODUCT_DETAIL_RE)

    b64_single_ex = trimmed.str.extract(BASE64URL_SINGLESEG_RE)
    b64_s_ex = trimmed.str.extract(BASE64URL_UNDER_S_RE)
    b64_shaare_ex = trimmed.str.extract(BASE64URL_UNDER_SHAARE_RE)
    stellen_ex = trimmed.str.extract(STELLENANGEBOTE_APPLY_RE)

    # --- flags ---
    num_val = pd.to_numeric(num_ex["num"], errors="coerce")
    is_num = num_val.notna() & num_ex["folder"].notna()

    hex_val = hex_ex["hex"]
    is_hex = hex_val.notna() & hex_ex["folder"].notna()

    slug_num_val = pd.to_numeric(slug_ex["num"], errors="coerce")
    is_slug = slug_num_val.notna() & slug_ex["folder"].notna()

    nested_num_val = pd.to_numeric(nested_ex["num"], errors="coerce")
    is_nested = nested_num_val.notna() & nested_ex["folder"].notna()

    phex_val = phex_ex["hex"]
    is_phex = phex_val.notna() & phex_ex["folder"].notna()

    pdig_num_val = pd.to_numeric(pdig_ex["num"], errors="coerce")
    is_pdig = pdig_num_val.notna() & pdig_ex["folder"].notna()

    sp_num_val = pd.to_numeric(sp_ex["num"], errors="coerce")
    is_sp = sp_num_val.notna() & sp_ex["folder"].notna()

    token_val = token_ex["id"]
    is_token = token_val.notna() & token_ex["folder"].notna()

    code3_val = code3_ex["id"]
    is_code3 = code3_val.notna() & code3_ex["folder"].notna()

    fixed3_val = fixed3_ex["id"]
    is_fixed3 = fixed3_val.notna() & fixed3_ex["folder"].notna()

    is_twohex = twohex_ex["folder"].notna() & twohex_ex["num1"].notna() & twohex_ex["num2"].notna()

    date_val = date_ex["date"]
    is_date = date_val.notna() & date_ex["folder"].notna()

    numpc_val = pd.to_numeric(numpc_ex["num"], errors="coerce")
    is_numpc = numpc_val.notna() & numpc_ex["folder"].notna()

    nn_b = pd.to_numeric(nn_ex["b"], errors="coerce")
    is_nn = nn_b.notna() & nn_ex["folder"].notna() & nn_ex["a"].notna()

    is_nnn = nnn_ex["seq"].notna() & nnn_ex["folder"].notna()

    tri_n3 = pd.to_numeric(tri_ex["n3"], errors="coerce")
    is_tri = tri_n3.notna() & tri_ex["folder"].notna()

    is_wix = wix_ex["code"].notna() & wix_ex["folder"].notna()
    is_new10 = new10_ex["slug"].notna() & new10_ex["folder"].notna()
    is_products = prod_ex["slug"].notna() & prod_ex["folder"].notna()

    longcode_val = longcode_ex["code"]
    is_longcode = longcode_val.notna() & longcode_ex["folder"].notna()

    yandex_code = yandex_ex["code"]
    is_yandex = yandex_code.notna() & yandex_ex["folder"].notna()

    parent_long_code = parent_long_ex["code"]
    is_parent_long = parent_long_code.notna() & parent_long_ex["folder"].notna()

    short_code = short_ex["code"]
    is_short = (
        short_code.notna()
        & short_ex["folder"].notna()
        & df["domain"].astype(str).str.lower().isin(SHORTENER_DOMAINS)
    )

    wiz_code = wizwid_ex["code"]
    is_wizwid = wiz_code.notna() & wizwid_ex["folder"].notna()

    b64_single_code = b64_single_ex["code"]
    is_b64_single = b64_single_code.notna() & b64_single_ex["folder"].notna()

    b64_s_code = b64_s_ex["code"]
    is_b64_s = b64_s_code.notna() & b64_s_ex["folder"].notna()

    b64_shaare_code = b64_shaare_ex["code"]
    is_b64_shaare = b64_shaare_code.notna() & b64_shaare_ex["folder"].notna()

    stellen_num = pd.to_numeric(stellen_ex["num"], errors="coerce")
    is_stellen = stellen_num.notna() & stellen_ex["folder"].notna()

    is_id = (
        is_num | is_hex | is_slug | is_nested | is_phex | is_pdig | is_sp
        | is_token | is_code3 | is_fixed3 | is_twohex | is_date | is_numpc
        | is_nn | is_nnn | is_tri
        | is_wix | is_new10 | is_products
        | is_longcode | is_yandex | is_parent_long
        | is_short | is_wizwid
        | is_b64_single | is_b64_s | is_b64_shaare | is_stellen
    )

    non_id_df = df[~is_id].copy()
    id_df = df[is_id].copy()
    if id_df.empty:
        return df.drop_duplicates(subset=["domain", "cleaned_path"])

    folder_arr = np.full(len(df), "", dtype=object)
    type_arr = np.full(len(df), "", dtype=object)
    key_num_arr = np.full(len(df), np.nan, dtype="float64")

    def _set(folder_series, mask, typ, key_series=None):
        folder_arr[mask.to_numpy()] = folder_series.loc[mask].astype(str).to_numpy()
        type_arr[mask.to_numpy()] = typ
        if key_series is not None:
            key_num_arr[mask.to_numpy()] = pd.to_numeric(key_series.loc[mask], errors="coerce").to_numpy(dtype="float64")

    _set(num_ex["folder"], is_num, "num", num_val)
    _set(slug_ex["folder"], is_slug, "slugnum", slug_num_val)

    if is_hex.any():
        _set(hex_ex["folder"], is_hex, "hex")
        hex_series = hex_val.loc[is_hex].astype(str)
        hex_int = hex_series.apply(lambda x: int(x, 16) if re.fullmatch(r"[0-9A-Fa-f]{8,16}", x) else np.nan)
        key_num_arr[is_hex.to_numpy()] = pd.to_numeric(hex_int, errors="coerce").to_numpy(dtype="float64")

    _set(nested_ex["folder"], is_nested, "nestednum", nested_num_val)
    _set(pdig_ex["folder"], is_pdig, "prefixdigits", pdig_num_val)

    if is_phex.any():
        _set(phex_ex["folder"], is_phex, "prefixhex")
        hex_series2 = phex_val.loc[is_phex].astype(str)
        hex_int2 = hex_series2.apply(lambda x: int(x, 16) if re.fullmatch(r"[0-9A-Fa-f]{8,16}", x) else np.nan)
        key_num_arr[is_phex.to_numpy()] = pd.to_numeric(hex_int2, errors="coerce").to_numpy(dtype="float64")

    _set(sp_ex["folder"], is_sp, "prefixspace", sp_num_val)
    _set(token_ex["folder"], is_token, "token")

    if is_code3.any():
        _set(code3_ex["folder"], is_code3, "code3")
        key_num_arr[is_code3.to_numpy()] = pd.to_numeric(
            code3_val.loc[is_code3].astype(str).apply(_base36_to_int),
            errors="coerce",
        ).to_numpy(dtype="float64")

    if is_fixed3.any():
        _set(fixed3_ex["folder"], is_fixed3, "fixedname_code3")
        key_num_arr[is_fixed3.to_numpy()] = pd.to_numeric(
            fixed3_val.loc[is_fixed3].astype(str).apply(_base36_to_int),
            errors="coerce",
        ).to_numpy(dtype="float64")

    if is_twohex.any():
        _set(twohex_ex["folder"], is_twohex, "two_layer_numdothex")
        n1 = pd.to_numeric(twohex_ex.loc[is_twohex, "num1"], errors="coerce").fillna(0).astype("int64")
        n2 = pd.to_numeric(twohex_ex.loc[is_twohex, "num2"], errors="coerce").fillna(0).astype("int64")
        h1 = twohex_ex.loc[is_twohex, "hex1"].astype(str).str.slice(0, 8).apply(
            lambda x: int(x, 16) if re.fullmatch(r"[0-9A-Fa-f]{1,8}", x) else 0
        )
        h2 = twohex_ex.loc[is_twohex, "hex2"].astype(str).str.slice(0, 8).apply(
            lambda x: int(x, 16) if re.fullmatch(r"[0-9A-Fa-f]{1,8}", x) else 0
        )
        mixed = (n1 * 10_000_000_000_000) + (n2 * 10_000_000_000) + (h1 * 100_000) + (h2 % 100_000)
        key_num_arr[is_twohex.to_numpy()] = pd.to_numeric(mixed, errors="coerce").to_numpy(dtype="float64")

    if is_date.any():
        _set(date_ex["folder"], is_date, "date")
        date_norm = date_val.loc[is_date].astype(str).str.replace("-", "", regex=False)
        key_num_arr[is_date.to_numpy()] = pd.to_numeric(date_norm, errors="coerce").to_numpy(dtype="float64")

    if is_numpc.any():
        _set(numpc_ex["folder"], is_numpc, "numpc", numpc_val)

    if is_nn.any():
        _set(nn_ex["folder"], is_nn, "num_num", nn_b)

    if is_nnn.any():
        _set(nnn_ex["folder"], is_nnn, "num_multi")
        first_num = nnn_ex.loc[is_nnn, "seq"].astype(str).str.split("_").str[0]
        key_num_arr[is_nnn.to_numpy()] = pd.to_numeric(first_num, errors="coerce").to_numpy(dtype="float64")

    if is_tri.any():
        _set(tri_ex["folder"], is_tri, "tri_num", tri_n3)

    if is_wix.any():
        _set(wix_ex["folder"], is_wix, "wix_so_code_c")
        code_norm = wix_ex.loc[is_wix, "code"].astype(str).str.replace(r"[_-]", "", regex=True)
        key_num_arr[is_wix.to_numpy()] = pd.to_numeric(code_norm.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_new10.any():
        _set(new10_ex["folder"], is_new10, "new_alpha10")
        key_num_arr[is_new10.to_numpy()] = pd.to_numeric(
            new10_ex.loc[is_new10, "slug"].astype(str).apply(_base36_to_int),
            errors="coerce",
        ).to_numpy(dtype="float64")

    if is_products.any():
        _set(prod_ex["folder"], is_products, "products_slug")

    if is_longcode.any():
        _set(longcode_ex["folder"], is_longcode, "longcode_after_numfolder")
        code_norm2 = longcode_val.loc[is_longcode].astype(str).str.replace(r"[_-]", "", regex=True)
        key_num_arr[is_longcode.to_numpy()] = pd.to_numeric(code_norm2.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_yandex.any():
        _set(yandex_ex["folder"], is_yandex, "yandex_dash_folder")
        y_norm = yandex_code.loc[is_yandex].astype(str).str.replace(r"[~_-]", "", regex=True)
        key_num_arr[is_yandex.to_numpy()] = pd.to_numeric(y_norm.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_parent_long.any():
        _set(parent_long_ex["folder"], is_parent_long, "parentdir_longcode")
        p_norm = parent_long_code.loc[is_parent_long].astype(str).str.replace(r"[_-]", "", regex=True)
        key_num_arr[is_parent_long.to_numpy()] = pd.to_numeric(p_norm.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_short.any():
        _set(short_ex["folder"], is_short, "shortener_single_seg")
        key_num_arr[is_short.to_numpy()] = pd.to_numeric(
            short_code.loc[is_short].astype(str).apply(_base36_to_int),
            errors="coerce",
        ).to_numpy(dtype="float64")

    if is_wizwid.any():
        _set(wizwid_ex["folder"], is_wizwid, "wizwid_product_detail")
        digits = wiz_code.loc[is_wizwid].astype(str).str.extract(r"(\d{8,20})")[0]
        key_num_arr[is_wizwid.to_numpy()] = pd.to_numeric(digits, errors="coerce").to_numpy(dtype="float64")

    if is_b64_single.any():
        _set(b64_single_ex["folder"], is_b64_single, "b64_single")
        norm = b64_single_code.loc[is_b64_single].astype(str).str.replace(r"[_-]", "", regex=True)
        key_num_arr[is_b64_single.to_numpy()] = pd.to_numeric(norm.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_b64_s.any():
        _set(b64_s_ex["folder"], is_b64_s, "b64_under_s")
        norm = b64_s_code.loc[is_b64_s].astype(str).str.replace(r"[_-]", "", regex=True)
        key_num_arr[is_b64_s.to_numpy()] = pd.to_numeric(norm.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_b64_shaare.any():
        _set(b64_shaare_ex["folder"], is_b64_shaare, "b64_under_shaare")
        norm = b64_shaare_code.loc[is_b64_shaare].astype(str).str.replace(r"[_-]", "", regex=True)
        key_num_arr[is_b64_shaare.to_numpy()] = pd.to_numeric(norm.apply(_base36_to_int), errors="coerce").to_numpy(dtype="float64")

    if is_stellen.any():
        _set(stellen_ex["folder"], is_stellen, "stellenangebote_apply", stellen_num)

    id_df["_folder"] = pd.Series(folder_arr, index=df.index).loc[id_df.index].astype(str)
    id_df["_type"] = pd.Series(type_arr, index=df.index).loc[id_df.index].astype(str)
    id_df["_key_num"] = pd.Series(key_num_arr, index=df.index).loc[id_df.index].astype("float64")

    id_df = id_df.sort_values(by=["domain", "_folder", "_type", "_key_num", "cleaned_path"], kind="mergesort")
    id_df = id_df.groupby(["domain", "_folder"], as_index=False, group_keys=False).head(MAX_ID_ITEMS_PER_FOLDER)
    id_df = id_df.drop(columns=["_folder", "_type", "_key_num"], errors="ignore")

    out = pd.concat([non_id_df, id_df], ignore_index=True)
    out = out.drop_duplicates(subset=["domain", "cleaned_path"])
    return out

# ========= 分桶只留三筆 =========
def _bucketize_path(path: str):
    if not isinstance(path, str):
        return None, None, np.nan

    p = path.strip()
    if not p.startswith("/"):
        p = "/" + p
    p = p.rstrip("/")

    m = CONTENT_PAGE_RE.match(p)
    if m:
        section = m.group("section").lower()
        slug = m.group("slug")
        key = f"bucket:content:{section}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", slug):
            order = float(int(slug.replace("-", "")))
        else:
            order = _hashish_to_float(slug)
        return "content", key, order

    m = LIST_PAGE_RE.match(p)
    if m:
        file = m.group("file").lower()
        key = f"bucket:list:{file}"
        order = float(int(m.group("b")))
        return "list", key, order

    m = SHARE_S_RE.match(p)
    if m:
        key = "bucket:share:/s"
        order = _hashish_to_float(m.group("code"))
        return "share", key, order

    m = SHARE_SHARE_RE.match(p)
    if m:
        key = "bucket:share:/share"
        order = _hashish_to_float(m.group("code"))
        return "share", key, order

    m = SHARE_DASH_RE.match(p)
    if m:
        key = "bucket:share:/-"
        order = _hashish_to_float(m.group("code"))
        return "share", key, order

    m = MEDIA_MEDIAS_RE.match(p)
    if m:
        key = "bucket:media:/medias"
        order = _hashish_to_float(m.group("id"))
        return "media", key, order

    m = MEDIA_ASSETS_RE.match(p)
    if m:
        key = "bucket:media:/assets"
        order = _hashish_to_float(m.group("rest"))
        return "media", key, order

    m = PRODUCT_DETAIL_BUCKET_RE.match(p)
    if m:
        key = "bucket:product:/product/detail"
        order = _hashish_to_float(m.group("id"))
        return "product", key, order

    m = AUTHORS_BUCKET_RE.match(p)
    if m:
        key = f"bucket:people:{m.group('section').lower()}"
        order = _hashish_to_float(m.group("slug"))
        return "people", key, order

    m = SPECIALIST_BUCKET_RE.match(p)
    if m:
        key = f"bucket:people:{m.group('section').lower()}"
        order = _hashish_to_float(m.group("slug"))
        return "people", key, order

    return None, None, np.nan

def limit_buckets_keep_3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaned_path"] = df["cleaned_path"].astype(str).str.strip()

    b = df["cleaned_path"].apply(_bucketize_path)
    df["_bkey"] = b.apply(lambda x: x[1])
    df["_okey"] = b.apply(lambda x: x[2])

    in_bucket = df["_bkey"].notna()
    bucket_df = df[in_bucket].copy()
    other_df = df[~in_bucket].copy()

    if bucket_df.empty:
        return df.drop(columns=["_bkey", "_okey"], errors="ignore")

    bucket_df = bucket_df.sort_values(by=["domain", "_bkey", "_okey", "cleaned_path"], kind="mergesort")
    bucket_df = bucket_df.groupby(["domain", "_bkey"], as_index=False, group_keys=False).head(MAX_BUCKET_ITEMS)

    out = pd.concat([other_df, bucket_df], ignore_index=True)
    out = out.drop_duplicates(subset=["domain", "cleaned_path"])
    out = out.drop(columns=["_bkey", "_okey"], errors="ignore")
    return out

# ========= ✅ 三種格式「只留三筆」 =========
def _special_pattern_bucketize(path: str):
    if not isinstance(path, str) or not path:
        return None, None, np.nan

    p = path.strip()
    if not p.startswith("/"):
        p = "/" + p
    p = p.rstrip("/")

    m = PATTERN_NUMNUMNUM_WORD_RE.match(p)
    if m:
        folder = m.group("folder")
        a = int(m.group("a"))
        b = int(m.group("b"))
        c = int(m.group("c"))
        w = m.group("w").lower()
        ext = m.group("ext").lower()
        key = f"sp1:num_num_num_word:{folder}:{w}:{ext}"
        order = float(a * 1_000_000_000_000 + b * 1_000_000 + c)
        return "sp1", key, order

    m = PATTERN_ENEN_LONGDIGITS_RE.match(p)
    if m:
        folder = m.group("folder")
        prefix = m.group("prefix").lower()
        num = int(m.group("num"))
        ext = m.group("ext").lower()
        key = f"sp2:enen_longdigits:{folder}:{prefix}:{ext}"
        order = float(num)
        return "sp2", key, order

    m = PATTERN_NUMDIR_NUMDOTNUM_RE.match(p)
    if m:
        folder = m.group("folder")
        n1 = int(m.group("n1"))
        n2 = int(m.group("n2"))
        n3 = int(m.group("n3"))
        ext = m.group("ext").lower()
        key = f"sp3:numdir_numdotnum:{folder}:{ext}"
        order = float(n1 * 1_000_000_000_000 + n2 * 1_000_000 + n3)
        return "sp3", key, order

    return None, None, np.nan

def limit_special_patterns_keep_3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaned_path"] = df["cleaned_path"].astype(str).str.strip()

    sp = df["cleaned_path"].apply(_special_pattern_bucketize)
    df["_spkey"] = sp.apply(lambda x: x[1])
    df["_spord"] = sp.apply(lambda x: x[2])

    in_sp = df["_spkey"].notna()
    sp_df = df[in_sp].copy()
    other_df = df[~in_sp].copy()

    if sp_df.empty:
        return df.drop(columns=["_spkey", "_spord"], errors="ignore")

    sp_df = sp_df.sort_values(by=["domain", "_spkey", "_spord", "cleaned_path"], kind="mergesort")
    sp_df = sp_df.groupby(["domain", "_spkey"], as_index=False, group_keys=False).head(3)

    out = pd.concat([other_df, sp_df], ignore_index=True)
    out = out.drop_duplicates(subset=["domain", "cleaned_path"])
    out = out.drop(columns=["_spkey", "_spord"], errors="ignore")
    return out

def drop_small_domains(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    counts = df.groupby("domain")["cleaned_path"].transform("count")
    return df[counts >= min_rows].copy()

# ========= ✅ 安全價值分類（高/中/低/其他） =========
HIGH_VALUE_RE = re.compile(
    r"(?i)(?:^|/)(?:admin|administrator|manage|management|dashboard|console|panel|cp|wp-admin|backend|control|debug|test|demo|dev|logs|"
    r"login|logout|signin|signout|auth|oauth|sso|callback|token|session|jwt|signup|register|services|"
    r"password|passwd|reset|forgot|recover|mfa|2fa|verify|account|my-account|"
    r"api|v1|v2|v3|graphql|rest|rpc|soap|swagger|portal|webapp|service|"
    r"import|backup|db|sql|dump|storage|"
    r"env|config|configuration|settings|setup|install|init|private|secret|vault|"
    r"shell|cmd|exec|eval|run|script|cgi-bin|bin|git|svn|npm|vendor|docker|yml|yaml|xml|json|asp|aspx|ashx|asmx"
    r")(?:/|$)"
)

MID_VALUE_RE = re.compile(
    r"(?i)(?:^|/)(?:"
    r"query|filter|find|explore|upload|uploads|user|users|download|downloads|"
    r"checkout|cart|order|orders|billing|invoice|payment|subscription|plan|profile|profiles|board|export|component|location|site-map|js|operator|operators|agent|"
    r"file|files|status|health|metrics|staging|"
    r"support|help|ticket|form|feedback|mail|email|message|inbox|community|html|devices|device|"
    r"app|ui"
    r")(?:/|$)"
)

LOW_VALUE_RE = re.compile(
    r"(?i)(?:^|/)(?:"
    r"about|imprint|privacy|terms|legal|aviso-legal|disclaimer|search|searches|info|doc|docs|view|list|lists|academy|providers|provider|business|media|job|jobs|opportunities|"
    r"contact|support-us|donate|"
    r"gallery|galleries|photo|photos|image|images|video|videos|"
    r"news|publication|publications|article|articles|artist|post|posts|blog|blogs|archive|archiv|archives|calendar|albums|themes|theme|events|event|collection|collections|items|commodity|brands|shop-by-categories|shop|shops|produkt|"
    r"product|products|product-tag|product-tags|category|categories|tag|tags|topic|topics|page|pages|courses|book|books|academics|admissions|person|areas|reward|"
    r"review|reviews|comment|comments|rating|settings|setting|"
    r"author|authors|people|staff|team|experts?|specialist|porn|porn-maker|porns|pornstar|pornstars|keywords|keyword|premium|watch|sex|onlineshop|contacts|contact|project|catalog|catalogs|"
    r"assets|static|resource|resources|css|fonts?|font|index|marketplace|join|sale|sales|img|imgs|document|documents|forum|language|languages|"
    r"en|es|fr|de|zh|tw|it|pt|ru|jp|kr|"
    r"stichwort|kategorie|kategorien"
    r")(?:/|$)"
)

LOW_SEGMENTS = {
    "tag", "tags", "category", "categories", "topic", "topics", "page", "pages",
    "archive", "archives", "archiv", "blog", "blogs", "post", "posts", "article", "articles",
    "stichwort", "kategorie", "kategorien",
}

def _path_segments(path: str):
    if not isinstance(path, str) or not path:
        return []
    p = path.strip()
    p = p.strip("/")
    if not p:
        return []
    return [s for s in p.split("/") if s]

def _is_forced_low_by_segments(path: str) -> bool:
    segs = _path_segments(path)
    if not segs:
        return False
    s0 = segs[0].lower()
    s1 = segs[1].lower() if len(segs) >= 2 else ""
    return (s0 in LOW_SEGMENTS) or (s1 in LOW_SEGMENTS)

def _keep_by_domain(df_part: pd.DataFrame, keep):
    if df_part.empty:
        return df_part
    if keep is None:
        return df_part
    keep = int(keep)
    return df_part.groupby("domain", group_keys=False).head(keep)

def assign_tier(df: pd.DataFrame) -> pd.DataFrame:
    """加上 _tier: 0=high,1=mid,2=low,3=other"""
    df = df.copy()
    p = df["cleaned_path"].astype(str)

    forced_low = p.apply(_is_forced_low_by_segments)

    is_high = p.str.contains(HIGH_VALUE_RE, regex=True)
    is_mid  = p.str.contains(MID_VALUE_RE, regex=True)
    is_low  = forced_low | p.str.contains(LOW_VALUE_RE, regex=True)

    df["_tier"] = np.select(
        [is_high, is_mid, is_low],
        [0, 1, 2],
        default=3,
    )
    return df

def prioritize_security_paths(
    df: pd.DataFrame,
    keep_high=KEEP_HIGH_PER_DOMAIN,
    keep_mid=KEEP_MID_PER_DOMAIN,
    keep_low=KEEP_LOW_PER_DOMAIN,
    keep_other=KEEP_OTHER_PER_DOMAIN,
) -> pd.DataFrame:
    df = df.copy()
    if "domain" not in df.columns or "cleaned_path" not in df.columns:
        return df

    df = assign_tier(df)
    df = df.sort_values(by=["domain", "_tier", "cleaned_path"], kind="mergesort")

    high_df = _keep_by_domain(df[df["_tier"] == 0], keep_high)
    mid_df  = _keep_by_domain(df[df["_tier"] == 1], keep_mid)
    low_df  = _keep_by_domain(df[df["_tier"] == 2], keep_low)
    oth_df  = _keep_by_domain(df[df["_tier"] == 3], keep_other)

    out = pd.concat([high_df, mid_df, low_df, oth_df], ignore_index=True)
    out = out.drop(columns=["_tier"], errors="ignore")
    out = out.drop_duplicates(subset=["domain", "cleaned_path"])
    return out

# ========= ✅ 硬性裁剪：低價值 segments 每 domain 只留 N =========
def enforce_low_segments_limit(df: pd.DataFrame, keep_per_domain: int = 3) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    p = df["cleaned_path"].astype(str)
    low_mask = p.apply(_is_forced_low_by_segments)

    low_df = df[low_mask].copy()
    rest_df = df[~low_mask].copy()

    if low_df.empty:
        return df

    # 穩定排序：先依 domain，再依 cleaned_path
    low_df = low_df.sort_values(by=["domain", "cleaned_path"], kind="mergesort")
    low_df = low_df.groupby("domain", group_keys=False).head(int(keep_per_domain))

    out = pd.concat([rest_df, low_df], ignore_index=True)
    out = out.drop_duplicates(subset=["domain", "cleaned_path"])
    return out

# ========= 主處理 =========
def process_one_file(parquet_path: str):
    filename = os.path.basename(parquet_path).replace(".parquet", ".csv")
    output_path = os.path.join(WORK_FOLDER, filename)

    other_output_path = None
    if OUTPUT_OTHER_LIST:
        other_output_path = os.path.join(WORK_FOLDER, filename.replace(".csv", OTHER_SUFFIX))

    if SKIP_IF_OUTPUT_EXISTS and os.path.exists(output_path) and (not OUTPUT_OTHER_LIST or os.path.exists(other_output_path)):
        print(f"[SKIP] 已存在輸出：{output_path}" + (f" & {other_output_path}" if other_output_path else ""))
        return

    print(f"\n處理中: {parquet_path} -> {filename}")

    df_raw = pd.read_parquet(parquet_path, columns=["url"])
    total_in = len(df_raw)

    df_raw[["domain", "cleaned_path"]] = pd.DataFrame(
        df_raw["url"].apply(clean_url_core).tolist(),
        index=df_raw.index,
    )

    df = df_raw.dropna(subset=["domain", "cleaned_path"]).copy()
    df["domain"] = df["domain"].astype(str).str.strip()
    df["cleaned_path"] = df["cleaned_path"].astype(str).str.strip()
    df = df[df["domain"].str.lower().ne("nan")]
    df = df[df["cleaned_path"].str.lower().ne("nan")]

    df = df.drop_duplicates(subset=["domain", "cleaned_path"])
    after_core = len(df)

    before_id = len(df)
    df = limit_id_suffix_per_folder(df)
    df = df.drop_duplicates(subset=["domain", "cleaned_path"])
    after_id = len(df)

    before_bucket = len(df)
    df = limit_buckets_keep_3(df)
    df = df.drop_duplicates(subset=["domain", "cleaned_path"])
    after_bucket = len(df)

    before_sp = len(df)
    df = limit_special_patterns_keep_3(df)
    df = df.drop_duplicates(subset=["domain", "cleaned_path"])
    after_sp = len(df)

    # ✅ 先標記 tier（用來產出 other 清單：以「prioritize 前」為準）
    df_tiered_before_prior = assign_tier(df)
    other_rows = df_tiered_before_prior[df_tiered_before_prior["_tier"] == 3].copy()

    before_prior = len(df)
    if ENABLE_PRIORITIZE_SECURITY_PATHS:
        df = prioritize_security_paths(df)
        df = df.drop_duplicates(subset=["domain", "cleaned_path"])
    after_prior = len(df)

    # ✅ 硬性裁剪：/category /tag /stichwort ... 每 domain 只留 N
    before_force_low = len(df)
    if ENFORCE_LOW_SEGMENTS_LIMIT:
        df = enforce_low_segments_limit(df, keep_per_domain=LOW_SEGMENTS_KEEP_PER_DOMAIN)
        df = df.drop_duplicates(subset=["domain", "cleaned_path"])
    after_force_low = len(df)

    before_small = len(df)
    df = drop_small_domains(df, MIN_ROWS_PER_DOMAIN)
    after_small = len(df)

    # ✅ other 清單：只輸出「最後主輸出 df」中存在的 domain 的 other rows
    if OUTPUT_OTHER_LIST:
        eligible_domains = set(df["domain"].unique().tolist())

        other_rows = other_rows.drop(columns=["_tier"], errors="ignore")
        other_rows = other_rows.drop_duplicates(subset=["domain", "cleaned_path"])
        other_rows = other_rows[other_rows["domain"].isin(eligible_domains)].copy()
        other_rows = other_rows.sort_values(by=["domain", "cleaned_path"], kind="mergesort")

        other_rows[["domain", "cleaned_path"]].to_csv(other_output_path, index=False, encoding="utf-8")

    df = df.sort_values(by=["domain", "cleaned_path"], kind="mergesort")
    df[["domain", "cleaned_path"]].to_csv(output_path, index=False, encoding="utf-8")

    print(f"  原始筆數: {total_in:,}")
    print(f"  clean_url_core 後有效: {after_core:,} (移除 {(total_in - after_core):,})")
    print(f"  limit_id_suffix_per_folder 後: {after_id:,} (移除 {(before_id - after_id):,})")
    print(f"  limit_buckets_keep_3 後: {after_bucket:,} (移除 {(before_bucket - after_bucket):,})")
    print(f"  limit_special_patterns_keep_3 後: {after_sp:,} (移除 {(before_sp - after_sp):,})")
    if ENABLE_PRIORITIZE_SECURITY_PATHS:
        print(f"  prioritize_security_paths 後: {after_prior:,} (移除 {(before_prior - after_prior):,})")
    if ENFORCE_LOW_SEGMENTS_LIMIT:
        print(f"  enforce_low_segments_limit 後: {after_force_low:,} (移除 {(before_force_low - after_force_low):,})")
    print(f"  drop_small_domains 後: {after_small:,} (移除 {(before_small - after_small):,})")
    print(f"  輸出: {output_path}")
    if OUTPUT_OTHER_LIST:
        print(f"  other 清單: {other_output_path} (rows={len(other_rows):,})")

def is_file_stable(path: str, stable_seconds: int) -> bool:
    try:
        mtime = os.path.getmtime(path)
        return (time.time() - mtime) >= stable_seconds
    except Exception:
        return False

# ========= 主程式入口 =========
print("--- clean_step1 啟動 ---")
print(f"SOURCE_FOLDER={SOURCE_FOLDER}")
print(f"WORK_FOLDER={WORK_FOLDER}")
print(f"WATCH_MODE={WATCH_MODE}, POLL_SECONDS={POLL_SECONDS}, FILE_STABLE_SECONDS={FILE_STABLE_SECONDS}")
print(f"SKIP_IF_OUTPUT_EXISTS={SKIP_IF_OUTPUT_EXISTS}")
print(f"EXCLUDE_DOMAINS={len(EXCLUDE_DOMAINS)}, BLACKLIST_TLDS={len(BLACKLIST_TLDS)}, BAD_DOMAIN_KEYWORDS={BAD_DOMAIN_KEYWORDS}")
print(f"BAD_PATH_KEYWORDS={BAD_PATH_KEYWORDS}, BAD_PATH_MATCH_MODE={BAD_PATH_MATCH_MODE}")
print(f"OUTPUT_OTHER_LIST={OUTPUT_OTHER_LIST}, OTHER_SUFFIX={OTHER_SUFFIX}")
print(f"ENFORCE_LOW_SEGMENTS_LIMIT={ENFORCE_LOW_SEGMENTS_LIMIT}, LOW_SEGMENTS_KEEP_PER_DOMAIN={LOW_SEGMENTS_KEEP_PER_DOMAIN}")
print("------------------------------------------------------------")

def scan_candidates():
    return sorted(glob(os.path.join(SOURCE_FOLDER, "*.parquet")))

if not WATCH_MODE:
    files = scan_candidates()
    if not files:
        print(f"找不到 parquet：{SOURCE_FOLDER}")
        raise SystemExit(1)
    print(f"--- 一次性處理 {len(files)} 個檔案 ---")
    for f in files:
        if is_file_stable(f, FILE_STABLE_SECONDS):
            process_one_file(f)
        else:
            print(f"[WAIT] 檔案可能還在寫入：{f}")
    print(f"\n--- 完成！輸出資料夾：{WORK_FOLDER} ---")
else:
    seen = set()
    while True:
        files = scan_candidates()
        new_files = [f for f in files if f not in seen]

        if not new_files:
            time.sleep(POLL_SECONDS)
            continue

        for f in new_files:
            if not is_file_stable(f, FILE_STABLE_SECONDS):
                continue

            process_one_file(f)
            seen.add(f)
