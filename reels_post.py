# -*- coding: utf-8 -*-
"""reels_post.py  -  note記事 -> 台本 -> 動画 -> Instagramリール 自動投稿

使い方（GitHub Actionsが自動で呼びます）
  python reels_post.py prepare   # 記事を選び、台本・校正・動画づくりまで
  python reels_post.py publish   # 動画をInstagramへ投稿

環境変数
  MODE               dry_run / publish
  GEMINI_API_KEY     台本づくり用（Gemini）
  IG_USER_ID         InstagramのユーザーID（数字）
  IG_ACCESS_TOKEN    Instagramのアクセストークン
  PAGES_URL          動画の公開URL（deploy-pagesが自動で渡す）
  NOTE_USER_URLNAME  noteのユーザー名（既定: kurashi_laborg）
"""
import asyncio
import datetime
import email.utils
import html
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import requests

JST = ZoneInfo("Asia/Tokyo")
NOTE_USER = os.environ.get("NOTE_USER_URLNAME", "kurashi_laborg")
IG_USER_ID = os.environ.get("IG_USER_ID", "").strip()
IG_TOKEN = os.environ.get("IG_ACCESS_TOKEN", "").strip()
IG_API_VERSION = os.environ.get("IG_API_VERSION", "v25.0").strip()
IG_AI_LABEL = os.environ.get("IG_AI_LABEL", "1").strip() != "0"
MAX_AGE_DAYS = int(os.environ.get("MAX_AGE_DAYS", "4"))
TTS_VOICE = os.environ.get("TTS_VOICE", "ja-JP-NanamiNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+5%")

W, H = 1080, 1920
SITE_DIR = "site"
WORK_DIR = "work"
POSTED_FILE = "posted.json"
MAX_REEL_SEC = 88

# 機械チェックで弾く言い回し（見つけたら足していく）
BAD_PATTERNS = [
    (r"完全に", "断定表現「完全に」"),
    (r"完璧に", "断定表現「完璧に」"),
    (r"確実に", "断定表現「確実に」"),
    (r"誰でも", "断定表現「誰でも」"),
    (r"100\s*[%％]", "根拠のない「100%」"),
    (r"間違いなく", "断定表現「間違いなく」"),
    (r"劇的", "誇張表現「劇的」"),
    (r"運気", "根拠のない「運気」"),
    (r"脳が[^。\n]{0,20}(認識|錯覚|喜)", "根拠の弱い脳科学の言い回し"),
    (r"\d+\s*(時間|分|枚|円|万円)[^。\n]{0,10}"
     r"(生み出|浮か|丸ごと消滅|丸ごとゼロ)", "根拠のない数字の断定"),
    (r"いつでも人を呼べる", "言い過ぎ「いつでも人を呼べる」"),
    (r"いかがでしたか", "定型句「いかがでしたか」"),
]

CTA_NARRATION = "続きは、プロフィールのリンクから、noteで読めます。"
CTA_HEADING = "続きはnoteで"
CAPTION_TAIL = "続きはプロフィールのリンクのnoteで読めます。"


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- Gemini
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash",
                  "gemini-3.1-flash-lite", "gemini-2.5-flash"]
_model_cache = {"dead": set(), "found": None}


def _version_key(name):
    nums = re.findall(r"(\d+(?:\.\d+)?)", name)
    return float(nums[0]) if nums else 0.0


def discover_models(key):
    if _model_cache["found"] is not None:
        return _model_cache["found"]
    found = []
    try:
        r = requests.get(GEMINI_BASE + "/models", params={"pageSize": 200},
                         headers={"x-goog-api-key": key}, timeout=30)
        if r.status_code == 200:
            skip = ("image", "tts", "audio", "live", "embedding", "robotics",
                    "computer", "preview", "exp", "thinking", "vision",
                    "learnlm", "gemma", "aqa", "omni", "translate",
                    "transcribe", "banana")
            for m in r.json().get("models", []):
                name = m.get("name", "").replace("models/", "")
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                if "gemini" not in name or "flash" not in name:
                    continue
                if any(w in name for w in skip):
                    continue
                found.append(name)
    except requests.RequestException as e:
        log("warn: model list error %s" % type(e).__name__)
    found.sort(key=lambda n: (-_version_key(n), "lite" in n, n))
    _model_cache["found"] = found
    return found


def _call_model(key, model, prompt, json_mode, temperature):
    url = GEMINI_BASE + "/models/" + model + ":generateContent"
    configs = [{"temperature": temperature, "maxOutputTokens": 16384}, {}]
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    err = ""
    for gen_conf in configs:
        gen_conf = dict(gen_conf)
        if json_mode:
            gen_conf["responseMimeType"] = "application/json"
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        if gen_conf:
            body["generationConfig"] = gen_conf
        bad_request = False
        for attempt in range(4):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=240)
            except requests.RequestException as e:
                err = type(e).__name__
                time.sleep(15)
                continue
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts).strip()
                except (KeyError, IndexError, ValueError):
                    return "", "empty response"
                return (text, "") if text else ("", "blank text")
            err = "%s %s %s" % (model, r.status_code, r.text[:200])
            if r.status_code in (429, 500, 503):
                time.sleep(20 * (attempt + 1))
                continue
            bad_request = (r.status_code == 400)
            break
        if not bad_request:
            break
    return "", err


def gemini(prompt, json_mode=False, temperature=0.7):
    key = os.environ["GEMINI_API_KEY"]
    first = [os.environ.get("GEMINI_MODEL", "").strip()] + DEFAULT_MODELS
    queue = [m for m in first if m]
    last_error, discovered, tried = "", False, []
    while True:
        queue = [m for m in queue
                 if m not in tried and m not in _model_cache["dead"]]
        if not queue:
            if discovered:
                break
            discovered = True
            queue = discover_models(key)
            continue
        model = queue.pop(0)
        tried.append(model)
        text, err = _call_model(key, model, prompt, json_mode, temperature)
        if text:
            return text
        last_error = err
        log("warn: model %s failed (%s)" % (model, err[:120]))
        _model_cache["dead"].add(model)
    raise RuntimeError("Gemini failed on all models: " + last_error)


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


# ------------------------------------------------------------ note articles
def strip_html(s):
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s or "")
    s = re.sub(r"(?i)<br\s*/?>|</p>|</h[1-6]>|</li>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\u3000]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def fetch_articles():
    """note公開RSSから記事一覧を取る（新しい順）。"""
    url = "https://note.com/%s/rss" % NOTE_USER
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"content": "http://purl.org/rss/1.0/modules/content/"}
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        desc = it.findtext("description") or ""
        full = it.findtext("content:encoded", namespaces=ns) or ""
        try:
            pub = email.utils.parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            pub = None
        items.append({"title": title, "link": link, "pub": pub,
                      "text": strip_html(full or desc)})
    items.sort(key=lambda x: x["pub"] or datetime.datetime.min.replace(
        tzinfo=datetime.timezone.utc), reverse=True)
    return items


def fetch_body_fallback(link):
    """RSSの本文が短いとき、noteの非公式APIで本文を補う（失敗しても続行）。"""
    m = re.search(r"/n/(n[0-9a-z]+)", link)
    if not m:
        return ""
    try:
        r = requests.get("https://note.com/api/v3/notes/" + m.group(1),
                         timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        return strip_html((r.json().get("data") or {}).get("body") or "")
    except (requests.RequestException, ValueError):
        return ""


def load_posted():
    if os.path.exists(POSTED_FILE):
        try:
            with open(POSTED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("links", [])
            data.setdefault("history", [])
            return data
        except (ValueError, OSError):
            log("warn: posted.json unreadable, starting empty")
    return {"links": [], "history": []}


def pick_article(items, posted):
    now = datetime.datetime.now(datetime.timezone.utc)
    for a in items:
        if not a["link"] or a["link"] in posted["links"]:
            continue
        if a["pub"] and (now - a["pub"]).days > MAX_AGE_DAYS:
            continue
        return a
    return None


# --------------------------------------------------------------- script
def rule_check(texts):
    issues = []
    joined = "\n".join(texts)
    for pattern, label in BAD_PATTERNS:
        if re.search(pattern, joined):
            issues.append(label)
    return issues


def script_texts(sc):
    texts = [sc.get("caption", "")]
    for s in [sc.get("hook", {})] + list(sc.get("points", [])):
        texts += [s.get("heading", ""), s.get("narration", "")]
    return texts


def make_script(title, text):
    prompt = (
        "あなたは「暮らしラボ」のInstagramリール台本の作成者です。"
        "下のnote記事の内容だけを使って、30〜40秒の縦型リールの台本を"
        "JSONだけで出力してください。\n"
        "記事タイトル: " + title + "\n"
        "記事本文(無料で読める部分):\n" + text[:3000] + "\n\n"
        "ルール:\n"
        "- 記事に書かれていることだけを使う。記事にない数字・実績・体験談・"
        "効果を足さない。\n"
        "- 構成: hook(気になる問いかけか結論) → points(3〜4個の要点)。"
        "最後の「続きはnoteで」は別に追加するので書かない。\n"
        "- hook: heading は20文字以内、narration は40文字以内。\n"
        "- 各point: heading は18文字以内、narration は45文字以内。"
        "話し言葉で短く。\n"
        "- narrationの合計は200文字前後にする。\n"
        "- 「完全に」「確実に」「誰でも」「劇的に」などの断定や誇張は使わず、"
        "「〜しやすくなります」と書く。\n"
        "- 洗剤・火・電気・健康に関わる話題で注意が必要なら、一言添える。\n"
        "- caption は120〜220文字。1行目は引きのある一文。記事の要点を2〜3文。"
        "最後は「" + CAPTION_TAIL + "」で終える。ハッシュタグは入れない。\n"
        "- hashtags は日本語、#なし、3〜5個。\n"
        "JSON形式:\n"
        "{\"hook\": {\"heading\": \"\", \"narration\": \"\"}, "
        "\"points\": [{\"heading\": \"\", \"narration\": \"\"}], "
        "\"caption\": \"\", \"hashtags\": [\"\"]}"
    )
    return parse_json(gemini(prompt, json_mode=True, temperature=0.7))


def proofread_script(title, text, sc, issues):
    extra = ""
    if issues:
        extra = ("機械チェックで見つかった問題（必ず直す）:\n"
                 + "\n".join("- " + i for i in issues) + "\n")
    prompt = (
        "あなたは厳しい校閲者です。下のリール台本(JSON)を、元の記事と照らして"
        "点検し、直したJSONだけを出力してください（解説は不要）。\n"
        "点検項目:\n"
        "1. 元の記事にない数字・実績・体験談・効果が入っていないか（あれば削除）。\n"
        "2. 「完全に」「確実に」「誰でも」「劇的」などの断定・誇張がないか"
        "（「〜しやすくなります」にやわらげる）。\n"
        "3. 見出しとナレーションが食い違っていないか。\n"
        "4. 誤字脱字、不自然な日本語、聞き取りにくい言い回し。\n"
        "5. 安全に関わる話題で、必要な注意が抜けていないか。\n"
        "6. 文字数: hookのheadingは20文字以内、各pointのheadingは18文字以内、"
        "narrationは45文字以内、hashtagsは5個以内。\n"
        + extra +
        "形式は入力と同じJSON（hook, points, caption, hashtags）にする。\n\n"
        "【元の記事】タイトル: " + title + "\n" + text[:3000] + "\n\n"
        "【台本JSON】\n" + json.dumps(sc, ensure_ascii=False)
    )
    return parse_json(gemini(prompt, json_mode=True, temperature=0.3))


def clean_script(sc):
    """形を整える（長すぎる文字を詰める、タグを5個までにする）。"""
    def one(s, hmax, nmax):
        return {"heading": re.sub(r"\s+", " ", str(s.get("heading", ""))).strip()[:hmax],
                "narration": re.sub(r"\s+", "", str(s.get("narration", "")))[:nmax]}
    hook = one(sc.get("hook", {}), 24, 50)
    points = [one(p, 22, 55) for p in sc.get("points", [])][:5]
    points = [p for p in points if p["heading"] and p["narration"]]
    tags = [str(t).lstrip("#＃").strip() for t in sc.get("hashtags", [])]
    tags = [t for t in tags if t][:5]
    caption = str(sc.get("caption", "")).strip()
    if CAPTION_TAIL not in caption:
        caption = caption.rstrip() + "\n" + CAPTION_TAIL
    return {"hook": hook, "points": points, "caption": caption, "hashtags": tags}


def build_script(title, text):
    sc = clean_script(make_script(title, text))
    issues = rule_check(script_texts(sc))
    for attempt in range(2):
        try:
            sc = clean_script(proofread_script(title, text, sc, issues))
        except Exception as e:
            log("warn: proofread failed: %s" % e)
            break
        issues = rule_check(script_texts(sc))
        if not issues:
            break
    if issues:
        raise RuntimeError("台本の校正が不合格: " + " / ".join(issues))
    if len(sc["points"]) < 2 or not sc["hook"]["heading"]:
        raise RuntimeError("台本の中身が足りません")
    log("script OK: hook=%s / points=%d" % (sc["hook"]["heading"], len(sc["points"])))
    return sc


# ------------------------------------------------------------- rendering
NO_START = set("、。，．」』）！？!?・ー〜…,.)")


def find_font(bold):
    prefs = ([("notoserifcjk", "bold"), ("notosanscjk", "bold"),
              ("notoserifcjk", ""), ("notosanscjk", ""), ("cjk", "")] if bold
             else [("notoserifcjk", "regular"), ("notoserifcjk", ""),
                   ("notosanscjk", "regular"), ("notosanscjk", ""),
                   ("cjk", "")])
    files = []
    for base in ("/usr/share/fonts", "/usr/local/share/fonts"):
        for root, _, names in os.walk(base):
            files += [os.path.join(root, n) for n in names
                      if n.lower().endswith((".ttc", ".otf", ".ttf"))]
    for a, b in prefs:
        for p in files:
            name = os.path.basename(p).lower()
            if a in name and b in name and "italic" not in name:
                return p
    raise RuntimeError("日本語フォントが見つかりません（fonts-noto-cjkを入れてください）")


def make_background(seed):
    from PIL import Image, ImageDraw, ImageFilter
    palettes = [((238, 228, 212), (214, 200, 180)),
                ((228, 234, 222), (200, 212, 196)),
                ((232, 235, 237), (208, 216, 220)),
                ((242, 230, 225), (222, 206, 200))]
    top, bottom = palettes[seed % len(palettes)]
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line((0, y, W, y), fill=tuple(int(top[i] + (bottom[i] - top[i]) * t)
                                        for i in range(3)))
    rnd = random.Random(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for _ in range(5):
        r = rnd.randint(160, 340)
        x, y = rnd.randint(0, W), rnd.randint(0, H)
        ld.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 90))
    layer = layer.filter(ImageFilter.GaussianBlur(60))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def wrap_text(draw, text, font, max_w):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
        elif draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        elif ch in NO_START and cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def split_kicker(title):
    m = re.match(r"^【(.+?)】\s*(.*)$", title)
    return (m.group(1), m.group(2)) if m else ("", title)


def render_slide(heading, body, kicker, idx, total, seed, out_path):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    img = make_background(seed)
    px0, py0, px1, py1 = 70, 480, W - 70, 1440
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle((px0, py0, px1, py1), radius=48, fill=255)
    img = Image.composite(img.filter(ImageFilter.GaussianBlur(16)), img, mask)
    img = Image.composite(Image.new("RGB", (W, H), (255, 255, 255)), img,
                          mask.point(lambda v: int(v * 0.66)))
    draw = ImageDraw.Draw(img)
    bold_p, reg_p = find_font(True), find_font(False)
    max_w = (px1 - px0) - 120

    def font(path, size):
        return ImageFont.truetype(path, size, index=0)

    # 見出し：3行に収まる大きさまで縮める
    size = 92
    while size > 52:
        hf = font(bold_p, size)
        hlines = wrap_text(draw, heading, hf, max_w)
        if len(hlines) <= 3:
            break
        size -= 4
    hf = font(bold_p, size)
    hlines = wrap_text(draw, heading, hf, max_w)
    bf = font(reg_p, 48)
    blines = wrap_text(draw, body, bf, max_w) if body else []
    kf = font(bold_p, 50)

    h_line = int(size * 1.35)
    b_line = int(48 * 1.6)
    block = (len(hlines) * h_line + (60 if blines else 0)
             + len(blines) * b_line + (80 if kicker else 0))
    y = py0 + (py1 - py0 - block) // 2
    if kicker:
        t = "【" + kicker + "】"
        draw.text(((W - draw.textlength(t, font=kf)) / 2, y), t, font=kf, fill=(95, 85, 75))
        y += 80
    for ln in hlines:
        draw.text(((W - draw.textlength(ln, font=hf)) / 2, y), ln, font=hf, fill=(60, 60, 60))
        y += h_line
    if blines:
        y += 60
        for ln in blines:
            draw.text(((W - draw.textlength(ln, font=bf)) / 2, y), ln, font=bf, fill=(95, 95, 95))
            y += b_line

    # 上部ロゴ・下部の進行バー
    lf = font(bold_p, 52)
    t = "暮らしラボ"
    draw.text(((W - draw.textlength(t, font=lf)) / 2, 230), t, font=lf, fill=(110, 100, 90))
    bar_x0, bar_x1, bar_y = 160, W - 160, 1620
    draw.rounded_rectangle((bar_x0, bar_y, bar_x1, bar_y + 14), radius=7, fill=(225, 220, 212))
    fill_to = bar_x0 + int((bar_x1 - bar_x0) * (idx + 1) / total)
    draw.rounded_rectangle((bar_x0, bar_y, fill_to, bar_y + 14), radius=7, fill=(150, 135, 115))
    img.save(out_path, "PNG")


# ----------------------------------------------------------------- audio
async def _edge_tts(text, path):
    import edge_tts
    await edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(path)


def make_audio(text, path):
    """音声を作る。edge-tts -> gTTS の順に試す。全部だめなら False。"""
    for attempt in range(3):
        try:
            asyncio.run(_edge_tts(text, path))
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                return True
        except Exception as e:
            log("warn: edge-tts failed (%s)" % type(e).__name__)
            time.sleep(3)
    try:
        from gtts import gTTS
        gTTS(text, lang="ja").save(path)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            log("info: used gTTS")
            return True
    except Exception as e:
        log("warn: gTTS failed (%s)" % type(e).__name__)
    return False


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("command failed: %s\n%s" % (cmd[0], r.stderr[-600:]))
    return r.stdout


def media_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", path])
    return float(out.strip())


def build_segment(png, audio, text, out_path):
    if audio:
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i", png,
             "-i", audio, "-af", "apad=pad_dur=0.5", "-shortest",
             "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
             "-r", "30", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
             "-ac", "2", out_path])
    else:
        dur = max(4.0, len(text) * 0.17)
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i", png,
             "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "%.2f" % dur,
             "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
             "-r", "30", "-c:a", "aac", "-b:a", "128k", "-shortest", out_path])


def build_video(title, sc, out_path):
    os.makedirs(WORK_DIR, exist_ok=True)
    kicker, main_title = split_kicker(title)
    slides = [{"heading": main_title or title, "body": sc["hook"]["narration"],
               "kicker": kicker, "say": sc["hook"]["narration"]}]
    for p in sc["points"]:
        slides.append({"heading": p["heading"], "body": p["narration"],
                       "kicker": "", "say": p["narration"]})
    slides.append({"heading": CTA_HEADING, "body": "プロフィールのリンクから",
                   "kicker": "", "say": CTA_NARRATION})
    seed = int(datetime.datetime.now(JST).strftime("%Y%m%d"))
    segs, silent = [], 0
    for i, s in enumerate(slides):
        png = os.path.join(WORK_DIR, "slide_%02d.png" % i)
        render_slide(s["heading"], s["body"], s["kicker"], i, len(slides), seed, png)
        mp3 = os.path.join(WORK_DIR, "voice_%02d.mp3" % i)
        ok = make_audio(s["say"], mp3)
        if not ok:
            silent += 1
        seg = os.path.join(WORK_DIR, "seg_%02d.mp4" % i)
        build_segment(png, mp3 if ok else None, s["say"], seg)
        segs.append(seg)
    if silent:
        log("warn: %d/%d slides have no voice (TTS failed)" % (silent, len(slides)))
    lst = os.path.join(WORK_DIR, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segs:
            f.write("file '%s'\n" % os.path.abspath(s))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "128k",
         "-ar", "44100", "-movflags", "+faststart", out_path])
    dur = media_duration(out_path)
    log("video OK: %.1f sec, %.1f MB" % (dur, os.path.getsize(out_path) / 1e6))
    if dur > MAX_REEL_SEC:
        raise RuntimeError("動画が長すぎます（%.0f秒）。台本を短くしてください" % dur)
    return dur


# ---------------------------------------------------------------- prepare
def set_output(key, value):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("%s=%s\n" % (key, value))


def cmd_prepare():
    posted = load_posted()
    items = fetch_articles()
    log("RSS: %d articles" % len(items))
    art = pick_article(items, posted)
    if not art:
        log("投稿する新しい記事がありません（%d日以内の未投稿記事なし）" % MAX_AGE_DAYS)
        set_output("skip", "true")
        return 0
    log("article: " + art["title"])
    text = art["text"]
    if len(text) < 400:
        more = fetch_body_fallback(art["link"])
        if len(more) > len(text):
            text = more
    log("article text: %d chars" % len(text))
    if len(text) < 300:
        raise RuntimeError("記事の本文が取得できませんでした（%d文字）" % len(text))

    sc = build_script(art["title"], text)
    os.makedirs(SITE_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    name = "reel_%s.mp4" % datetime.datetime.now(JST).strftime("%Y%m%d_%H%M")
    build_video(art["title"], sc, os.path.join(SITE_DIR, name))

    caption = sc["caption"]
    if sc["hashtags"]:
        caption += "\n\n" + " ".join("#" + t for t in sc["hashtags"])
    meta = {"file": name, "caption": caption[:2200], "title": art["title"],
            "link": art["link"]}
    with open(os.path.join(WORK_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(WORK_DIR, "script.json"), "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=2)
    log("caption:\n" + meta["caption"])
    set_output("skip", "false")
    return 0


# ---------------------------------------------------------------- publish
def ig_url(path):
    return "https://graph.instagram.com/%s%s" % (IG_API_VERSION, path)


def ig_headers():
    return {"Authorization": "Bearer " + IG_TOKEN}


def ig_error_text(r):
    try:
        e = r.json().get("error", {})
        return "%s code=%s %s" % (r.status_code, e.get("code"), e.get("message", ""))
    except ValueError:
        return "%s %s" % (r.status_code, r.text[:200])


def refresh_token_best_effort():
    """トークンの有効期限を延ばす（失敗しても続行。トークンは表示しない）。"""
    try:
        r = requests.get("https://graph.instagram.com/refresh_access_token",
                         params={"grant_type": "ig_refresh_token",
                                 "access_token": IG_TOKEN}, timeout=30)
        if r.status_code == 200:
            days = int(r.json().get("expires_in", 0)) // 86400
            log("info: token refreshed, valid for about %d days" % days)
        else:
            log("info: token refresh skipped (%s)" % ig_error_text(r)[:150])
    except requests.RequestException as e:
        log("info: token refresh error (%s)" % type(e).__name__)


def wait_public(url, timeout_sec=420):
    """GitHub Pagesに動画が公開されるまで待つ。"""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = requests.head(url, timeout=20, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "video" in ctype:
                log("info: video is public (%s)" % ctype)
                return True
            log("info: waiting for the video to be public (%s %s)"
                % (r.status_code, ctype))
        except requests.RequestException as e:
            log("info: waiting (%s)" % type(e).__name__)
        time.sleep(15)
    return False


def create_container(video_url, caption):
    payload = {"media_type": "REELS", "video_url": video_url,
               "caption": caption, "share_to_feed": True}
    if IG_AI_LABEL:
        payload["is_ai_generated"] = True
    r = requests.post(ig_url("/%s/media" % IG_USER_ID), headers=ig_headers(),
                      json=payload, timeout=90)
    if r.status_code != 200 and "is_ai_generated" in r.text and IG_AI_LABEL:
        log("warn: is_ai_generated rejected, retrying without it")
        payload.pop("is_ai_generated")
        r = requests.post(ig_url("/%s/media" % IG_USER_ID), headers=ig_headers(),
                          json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError("コンテナ作成に失敗: " + ig_error_text(r))
    return r.json()["id"]


def wait_container(cid, tries=24, interval=30):
    for i in range(tries):
        r = requests.get(ig_url("/" + cid), headers=ig_headers(),
                         params={"fields": "status_code,status"}, timeout=60)
        if r.status_code == 200:
            data = r.json()
            code = data.get("status_code")
            log("info: container status = %s" % code)
            if code == "FINISHED":
                return
            if code in ("ERROR", "EXPIRED"):
                raise RuntimeError("動画の処理に失敗: %s" % data.get("status", code))
        else:
            log("warn: status check %s" % ig_error_text(r)[:150])
        time.sleep(interval)
    raise RuntimeError("動画の処理が時間内に終わりませんでした")


def publish_container(cid):
    last = ""
    for attempt in range(4):
        r = requests.post(ig_url("/%s/media_publish" % IG_USER_ID),
                          headers=ig_headers(), json={"creation_id": cid},
                          timeout=90)
        if r.status_code == 200:
            return r.json()["id"]
        last = ig_error_text(r)
        log("warn: publish attempt %d failed: %s" % (attempt + 1, last[:150]))
        time.sleep(15)
    raise RuntimeError("投稿に失敗: " + last)


def cmd_publish():
    if not IG_USER_ID or not IG_TOKEN:
        raise RuntimeError("IG_USER_ID / IG_ACCESS_TOKEN が設定されていません")
    with open(os.path.join(WORK_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    base = os.environ.get("PAGES_URL", "").strip()
    if not base:
        raise RuntimeError("PAGES_URL がありません（Pagesの設定を確認してください）")
    video_url = base.rstrip("/") + "/" + meta["file"]
    log("video url: " + video_url)
    if not wait_public(video_url):
        raise RuntimeError("動画が公開されませんでした。Pagesの設定を確認してください")
    refresh_token_best_effort()
    cid = create_container(video_url, meta["caption"])
    log("container: " + cid)
    wait_container(cid)
    media_id = publish_container(cid)
    log("POSTED: instagram media id " + media_id)

    posted = load_posted()
    posted["links"].append(meta["link"])
    posted["history"].append({
        "date": datetime.datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "title": meta["title"], "link": meta["link"], "media_id": media_id})
    posted["links"] = posted["links"][-200:]
    posted["history"] = posted["history"][-200:]
    with open(POSTED_FILE, "w", encoding="utf-8") as f:
        json.dump(posted, f, ensure_ascii=False, indent=2)
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "publish"):
        print("usage: python reels_post.py prepare|publish")
        return 2
    try:
        return cmd_prepare() if sys.argv[1] == "prepare" else cmd_publish()
    except Exception as e:
        log("ERROR: %s" % e)
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
