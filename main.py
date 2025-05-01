import os, random, datetime, tempfile, pickle, librosa, numpy as np, tensorflow as tf
import soundfile as sf, resampy, asyncpg
from dotenv import load_dotenv
from keras.layers import TFSMLayer
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters


load_dotenv()
BOT_TOKEN   = "BOT_TOKEN"
DB_DSN      = "postgresql://postgres:postgres@localhost:5432/postgres"
pool: asyncpg.Pool | None = None

def preprocess_text(txt: str) -> str:
    import re, string
    txt = txt.lower()
    txt = re.sub(f"[{re.escape(string.punctuation)}]", "", txt)
    return re.sub(r"\s+", " ", txt).strip()

def noise(data): return data + 0.035*np.random.uniform()*np.amax(data)*np.random.normal(size=data.shape[0])
def pitch(data, sr, n=2): return librosa.effects.pitch_shift(data, sr=sr, n_steps=n)

def extract_features(data, sr, frame=2048, hop=512):
    stft = np.abs(librosa.stft(data, n_fft=frame, hop_length=hop))
    zcr  = librosa.feature.zero_crossing_rate(y=data, frame_length=frame, hop_length=hop).squeeze()
    rmse = librosa.feature.rms(S=stft).squeeze()
    mel  = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=frame, hop_length=hop)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=13).T.ravel()
    return np.hstack([zcr, rmse, mfcc])

def augment_and_extract(data, sr):
    aug = [data, noise(data), pitch(data, sr, 2), pitch(noise(data), sr, 2)]
    return np.vstack([extract_features(a, sr) for a in aug])

advice_dict = {
    "joy": [
        "Great that you’re feeling joy! Capture the moment—take a photo or a quick note.",
        "Reward yourself with a small treat; reinforce the link between effort and success.",
        "Share your achievement with someone close—shared joy grows stronger."
    ],
    "sadness": [
        "It's okay to feel sad. Allow it and do something soothing like a short walk.",
        "Write down one thing you’re grateful for; it can restore balance.",
        "Warm tea, calm music, or a favorite movie can comfort you when sadness returns."
    ],
    "anger": [
        "Pause for ninety seconds, breathing deeply—the chemistry of anger subsides.",
        "Write what crossed your boundaries and craft a constructive response.",
        "A brisk five‑minute walk helps release tension and clear your mind."
    ],
    "fear": [
        "Focus on what you can control and note the next small step.",
        "Try the 5‑4‑3‑2‑1 grounding technique to calm anxiety.",
        "Write the scary thought down and define the most realistic outcome."
    ],
    "surprise": [
        "Take three slow breaths to assess the unexpected situation calmly.",
        "List exactly what changed; seeing facts helps decision‑making.",
        "Identify the one urgent item and postpone the rest to regain control."
    ],
}
emotion_labels = ["sadness", "surprise", "fear", "anger", "joy"]

text_model  = TFSMLayer("model_bigru", call_endpoint="serving_default")
audio_model = tf.keras.models.load_model("emotion_recognition_model.h5")
with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)

async def init_db():
    global pool
    pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=1, max_size=5)
    async with pool.acquire() as con:
        await con.execute("""CREATE TABLE IF NOT EXISTS emotionslog(
                                user_id BIGINT,
                                ts      TIMESTAMPTZ,
                                emotion TEXT
                            );""")

async def log_emotion(uid: int, emotion: str):
    ts = datetime.datetime.utcnow()
    async with pool.acquire() as con:
        await con.execute("INSERT INTO emotionslog VALUES ($1,$2,$3)", uid, ts, emotion)

async def fetch_today(uid: int):
    async with pool.acquire() as con:
        return await con.fetch("""SELECT emotion, COUNT(*) c
                                  FROM emotionslog
                                  WHERE user_id=$1 AND ts::date=CURRENT_DATE
                                  GROUP BY emotion""", uid)

async def fetch_week(uid: int):
    async with pool.acquire() as con:
        return await con.fetch("""SELECT emotion, COUNT(*) c
                                  FROM emotionslog
                                  WHERE user_id=$1
                                    AND ts >= CURRENT_DATE - INTERVAL '6 days'
                                  GROUP BY emotion""", uid)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = preprocess_text(update.message.text)
    out = text_model(tf.constant([[txt]]))
    if isinstance(out, dict): out = next(iter(out.values()))
    emotion = emotion_labels[int(np.argmax(out[0].numpy()))]

    await log_emotion(update.effective_user.id, emotion)
    await update.message.reply_text(
        f"*Emotion:* `{emotion}`\n\n*Advice:* _{random.choice(advice_dict[emotion])}_",
        parse_mode="Markdown"
    )

async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vf = await update.message.voice.get_file()
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f_ogg:
        await vf.download_to_drive(custom_path=f_ogg.name)
        data, sr = sf.read(f_ogg.name)
    if sr != 22050: data = resampy.resample(data, sr, 22050); sr = 22050

    feats = augment_and_extract(data, sr)
    exp = scaler.mean_.shape[0]
    feats = feats[:, :exp] if feats.shape[1] > exp else np.pad(feats, ((0,0),(0,exp-feats.shape[1])), "constant")
    feats = scaler.transform(feats).reshape(feats.shape[0], feats.shape[1], 1)

    probs = audio_model.predict(feats, verbose=0).mean(axis=0)
    emotion = emotion_labels[int(np.argmax(probs))]

    await log_emotion(update.effective_user.id, emotion)
    await update.message.reply_text(
        f"*Emotion:* `{emotion}`\n\n*Advice:* _{random.choice(advice_dict[emotion])}_",
        parse_mode="Markdown"
    )

async def today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await fetch_today(update.effective_user.id)
    if not rows:
        await update.message.reply_text("No emotions logged today yet.")
        return
    summary = "\n".join(f"{r['emotion']}: {r['c']}" for r in rows)
    await update.message.reply_text(f"*Today's emotions*\n{summary}", parse_mode="Markdown")

async def week_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = await fetch_week(update.effective_user.id)
    if not rows:
        await update.message.reply_text("No data for the last 7 days.")
        return
    top = max(rows, key=lambda r: r['c'])['emotion']
    summary = "\n".join(f"{r['emotion']}: {r['c']}" for r in rows)
    advice = random.choice(advice_dict[top])
    await update.message.reply_text(
        f"*7‑day summary*\n{summary}\n\n*Most frequent:* `{top}`"
        f"\n\n*Advice:* _{advice}_", parse_mode="Markdown"
    )

async def init_db_hook(app):
    await init_db()

def main():
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(init_db_hook)
        .build()
    )

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.add_handler(CommandHandler("today", today_cmd))
    app.add_handler(CommandHandler("week", week_cmd))

    print("Emotion bot is running …")
    app.run_polling()

if __name__ == "__main__":
    main()