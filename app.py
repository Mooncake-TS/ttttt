# =========================================
# Streamlit LSTM 감정분류 앱
# - 입력 데이터: Review.xlsx (Sentiment, Review)
# - 레포 구조:
#   ├─ app.py
#   ├─ Review.xlsx
#   └─ requirements.txt
# =========================================

import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# ----------------------------
# Streamlit 기본 설정
# ----------------------------
st.set_page_config(page_title="리뷰 감정 분석", layout="centered")
st.title("📊 리뷰 감정 분석 (LSTM)")
st.caption("Positive / Negative 분류")

# ----------------------------
# 1) 데이터 로드
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Review.xlsx"

if not DATA_PATH.exists():
    st.error("❌ Review.xlsx를 찾을 수 없습니다.")
    st.stop()

df = pd.read_excel(DATA_PATH)

required_cols = {"Sentiment", "Review"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ 필요한 컬럼이 없습니다: {required_cols}")
    st.stop()

df = df[["Sentiment", "Review"]].dropna()
df["Review"] = df["Review"].astype(str)

st.success(f"✅ 데이터 로드 완료 ({len(df)} rows)")
st.write(df["Sentiment"].value_counts())

# ----------------------------
# 2) 불용어 & 전처리
# ----------------------------
stopwords = set([
    "이", "가", "을", "를", "은", "는", "에", "에서", "에게",
    "의", "와", "과", "도", "로", "으로",
    "하다", "되다", "있다", "없다",
    "그", "저", "것", "수",
    "좀", "잘", "매우", "정말", "너무",
    "때문", "같다"
])

def clean_text(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(tokens)

texts = [clean_text(t) for t in df["Review"].tolist()]
labels = df["Sentiment"].tolist()

# ----------------------------
# 3) 라벨 인코딩
# ----------------------------
le = LabelEncoder()
y = le.fit_transform(labels)

st.write("🔖 라벨 매핑:", dict(zip(le.classes_, range(len(le.classes_)))))

# ----------------------------
# 4) 데이터 분할
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5) Tokenizer + Padding
# ----------------------------
VOCAB_SIZE = 15000
MAX_LEN = 120

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def encode(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

X_train_pad = encode(X_train)
X_test_pad = encode(X_test)

# ----------------------------
# 6) LSTM 모델
# ----------------------------
model = models.Sequential([
    layers.Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# 7) 학습
# ----------------------------
with st.spinner("🧠 모델 학습 중..."):
    model.fit(
        X_train_pad,
        y_train,
        epochs=6,
        batch_size=16,
        validation_split=0.1,
        verbose=0
    )

st.success("✅ 학습 완료!")

# ----------------------------
# 8) 리뷰 입력 → 예측
# ----------------------------
st.subheader("✍️ 리뷰 입력")

user_input = st.text_area(
    "리뷰를 입력하세요",
    placeholder="예: 담배 냄새가 너무 심해서 불쾌했어요"
)

if st.button("예측하기"):
    if user_input.strip() == "":
        st.warning("리뷰를 입력해주세요.")
    else:
        proc = clean_text(user_input)
        pad = encode([proc])
        probs = model.predict(pad)[0]
        idx = int(np.argmax(probs))
        label = le.inverse_transform([idx])[0]
        conf = float(np.max(probs))

        st.markdown(f"### 🧾 예측 결과: **{label}**")
        st.progress(conf)

        st.write("📊 클래스별 확률")
        for c, p in zip(le.classes_, probs):
            st.write(f"- {c}: {p:.3f}")

st.caption("✔ LSTM / Tokenizer / Padding / 불용어 제거 적용 완료")
