import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import tools
import wikipedia
import google.generativeai as genai
import urllib.parse
import pandas as pd
from datetime import datetime

# ====== Inisialisasi Session State (Pindahkan ke atas) ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_city" not in st.session_state:
    st.session_state.last_city = None
if "show_clear_confirmation" not in st.session_state:
    st.session_state.show_clear_confirmation = False

# ====== Load .env ======
load_dotenv()
cohere_api_key = st.secrets["COHERE_API_KEY"]  # Ganti baris load_dotenv  # Ambil dari .env

# ====== Streamlit Config ======
st.set_page_config(page_title="Travel Chatbot", layout="centered")
st.title("🧳 Travel Assistant Agentic Chatbot")

st.markdown("""
✈️ **Asisten Perjalanan Cerdas**

Didukung oleh AI, RAG, LangChain, dan Google Maps.🌍  

Halo! Saya adalah Asisten Perjalanan AI Anda.

Saya dapat membantu Anda merencanakan perjalanan ke berbagai kota di Indonesia.

Contohnya, Anda bisa bertanya:

- Tempat wisata terkenal di Surabaya?
- transportasi buat pergi ke surabaya yang tersedia apa aja yaa?
- Aku mau pergi ke Surabaya. Tolong kasih informasi promo yang ada disana dong.

Saya juga dapat memberikan:
- Link lokasi via Google Maps
- Rekomendasi transportasi dan hotel
- Promo perjalanan yang tersedia

Masukkan nama kota untuk mulai mengeksplorasi destinasi Anda!
""")

# ====== Sidebar Input API Key ======
st.sidebar.header("🔐 API Configuration")
google_api_key = st.sidebar.text_input("🔑 Google API Key", type="password")

# ====== Petunjuk Pengambilan API Key ======
st.sidebar.markdown("---")
st.sidebar.markdown("❓ **Cara Mendapatkan API Key:**")

with st.sidebar.expander("📍 Cara mendapatkan Google API Key"):
    st.markdown("""
1. Buka [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Login dengan akun Google Anda
3. Klik tombol **"Get API Key"**
4. Salin API key dan tempel di atas

⚠️ Gunakan hanya untuk penggunaan pribadi dan edukasi.
""", unsafe_allow_html=True)

with st.sidebar.expander("📍 Cara mendapatkan Cohere API Key"):
    st.markdown("""
1. Buka [Cohere Dashboard](https://dashboard.cohere.ai/api-keys)
2. Login atau buat akun
3. Salin API Key dan simpan ke file `.env`:
```env
COHERE_API_KEY=isi_api_key_anda
""")

st.sidebar.markdown("> ⚠️ *Pastikan semua API key valid untuk menjalankan aplikasi ini.*")

# ====== Riwayat Chat di Sidebar dengan Hapus Chat ======
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Riwayat Chat")
if st.sidebar.button("Hapus Chat"):
    st.session_state.show_clear_confirmation = True

if st.session_state.show_clear_confirmation:
    st.sidebar.write("Yakin ingin menghapus?")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Ya"):
            st.session_state.chat_history = []
            st.session_state.show_clear_confirmation = False
            st.rerun()
    with col2:
        if st.sidebar.button("Tidak"):
            st.session_state.show_clear_confirmation = False
            st.rerun()

for role, msg in st.session_state.chat_history:
    if role == "User":
        st.sidebar.markdown(f"🧟‍♂️ **User:** {msg}")
    elif role == "Bot":
        st.sidebar.markdown(f"🤖 **Bot:** {msg}")
    elif role == "Table":
        st.sidebar.dataframe(msg.reset_index(drop=True))

# ====== Validasi dan Konfigurasi API ======
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    genai.configure(api_key=google_api_key)

if not (google_api_key and cohere_api_key):
    st.warning("🔐 Masukkan semua API Key di sidebar untuk mulai menggunakan aplikasi.")
    st.stop()

# ====== Inisialisasi LLM LangChain ======
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    verbose=True
)

# ====== Embeddings & Vectorstore ======
embeddings = CohereEmbeddings(
    model="embed-multilingual-light-v3.0",
    cohere_api_key=cohere_api_key  # Teruskan kunci API dari .env
)
vectorstore = FAISS.load_local("faiss_travel_assistant", embeddings, allow_dangerous_deserialization=True)

# ====== Tools LangChain ======
def wrap_tool_with_context(tool_func):
    def wrapper(input_str):
        print(f"Raw input to tool: {input_str}")  # Debugging
        if st.session_state.last_city:
            # Selalu timpa dengan last_city
            input_str = f"location: {st.session_state.last_city}"
            print(f"Modified input to tool: {input_str}")  # Debugging
        return tool_func(input_str)
    return wrapper

tool_list = [
    Tool.from_function(name="get_transport_schedule", func=wrap_tool_with_context(tools.get_transport_schedule),
                       description="Cari jadwal transportasi ke kota tertentu berdasarkan vektor dari FAISS."),
    Tool.from_function(name="get_promo", func=wrap_tool_with_context(tools.get_promo),
                       description="Lihat semua promo perjalanan yang tersedia."),
    Tool.from_function(name="get_promo_by_city", func=wrap_tool_with_context(tools.get_promo_by_city),
                       description="Lihat promo perjalanan untuk kota tertentu."),
    Tool.from_function(name="get_destination_info", func=wrap_tool_with_context(tools.get_destination_info),
                       description="Dapatkan info tempat wisata dan cuaca dari lokasi tertentu."),
    Tool.from_function(name="get_hotel_availability", func=wrap_tool_with_context(tools.get_hotel_availability),
                       description="Lihat hotel yang tersedia di lokasi tertentu."),
    Tool.from_function(name="get_translate_response", func=wrap_tool_with_context(tools.get_translate_response),
                       description="Terjemahkan teks ke bahasa yang diminta."),
    Tool.from_function(name="get_current_date", func=tools.get_current_date,
                       description="Tampilkan tanggal dan waktu saat ini."),
    Tool.from_function(name="get_recommendation_bundle", func=wrap_tool_with_context(tools.get_recommendation_bundle),
                       description="Rekomendasi kendaraan & hotel berdasarkan kota tujuan"),
    Tool.from_function(name="get_all_kendaraan_kota", func=tools.get_all_kendaraan_kota,
                       description="Tampilkan semua moda transportasi dan kota tujuannya.")
]

# ====== Memory ======
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# ====== Agent Executor ======
agent_executor = initialize_agent(
    tools=tool_list,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory
)

# ====== Wikipedia & Gemini Helper ======
def get_maps_link(place, city):
    query = urllib.parse.quote(f"{place} {city}")
    return f"https://www.google.com/maps/search/?api=1&query={query}"

def get_city_description(city):
    try:
        wikipedia.set_lang("id")  # ganti bahasa ke Indonesia
        summary = wikipedia.summary(city, sentences=3, auto_suggest=False)
        return summary
    except wikipedia.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)
        except:
            return "❗ Deskripsi kota tidak ditemukan."
    except:
        return "❗ Deskripsi kota tidak ditemukan."

def get_travel_info_gemini(city):
    prompt = f"""Berikan informasi perjalanan singkat dengan menggunakan bahasa indonesia untuk kota {city} meliputi:
1. Tiga tempat terkenal
2. Tiga makanan khas
3. Tiga mall terbaik
4. Tiga restoran rekomendasi
Jawab hanya dalam bentuk bullet point nama saja, tanpa penjelasan."""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    lines = raw_text.splitlines()
    result = ""

    # Tambahkan kalimat pembuka secara eksplisit, agar tidak diproses Gemini
    result += "**Tentu, berikut informasi perjalanan singkat di {} dalam bentuk bullet point:**\n\n".format(city.title())

    current_category = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Deteksi heading kategori
        if any(heading in line.lower() for heading in [
            "tempat terkenal", "makanan khas", "mall terbaik", "restoran rekomendasi"
        ]):
            current_category = line.strip(":")
            result += f"\n**{current_category}**\n"
        elif line.startswith(("-", "*")):
            item = line.lstrip("-* ").strip()
            maps_link = get_maps_link(item, city)
            result += f"- [{item}]({maps_link})\n"
        else:
            # baris yang bukan heading dan bukan bullet — lewati
            continue

    return result

def get_wikipedia_summary(city: str) -> str:
    try:
        wikipedia.set_lang("id")
        summary = wikipedia.summary(city, sentences=3)
        return summary
    except wikipedia.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=3)
        except:
            return "❗ Maaf, tidak ditemukan informasi spesifik dari Wikipedia."
    except:
        return "❗ Maaf, tidak ditemukan informasi dari Wikipedia."

def get_gemini_general_info(question: str, chat_history: list = None) -> str:
    model = genai.GenerativeModel('gemini-2.0-flash')
    context = "\n".join([f"{role}: {msg}" for role, msg in (chat_history or [])]) if chat_history else "No previous context"
    prompt = f"Berikan jawaban dalam bahasa Indonesia berdasarkan konteks berikut:\n{context}\nPertanyaan: {question}\nJika konteks menyebutkan kota sebelumnya (misalnya Surabaya), gunakan kota itu sebagai default kecuali pengguna menyebut kota baru."
    response = model.generate_content(prompt)
    return response.text.strip()

# ====== Pencarian Informasi Kota dari Wikipedia & Maps ======
st.subheader("🌍 Eksplorasi Kota")
city_query = st.text_input("🔍 Masukkan Nama Kota", placeholder="Contoh: Surabaya")

if city_query:
    with st.spinner("🔍 Mencari informasi kota..."):
        city_desc = get_city_description(city_query)
        travel_info = get_travel_info_gemini(city_query)
        maps_url = get_maps_link(city_query, city_query)
        st.session_state.last_city = city_query  # Simpan kota terakhir

        st.markdown(f"### 📌 {city_query.title()}")
        st.write(city_desc)

        st.markdown("#### 🌐 Google Maps:")
        st.markdown(f"[📍 Lihat di Google Maps]({maps_url})", unsafe_allow_html=True)

        st.markdown("#### 🗺️ Rekomendasi Perjalanan:")
        st.markdown(travel_info, unsafe_allow_html=True)

# ====== Tampilkan Riwayat Chat di Area Utama ======
st.markdown("---")
for role, msg in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"🧟‍♂️ **User:** {msg}")
    elif role == "Bot":
        st.markdown(f"🤖 **Bot:** {msg}")
    elif role == "Table":
        st.dataframe(msg.reset_index(drop=True))
st.markdown("---")

# ====== Area Chat dan Footer ======
chat_container = st.container()
with chat_container:
    user_input = st.chat_input("💬 Assalamualaikum Admin!")
    st.markdown('Copyright © 2025 [mza_offc](https://sociabuzz.com/mza_offc/tribe). Created by a Muslim from Indonesia for Travel Assistant with ❤️', unsafe_allow_html=True)

# ====== Deteksi Otomatis Promo & Bundle ======
def detect_city_for_promo(text: str):
    for kota in tools.df_promo["location"].unique():
        if kota.lower() in text.lower():
            return kota
    return st.session_state.last_city  # Gunakan kota terakhir jika tidak ada yang baru

def detect_recommendation_bundle(text: str):
    text = text.lower()
    if "rekomendasi" in text and ("hotel" in text or "penginapan" in text) and ("kendaraan" in text or "transport" in text):
        for kota in tools.df_hotel["location"].unique():
            if kota.lower() in text:
                return kota
    return st.session_state.last_city  # Gunakan kota terakhir jika tidak ada yang baru

def handle_transport_query(user_input, chat_history):
    kota_dicari = None
    mode_dicari = None

    # Cari kota berdasarkan riwayat atau input
    for role, msg in chat_history:
        if role == "User" and any(kota.lower() in msg.lower() for kota in tools.df_transport["destination"].unique()):
            kota_dicari = next(kota for kota in tools.df_transport["destination"].unique() if kota.lower() in msg.lower())
            break
    if not kota_dicari:
        for kota in tools.df_transport["destination"].unique():
            if kota.lower() in user_input.lower():
                kota_dicari = kota
                break
    if not kota_dicari and st.session_state.last_city in tools.df_transport["destination"].unique():
        kota_dicari = st.session_state.last_city

    # Cari jenis kendaraan dari kolom 'mode'
    for mode in tools.df_transport["mode"].unique():
        if mode.lower() in user_input.lower():
            mode_dicari = mode
            break

    if kota_dicari:
        df = tools.df_transport.copy()
        df_filtered = df[df["destination"].str.lower() == kota_dicari.lower()]

        # Filter juga jika mode kendaraan disebut
        if mode_dicari:
            df_filtered = df_filtered[df_filtered["mode"].str.lower() == mode_dicari.lower()]

        if not df_filtered.empty:
            st.markdown(f"### 🚍 Informasi Transportasi ke {kota_dicari.title()}")
            st.dataframe(df_filtered.reset_index(drop=True))
            st.session_state.chat_history.append(("Bot", f"Berikut info {mode_dicari or 'transportasi'} menuju {kota_dicari.title()}"))
            st.session_state.chat_history.append(("Table", df_filtered))
            return True  # tanda bahwa pertanyaan sudah ditangani

    return False  # tidak ada yang ditampilkan

# ====== Proses Utama ======
if user_input:
    kota_promo = detect_city_for_promo(user_input)
    kota_bundle = detect_recommendation_bundle(user_input)

    with st.spinner("⏳ Menjawab..."):
        try:
            st.session_state.chat_history.append(("User", user_input))

            if kota_bundle:
                # Pastikan kota_bundle ada, gunakan last_city sebagai fallback
                city_to_use = kota_bundle or st.session_state.last_city
                if not city_to_use:
                    st.session_state.chat_history.append(("Bot", "Kota tujuan belum ditentukan. Silakan masukkan kota terlebih dahulu."))
                else:
                    result = tools.get_recommendation_bundle(city_to_use)
                    deskripsi = f"**Rekomendasi untuk kota {result['location']}:**"
                    st.session_state.chat_history.append(("Bot", deskripsi))

                    if result["transport"] is not None:
                        st.session_state.chat_history.append(("Table", result["transport"]))
                    if result["hotel"] is not None:
                        st.session_state.chat_history.append(("Table", result["hotel"]))

            elif kota_promo and "promo" in user_input.lower():
                promo_df = tools.df_promo[tools.df_promo["location"].str.contains(kota_promo, case=False)]
                if not promo_df.empty:
                    pesan = f"🏱 Berikut promo yang tersedia untuk kota **{kota_promo.title()}**:"
                    st.session_state.chat_history.append(("Bot", pesan))
                    st.session_state.chat_history.append(("Table", promo_df))
                else:
                    pesan = f"🏱 Tidak ada promo tersedia untuk kota **{kota_promo.title()}**."
                    st.session_state.chat_history.append(("Bot", pesan))
                    
            elif any(kata in user_input.lower() for kata in ["transportasi", "kendaraan", "harga", "tiket", "biaya"]):
                if handle_transport_query(user_input, st.session_state.chat_history):
                    pass  # Sudah ditangani, tidak lanjut ke Gemini/Wikipedia
                else:
                    st.session_state.chat_history.append(("Bot", "⚠️ Tidak ditemukan data transportasi yang cocok."))

            else:
                # Tambahkan konteks kota ke user_input jika ada last_city
                enriched_input = user_input
                if st.session_state.last_city:
                    enriched_input = f"Untuk {st.session_state.last_city}, {user_input}"
                
                try:
                    print(f"Running agent with input: {enriched_input}")  # Debugging
                    response = agent_executor.run(enriched_input)
                    if not response or "I don't know" in response.lower():
                        raise ValueError("Jawaban tidak relevan, pakai fallback")
                except Exception as e:
                    # Fallback ke Gemini dengan konteks riwayat
                    gemini_response = get_gemini_general_info(enriched_input, st.session_state.chat_history)
                    response = gemini_response

                st.session_state.chat_history.append(("Bot", response))

        except Exception as e:
            st.session_state.chat_history.append(("Bot", f"🚨 Kesalahan: {e}"))

    # Memaksa rerender untuk memperbarui riwayat chat
    st.rerun()
