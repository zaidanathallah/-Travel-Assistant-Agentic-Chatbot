from datetime import datetime
import pandas as pd
import json
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# ====== Load datasets ======
TRANSPORT_PATH = "D:\zaidan\Project Akhir\dataset\Transport_schedule.csv"
PROMO_PATH = "D:\zaidan\Project Akhir\dataset\promo_travel.csv"
DESTINATION_PATH = "D:\zaidan\Project Akhir\dataset\destination_info.csv"
HOTEL_PATH = "D:\zaidan\Project Akhir\dataset\hotel_availability.csv"

df_transport = pd.read_csv(TRANSPORT_PATH)
df_promo = pd.read_csv(PROMO_PATH)
df_destination = pd.read_csv(DESTINATION_PATH)
df_hotel = pd.read_csv(HOTEL_PATH)

# Load VectorDB (gunakan embeddings dari app.py)
# Catatan: Jangan inisialisasi embeddings di sini, gunakan yang dari app.py
# vectorstore akan diimpor dari app.py yang sudah dikonfigurasi

# Di bagian atas tools.py, setelah memuat dataset
print("Hotel Data:", df_hotel)  # Tambahkan ini untuk memeriksa isi df_hotel

def extract_args(input_str):
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        # Coba ekstrak location dari string biasa
        input_str = input_str.lower().strip()
        if "location:" in input_str:
            location_part = input_str.split("location:")[1].strip()
            for city in df_hotel["location"].unique():
                if city.lower() in location_part:
                    print(f"Extracted location: {city}")  # Debugging
                    return {"location": city}
        return {"input": input_str}

# ========== TOOLS ==========
def get_transport_schedule(input_str: str) -> str:
    args = extract_args(input_str)
    destination = args.get("destination", args.get("input", input_str))
    # Cari dalam VectorDB (diasumsikan diimpor dari app.py)
    global vectorstore  # Gunakan vectorstore dari app.py
    query = f"transportasi ke {destination}"
    results = vectorstore.similarity_search(query, k=5)  # Ambil 5 hasil teratas
    if results:
        transport_data = [doc.page_content for doc in results]
        return "\n".join(transport_data) if transport_data else f"🚫 Tidak ada jadwal ke **{destination}**."
    return f"🚫 Tidak ada jadwal ke **{destination}**."

def get_promo(input_str: str = "") -> str:
    return df_promo.to_string(index=False)

def get_promo_by_city(input_str: str) -> str:
    args = extract_args(input_str)
    city = args.get("city", args.get("input", input_str))
    match = df_promo[df_promo['location'].str.contains(city, case=False)]
    if not match.empty:
        return match.to_string(index=False)
    else:
        return f"🎁 Tidak ada promo tersedia untuk kota **{city}**."

def get_destination_info(input_str: str) -> str:
    args = extract_args(input_str)
    location = args.get("location", args.get("input", input_str))
    match = df_destination[df_destination['location'].str.contains(location, case=False)]
    if not match.empty:
        result = match.to_string(index=False)
        # Tambahkan prediksi cuaca sederhana berdasarkan waktu
        current_time = datetime.now().hour
        weather_note = "Cuaca mendukung untuk bepergian malam." if 18 <= current_time <= 23 else "Cuaca mungkin tidak ideal untuk bepergian malam, pertimbangkan waktu lain."
        return f"{result}\n\n📅 Catatan Cuaca: {weather_note}"
    else:
        return f"📍 Tidak ada informasi tentang **{location}**."

def get_hotel_availability(input_str: str) -> str:
    args = extract_args(input_str)
    location = args.get("location", args.get("input", input_str))
    match = df_hotel[df_hotel['location'].str.contains(location, case=False)]
    print(f"Checking hotel availability for location: {location}")  # Debugging
    return match.to_string(index=False) if not match.empty else f"🏨 Tidak ada hotel tersedia di **{location}**."

def get_translate_response(input_str: str, target_lang="id") -> str:
    args = extract_args(input_str)
    text = args.get("text", args.get("input", input_str))
    lang = args.get("lang", target_lang)
    if lang.lower() == "en":
        return f"(EN) {text}"
    elif lang.lower() == "id":
        return f"(ID) {text}"
    else:
        return f"(Translated [{lang}]) {text}"

def get_current_date(input_str: str = "") -> str:
    now = datetime.now()
    return now.strftime("📅 %A, %d %B %Y %H:%M")

def get_recommendation_bundle(input_str: str) -> dict:
    args = extract_args(input_str)
    location = args.get("location", args.get("input", input_str))
    transport_df = df_transport[df_transport["destination"].str.contains(location, case=False)]
    hotel_df = df_hotel[df_hotel["location"].str.contains(location, case=False)]
    return {
        "location": location.title(),
        "transport": transport_df if not transport_df.empty else None,
        "hotel": hotel_df if not hotel_df.empty else None
    }

def get_all_kendaraan_kota(input_str: str = "") -> str:
    if "mode" not in df_transport.columns or "destination" not in df_transport.columns:
        return "⚠️ Dataset transport tidak memiliki kolom 'mode' dan 'destination'."
    grouped = df_transport.groupby("mode")["destination"].unique()
    result = "📋 **Daftar semua kendaraan dan kota tujuannya:**\n"
    for mode, destinations in grouped.items():
        kota_list = ", ".join(sorted(destinations))
        result += f"- **{mode.title()}**: {kota_list}\n"
    return result.strip()