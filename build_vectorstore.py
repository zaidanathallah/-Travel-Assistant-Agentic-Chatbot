import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import getpass

# Load .env (agar COHERE_API_KEY bisa dipanggil dengan aman)
# ====== Load .env ======
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")  # Ambil dari .env

# Pastikan API Key tersedia
# Atur API Key
if not (cohere_api_key):
    st.warning("🔐 Masukkan COHERE_API_KEY di .env.")
    st.stop()

# Inisialisasi embeddings
embeddings = CohereEmbeddings(
    model="embed-multilingual-light-v3.0",
    cohere_api_key=cohere_api_key
)

# Fungsi bantu untuk ubah DataFrame ke list of Documents
def df_to_documents(df, label):
    docs = []
    for _, row in df.iterrows():
        content = f"{label}: " + ', '.join([f"{col}: {val}" for col, val in row.items()])
        docs.append(Document(page_content=content))
    return docs

# Load semua dataset
datasets = {
    "Transport Schedule": "D:\zaidan\Project Akhir\dataset\Transport_schedule.csv",
    "Promo Travel": "D:\zaidan\Project Akhir\dataset\promo_travel.csv",
    "Destination Info": "D:\zaidan\Project Akhir\dataset\destination_info.csv",
    "Hotel Availability": "D:\zaidan\Project Akhir\dataset\hotel_availability.csv"
}

# Gabungkan semua dokumen
all_documents = []
for label, path in datasets.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path)
    all_documents.extend(df_to_documents(df, label))

# Buat FAISS index dan simpan
vectorstore = FAISS.from_documents(all_documents, embeddings)
vectorstore.save_local("faiss_travel_assistant")
print("✅ Vectorstore berhasil dibuat dan disimpan ke folder 'faiss_travel_assistant'")
