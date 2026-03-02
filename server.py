from flask import Flask, request, jsonify
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# ==================================================
# LOAD MODEL XGBOOST
# ==================================================
model = joblib.load("xgboost_anemia_noninvasive.pkl")

# ==================================================
# FIREBASE INITIALIZATION
# ==================================================
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://alat-deteksi-anemia-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference("data_anemia")

# ==================================================
# FLASK APP
# ==================================================
app = Flask(__name__)

# ==================================================
# FUNGSI KLASIFIKASI ANEMIA (WHO)
# ==================================================
def klasifikasi_anemia(hb, kategori):
    """
    kategori:
    0 = perempuan tidak hamil
    1 = perempuan hamil
    2 = laki-laki
    """

    if kategori == 0:  # perempuan tidak hamil
        if hb < 8.0:
            return "Berat"
        elif hb < 11.0:
            return "Sedang"
        elif hb < 12.0:
            return "Ringan"
        else:
            return "Normal"

    elif kategori == 1:  # perempuan hamil
        if hb < 7.0:
            return "Berat"
        elif hb < 10.0:
            return "Sedang"
        elif hb < 11.0:
            return "Ringan"
        else:
            return "Normal"

    else:  # laki-laki
        if hb < 8.0:
            return "Berat"
        elif hb < 11.0:
            return "Sedang"
        elif hb < 13.0:
            return "Ringan"
        else:
            return "Normal"

# ==================================================
# ENDPOINT PREDIKSI (DIPANGGIL ESP32)
# ==================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # ------------------------------
        # INPUT DARI ESP32
        # ------------------------------
        red = float(data["red"])
        ir = float(data["ir"])
        kategori = int(data["kategori"])  # 0 / 1 / 2

        # ------------------------------
        # PREDIKSI XGBOOST
        # ------------------------------
        features = np.array([[red, ir]])
        hb = model.predict(features)[0]
        hb = round(float(hb), 2)

        # ------------------------------
        # KLASIFIKASI ANEMIA
        # ------------------------------
        status = klasifikasi_anemia(hb, kategori)

        # ------------------------------
        # DATA YANG DISIMPAN
        # ------------------------------
        result = {
            "red": red,
            "ir": ir,
            "hemoglobin": hb,
            "kategori": kategori,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

        # Simpan ke Firebase
        ref.push(result)

        # ------------------------------
        # RESPONSE KE ESP32
        # ------------------------------
        return jsonify({
            "hb": hb,
            "status": status
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# ==================================================
# HOME ENDPOINT
# ==================================================
@app.route("/", methods=["GET"])
def home():
    return "BACKEND DETEKSI ANEMIA AKTIF"

# ==================================================
# RUN SERVER
# ==================================================
if __name__ == "__main__":
    app.run()
