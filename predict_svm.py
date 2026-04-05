import os
import sys

import joblib

from feature_extraction import extract_features

# Ép Terminal Windows dùng UTF-8
sys.stdout.reconfigure(encoding="utf-8")


def predict_tuna_svm(image_path, model_path):
    """Dự đoán nhãn bằng SVM (cần file .pkl). Muốn chỉ xem HOG/histogram, dùng show_features.py."""
    print(f"Đang tải mô hình SVM từ: {model_path} ...")
    try:
        svm_model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {e}")
        return None

    print("Đang trích xuất đặc trưng và dự đoán...")
    try:
        features = extract_features(image_path)
        if features is None:
            raise ValueError("Không thể đọc được ảnh, kiểm tra lại đường dẫn!")

        features = features.reshape(1, -1)
        predicted_label = svm_model.predict(features)[0]

        print("\n" + "=" * 40)
        print("KẾT QUẢ DỰ ĐOÁN (SVM)")
        print(f"   Nhãn: {predicted_label}")
        print("=" * 40 + "\n")

        return predicted_label

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


# ==========================================
# CÁCH SỬ DỤNG — dự đoán loài (cần model đã train)
# ==========================================
if __name__ == "__main__":
    duong_dan_model = r"C:\Users\khanh\Desktop\py\svm_tuna_model.pkl"
    duong_dan_anh_test = r"C:\Users\khanh\Desktop\py\images\mt1.png"

    if not os.path.exists(duong_dan_anh_test):
        print(f"⚠️ Không tìm thấy ảnh tại: {duong_dan_anh_test}")
    else:
        predict_tuna_svm(duong_dan_anh_test, duong_dan_model)
