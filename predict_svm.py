import os
import sys
import cv2
import numpy as np
from skimage.feature import hog
import joblib

# Ép Terminal Windows dùng UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# BẮT BUỘC PHẢI GIỮ LẠI HÀM TRÍCH XUẤT ĐẶC TRƯNG Y HỆT LÚC HUẤN LUYỆN
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc được ảnh, kiểm tra lại đường dẫn!")
        
    img_resized = cv2.resize(img, (128, 128))
    
    # Đặc trưng HOG
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    
    # Đặc trưng Màu sắc
    hist_b = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [32], [0, 256]).flatten()
    color_features = np.hstack([hist_b, hist_g, hist_r])
    
    return np.hstack([hog_features, color_features])

def predict_tuna_svm(image_path, model_path):
    print(f"Đang tải mô hình SVM từ: {model_path} ...")
    try:
        # Load mô hình SVM
        svm_model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Lỗi tải mô hình: {e}")
        return

    print("Đang trích xuất đặc trưng bức ảnh...")
    try:
        # Trích xuất 1 vector đặc trưng của ảnh đầu vào
        features = extract_features(image_path)
        
        # Reshape lại thành mảng 2D (1 hàng, n cột) vì SVM yêu cầu form này
        features = features.reshape(1, -1)
        
        # Dự đoán
        predicted_label = svm_model.predict(features)[0]

        # In kết quả
        print("\n" + "="*40)
        print("📊 KẾT QUẢ TỪ BĂNG CHUYỀN:")
        print(f"   🐟 Nhãn dự đoán: {predicted_label}")
        print("="*40 + "\n")
        
        return predicted_label

    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Đường dẫn tới file model bạn vừa train xong
    duong_dan_model = r"C:\Users\khanh\Desktop\py\svm_tuna_model.pkl"
    
    # Lấy 1 tấm ảnh test bất kỳ để thử (đổi tên file cho đúng)
    duong_dan_anh_test = r"C:\Users\khanh\Desktop\py\images\mt1.png" 
    
    if not os.path.exists(duong_dan_anh_test):
        print(f"⚠️ Không tìm thấy ảnh tại: {duong_dan_anh_test}")
    else:
        predict_tuna_svm(duong_dan_anh_test, duong_dan_model)