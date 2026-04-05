import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from feature_extraction import extract_features

# THÊM 3 DÒNG NÀY VÀO ĐỂ FIX LỖI TIẾNG VIỆT
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_svm_model(dataset_path, model_save_path):
    X = [] # Chứa các vector đặc trưng
    y = [] # Chứa nhãn (tên loài cá)
    
    print("Bắt đầu đọc ảnh và trích xuất đặc trưng...")
    
    # Duyệt qua từng thư mục con trong dataset
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        
        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(class_dir):
            continue
            
        print(f"  -> Đang xử lý loài: {class_name}")
        
        # Duyệt qua từng ảnh trong thư mục
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(class_name) # Lấy tên thư mục làm nhãn luôn
    
    # Chuyển list sang mảng numpy để huấn luyện
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTổng số ảnh đã xử lý: {len(X)}")
    
    # Cắt bộ dữ liệu: 80% để Học (Train), 20% để Thi thử (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Đang huấn luyện mô hình SVM (Có thể mất vài phút)...")
    # Khởi tạo mô hình SVM. kernel='linear' hoặc 'rbf' thường cho kết quả tốt
    svm_model = SVC(kernel='linear', probability=True) 
    svm_model.fit(X_train, y_train)
    
    print("Đang đánh giá độ chính xác trên tập Test (20% ảnh chưa từng thấy)...")
    y_pred = svm_model.predict(X_test)
    
    # In ra kết quả đánh giá chi tiết
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nĐỘ CHÍNH XÁC TỔNG THỂ: {accuracy * 100:.2f}%")
    print("\nBáo cáo chi tiết từng loài:")
    print(classification_report(y_test, y_pred))
    
    # Lưu mô hình lại thành file .pkl
    joblib.dump(svm_model, model_save_path)
    print(f"\nĐã lưu mô hình thành công tại: {model_save_path}")

# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    thu_muc_dataset = r"D:\StudyUni\66.ThietKeWeb1\py\dataset_ca_ngu" 
    ten_file_luu = r"D:\StudyUni\66.ThietKeWeb1\py\svm_tuna_model.pkl"
    
    train_svm_model(thu_muc_dataset, ten_file_luu)