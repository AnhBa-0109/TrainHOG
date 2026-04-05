import os
import sys

# Cách mới an toàn hơn cho Python 3.7+: Chỉ cấu hình lại bảng mã, không ghi đè luồng
sys.stdout.reconfigure(encoding='utf-8')
def rename_dataset_images(dataset_path):
    """
    Hàm tự động đổi tên ảnh trong dataset theo định dạng: ten_loai_001.jpg
    """
    print(f"Bắt đầu xử lý thư mục: {dataset_path}")
    
    # Biến đếm tổng số ảnh đã đổi tên
    total_renamed = 0
    
    # Định dạng các đuôi file ảnh được chấp nhận
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # 1. Duyệt qua từng thư mục loài cá (Ca_ngu_mat_to, Ca_ngu_vay_dai,...)
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        
        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(class_dir):
            continue
            
        print(f"\n  -> Đang xử lý loài: {class_name}")
        
        # Biến đếm số ảnh của riêng loài này, bắt đầu từ 1
        count_species = 1
        
        # 2. Duyệt qua từng file trong thư mục loài
        # Sắp xếp tên file để đảm bảo thứ tự đổi tên là nhất quán (tùy chọn)
        sorted_files = sorted(os.listdir(class_dir))
        
        for file_name in sorted_files:
            file_path = os.path.join(class_dir, file_name)
            
            # 3. Chỉ xử lý nếu là file ảnh
            if not file_name.lower().endswith(image_extensions):
                continue
                
            # 4. Tạo tên file mới chuẩn hóa
            # Lấy đuôi file gốc (vd: .jpg)
            file_ext = os.path.splitext(file_name)[1]
            # Tạo tên mới, vd: Ca_ngu_mat_to_001.jpg, :03d giúp chèn số 0 ở đầu (001, 010, 100)
            new_file_name = f"{class_name}_{count_species:03d}{file_ext}"
            new_file_path = os.path.join(class_dir, new_file_name)
            
            # 5. Thực hiện đổi tên (An toàn: kiểm tra xem file mới có tồn tại chưa để tránh ghi đè)
            if not os.path.exists(new_file_path):
                os.rename(file_path, new_file_path)
                print(f"    - Đã đổi: '{file_name}' -> '{new_file_name}'")
                count_species += 1
                total_renamed += 1
            else:
                # Nếu xui xẻo file mới đã tồn tại, chúng ta bỏ qua để an toàn
                print(f"    [Bỏ qua] File mới '{new_file_name}' đã tồn tại!")

    print(f"\n✅ HOÀN THÀNH. Tổng số ảnh đã được đổi tên: {total_renamed}")

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Thay bằng đường dẫn tuyệt đối đến thư mục dataset của bạn
    thu_muc_dataset = r"D:\StudyUni\66.ThietKeWeb1\py\dataset_ca_ngu" 
    
    # ⚠️ CẢNH BÁO: Hãy backup thư mục dataset của bạn trước khi chạy để đảm bảo an toàn dữ liệu!
    rename_dataset_images(thu_muc_dataset)