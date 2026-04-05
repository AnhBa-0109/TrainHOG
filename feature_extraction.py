import cv2
import numpy as np
from skimage.feature import hog


def _preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.resize(img, (128, 128))


HOG_KWARGS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def _hog_and_color_from_resized(img_resized, return_hog_vis=False):
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    if return_hog_vis:
        hog_features, hog_image = hog(
            gray_img, **HOG_KWARGS, visualize=True
        )
    else:
        hog_features = hog(gray_img, **HOG_KWARGS, visualize=False)
        hog_image = None
    hist_b = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [32], [0, 256]).flatten()
    color_features = np.hstack([hist_b, hist_g, hist_r])
    return hog_features, color_features, hog_image


def extract_hog_features(image_path):
    """Chỉ vector HOG (hình dáng)."""
    img = _preprocess_image(image_path)
    if img is None:
        return None
    hog_f, _, _ = _hog_and_color_from_resized(img)
    return hog_f


def extract_color_histogram_features(image_path):
    """Chỉ histogram màu B–G–R (96 chiều)."""
    img = _preprocess_image(image_path)
    if img is None:
        return None
    _, color_f, _ = _hog_and_color_from_resized(img)
    return color_f


def extract_features_separate(image_path):
    """
    Trả về (hog_vector, color_vector). Đọc ảnh một lần.
    Nếu không đọc được ảnh: (None, None).
    """
    img = _preprocess_image(image_path)
    if img is None:
        return None, None
    hog_f, color_f, _ = _hog_and_color_from_resized(img)
    return hog_f, color_f


def extract_features_for_display(image_path):
    """
    Giống extract_features_separate nhưng thêm ảnh HOG (skimage) và ảnh xám 128×128 để hiển thị.
    Trả về dict hoặc None nếu không đọc được ảnh.
    """
    img = _preprocess_image(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_f, color_f, hog_vis = _hog_and_color_from_resized(img, return_hog_vis=True)
    return {
        "hog": hog_f,
        "color_histogram": color_f,
        "hog_visualization": hog_vis,
        "gray_128": gray,
    }


def extract_features(image_path):
    """Vector ghép [HOG | color] — dùng cho huấn luyện / SVM (giữ tương thích cũ)."""
    hog_f, color_f = extract_features_separate(image_path)
    if hog_f is None:
        return None
    return np.hstack([hog_f, color_f])
