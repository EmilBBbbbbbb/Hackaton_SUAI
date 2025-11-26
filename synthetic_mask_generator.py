import numpy as np
import matplotlib.pyplot as plt
import cv2

# Генерация пустого снимка
def generate_empty_canvas(min_size=512, max_size=2048):
    H = np.random.randint(min_size, max_size+1)
    W = np.random.randint(min_size, max_size+1)

    canvas = np.zeros((H, W), dtype=np.uint8)
    return canvas

# Генерация прямоугольника с перспективой
def generate_perspective_corners(doc_w, doc_h, max_shift=0.3):

    tl = np.array([0, 0], dtype=np.float32)
    tr = np.array([doc_w, 0], dtype=np.float32)
    br = np.array([doc_w, doc_h], dtype=np.float32)
    bl = np.array([0, doc_h], dtype=np.float32)

    def shift_point(pt, max_shift_w, max_shift_h):
        dx = np.random.uniform(-max_shift_w, max_shift_w)
        dy = np.random.uniform(-max_shift_h, max_shift_h)
        return pt + np.array([dx, dy])

    max_dx = doc_w * max_shift
    max_dy = doc_h * max_shift

    tl = shift_point(tl, max_dx, max_dy)
    tr = shift_point(tr, max_dx, max_dy)
    br = shift_point(br, max_dx, max_dy)
    bl = shift_point(bl, max_dx, max_dy)

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return corners

# Генерация маски
def generate_document_mask(min_size=512, max_size=2048, max_shift=0.3):
    H = np.random.randint(min_size, max_size+1)
    W = np.random.randint(min_size, max_size+1)
    canvas = np.zeros((H, W), dtype=np.uint8)


    doc_w = np.random.randint(int(W*0.25), int(W*0.6))
    doc_h = np.random.randint(int(H*0.25), int(H*0.6))

    corners = generate_perspective_corners(doc_w, doc_h, max_shift=max_shift)

    offset_x = np.random.randint(0, W - doc_w)
    offset_y = np.random.randint(0, H - doc_h)
    corners += np.array([offset_x, offset_y])

    corners_int = corners.astype(np.int32)
    cv2.fillPoly(canvas, [corners_int], color=1)

    return canvas, corners_int

if __name__=='__main__':

    mask, corners = generate_document_mask()
    print("Corners:\n", corners)
    print("Mask shape:", mask.shape)

    plt.imshow(mask, cmap='gray')
    plt.show()