import numpy as np

# this file is mainly used for product


# TODO reformat the following function
def get_iou_cpu(A, B, return_ioa=False):
    x1_a, y1_a, x2_a, y2_a, *_ = A
    x1_b, y1_b, x2_b, y2_b, *_ = B

    wa = x2_a - x1_a + 1
    ha = y2_a - y1_a + 1

    wb = x2_b - x1_b + 1
    hb = y2_b - y1_b + 1

    xx1 = min(x1_a, x1_b)
    yy1 = min(y1_a, y1_b)
    xx2 = max(x2_a, x2_b)
    yy2 = max(y2_a, y2_b)

    wi = max(0, wa + wb - (xx2 - xx1 + 1))
    hi = max(0, ha + hb - (yy2 - yy1 + 1))
    interseact_area = wi * hi
    if not return_ioa:
        union_area = wa * ha + wb * hb - interseact_area
        IoU = float(interseact_area) / float(union_area)
        return IoU
    else:
        return float(interseact_area) / float(wa * ha)


def rotate_box(box, M):
    x1, y1, x2, y2 = box
    points = np.array([[x1, y1, 1], [x1, y2, 1], [x2, y1, 1], [x2, y2, 1]])

    rotated_points = M.dot(points.T).T

    x1 = np.min(rotated_points[:, 0])
    y1 = np.min(rotated_points[:, 1])
    x2 = np.max(rotated_points[:, 0])
    y2 = np.max(rotated_points[:, 1])

    return [x1, y1, x2, y2]
