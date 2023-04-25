# From https://github.com/hysts/anime-face-detector/blob/main/demo.ipynb
import cv2
import matplotlib.pyplot as plt
import numpy as np

from mmdet.apis import init_detector, inference_detector, show_result_pyplot



#@title Contour Definition

# https://github.com/hysts/anime-face-detector/blob/main/assets/landmarks.jpg
FACE_BOTTOM_OUTLINE = np.arange(0, 5)
LEFT_EYEBROW = np.arange(5, 8)
RIGHT_EYEBROW = np.arange(8, 11)
LEFT_EYE_TOP = np.arange(11, 14)
LEFT_EYE_BOTTOM = np.arange(14, 17)
RIGHT_EYE_TOP = np.arange(17, 20)
RIGHT_EYE_BOTTOM = np.arange(20, 23)
NOSE = np.array([23])
MOUTH_OUTLINE = np.arange(24, 28)

FACE_OUTLINE_LIST = [FACE_BOTTOM_OUTLINE, LEFT_EYEBROW, RIGHT_EYEBROW]
LEFT_EYE_LIST = [LEFT_EYE_TOP, LEFT_EYE_BOTTOM]
RIGHT_EYE_LIST = [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM]
NOSE_LIST = [NOSE]
MOUTH_OUTLINE_LIST = [MOUTH_OUTLINE]

# (indices, BGR color, is_closed)
CONTOURS = [
    (FACE_OUTLINE_LIST, (0, 170, 255), False),
    (LEFT_EYE_LIST, (50, 220, 255), False),
    (RIGHT_EYE_LIST, (50, 220, 255), False),
    (NOSE_LIST, (255, 30, 30), False),
    (MOUTH_OUTLINE_LIST, (255, 30, 30), True),
]



#@title Visualization Function


def visualize_box(image,
                  box,
                  score,
                  lt,
                  box_color=(0, 255, 0),
                  text_color=(255, 255, 255),
                  show_box_score=True):

    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), box_color, lt)
    if not show_box_score:
        return
    cv2.putText(image,
                f'{round(score * 100, 2)}%', (box[0], box[1] - 2),
                0,
                lt / 2,
                text_color,
                thickness=max(lt, 1),
                lineType=cv2.LINE_AA)


def visualize_landmarks(image, pts, lt, landmark_score_threshold):
    for *pt, score in pts:
        pt = tuple(np.round(pt).astype(int))
        if score < landmark_score_threshold:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(image, pt, lt, color, cv2.FILLED)


def draw_polyline(image, pts, color, closed, lt, skip_contour_with_low_score,
                  score_threshold):
    if skip_contour_with_low_score and (pts[:, 2] < score_threshold).any():
        return
    pts = np.round(pts[:, :2]).astype(int)
    cv2.polylines(image, np.array([pts], dtype=np.int32), closed, color, lt)


def visualize_contour(image, pts, lt, skip_contour_with_low_score,
                      score_threshold):
    for indices_list, color, closed in CONTOURS:
        for indices in indices_list:
            draw_polyline(image, pts[indices], color, closed, lt,
                          skip_contour_with_low_score, score_threshold)

def box_score(pred: np.ndarray,image: np.ndarray):
    box = pred['bbox']
    box, score = box[:4], box[4]
    box = np.round(box).astype(int)
    if box[0] < 0: # x1 is too small
        box[0] = 0
    if box[1] < 0: # y1 too small
        box[1] = 0
    ylim,xlim,n = image.shape
    if box[2] > xlim:
        box[2] = xlim
    if box[3] > ylim:
        box[3] = ylim
    return box, score

def visualize(image: np.ndarray,
              preds: np.ndarray,
              face_score_threshold: float,
              landmark_score_threshold: float,
              show_box_score: bool = True,
              draw_contour: bool = True,
              skip_contour_with_low_score=False):
    res = image.copy()

    box,score = box_score(preds,image)
    pred_pts = preds['keypoints']

    # line_thickness
    lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

    visualize_box(res, box, score, lt, show_box_score=show_box_score)
    if draw_contour:
        visualize_contour(
            res,
            pred_pts,
            lt,
            skip_contour_with_low_score=skip_contour_with_low_score,
            score_threshold=landmark_score_threshold)
    visualize_landmarks(res, pred_pts, lt, landmark_score_threshold)

    return res

def crop_faces(image: np.ndarray,
              preds: np.ndarray,
              face_score_threshold: float):
    cropped_images = [] # list of ndarrays
    for pred in preds:
        box, score = box_score(pred,image)
        if score > face_score_threshold:
            cropped_images.append(image[box[1]:box[3],box[0]:box[2],:])
    return cropped_images


if __name__ == "__main__":
    config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
    
    #@title Visualization Arguments

    face_score_threshold = 0.1  #@param {type: 'slider', min: 0, max: 1, step:0.1}
    landmark_score_threshold = 0.1  #@param {type: 'slider', min: 0, max: 1, step:0.1}
    show_box_score = True  #@param {'type': 'boolean'}
    draw_contour = True  #@param {'type': 'boolean'}
    skip_contour_with_low_score = True  #@param {'type': 'boolean'}

    image = cv2.imread('Images/google-images/Adolf_Junkers/Adolf_Junkers_91.jpeg')
    # image = cv2.imread('Images/original-images/Adolf_Junkers/294002.jpg')
    result = inference_detector(model, image)
    show_result_pyplot(model, image, result)
    res = visualize(image, result.pred_instances, face_score_threshold, landmark_score_threshold,
                show_box_score, draw_contour, skip_contour_with_low_score)

    plt.figure(figsize=(30, 30))
    plt.imshow(res[:, :, ::-1])
    plt.axis('off')
    plt.show()

    cropped_images = crop_faces(image,preds,.01)
    for idx in range(len(cropped_images)):
        cv2.imwrite('test'+str(idx)+'.png',cropped_images[idx])
