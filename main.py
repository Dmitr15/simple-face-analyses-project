from pprint import pprint
import matplotlib.image as mpimg
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import pandas as pd


def face_detect(backends):
    fig, axis = plt.subplots(3, 2, figsize=(15, 10))
    axis = axis.flatten()
    for i, b in enumerate(backends):
        try:
            face = DeepFace.detectFace('Brad/brad_1.png', target_size=(224, 224), detector_backend=b)
            axis[i].imshow(face[0])
            axis[i].set_title(b)
        except:
            pass
    plt.show()


def verification():
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
    res = DeepFace.verify('Brad/brad_1.png', 'Brad/brad_4.png', model_name=models[1])
    fig, axis = plt.subplots(1, 2, figsize=(15, 5))
    # axis[0].imshow(mpimg.imread('Brad/brad_1.png'))
    # axis[1].imshow(mpimg.imread('Brad/brad_4.png'))
    # fig.suptitle(f'Verified {res['verified']}')
    # plt.show()
    pprint(res)


def face_find():
    res = DeepFace.find(img_path='Brad/brad_1.png', db_path='Brad/')
    pprint(res)


def facial_attribute_analysis():
    res = DeepFace.analyze(img_path='Brad/brad_1.png')
    print(pd.DataFrame(res["emotion"], index=[0]).T.plot(kind="bar"))


if __name__ == '__main__':
    backends = ["opencv", "ssd", "dlib" "mtcnn", "retinaface", "mediapipe"]
    # face_detect(backends)
    # verification()
    # face_find()
    facial_attribute_analysis()
