import cv2
import os
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.externals import joblib
from random import randint
import random
random.seed("487238956")


train_data_dir = "../data/training_data/data_100"
hard_neg_dir = "../data/training_data/hard_neg"
hn_dir = "../data/training_data/hn_iter"

if not os.path.exists(hard_neg_dir):
    os.makedirs(hard_neg_dir)


def hog_multi(data):
    try:
        len(data[0, 0])
        return np.concatenate([hog(data[:, :, 0]),
                               hog(data[:, :, 1]),
                               hog(data[:, :, 2])])
    except Exception as err:
        # print(err)
        # raise

        print("generating grayscale features")
        return hog(data)


def load_core_data(data_dir, color=0):
    pos, neg = [], []
    for file_name in os.listdir(train_data_dir):
        if file_name.startswith("pos"):
            pos.append(cv2.imread(os.path.join(train_data_dir, file_name), color))
        else:
            neg.append(cv2.imread(os.path.join(train_data_dir, file_name), color))
    return pos, neg


def create_train_data(pos, neg, im_modifier=lambda x: x):
    # Collect positive and negative samples from data_dir
    pos, neg = [im_modifier(i) for i in pos], [im_modifier(i) for i in neg]
    print("creating HOG descriptor")
    pos_hog = [hog_multi(i) for i in pos]
    neg_hog = [hog_multi(i) for i in neg]
    return pos_hog, neg_hog


def train_svm(pos, neg, prob=False):
    train_data = pos + neg
    train_labels = [1 for i in pos] + [0 for i in neg]
    print("creating SVM")
    svm = SVC(kernel="linear", class_weight="balanced", probability=prob)
    print("Fitting")
    svm.fit(train_data, train_labels)
    return svm


def hard_negative(svm, negs, imsize=50):
    # assumes that the neg image is larger than the SVM
    w, h, _ = negs[0].shape
    negs_ss = []
    print("generating subsamples")
    for neg in negs:
        i, j = 0, 0
        imax, jmax = h-imsize, w-imsize
        while (i < imax and j < jmax):
            i += random.randint(1, 5)
            j += random.randint(1, 5)
            if i > imax or j > jmax:
                break
            else:
                negs_ss.append(neg[i:i+imsize, j:j+imsize])
    print("ss_generation completed")
    negs_ss_hog = [hog_multi(i) for i in negs_ss]
    predicted = svm.predict(negs_ss_hog)
    for num, i in enumerate(predicted):
        if i == 1:
            cv2.imwrite(os.path.join(hard_neg_dir, "_".join(["hneg", str(num)])) + ".png", negs_ss[num])




#  first run
def first_run(color=0):
    # Uses the original training data to bootstrap negative training set production
    print("creating training data")
    pos, neg = load_core_data(train_data_dir, color=color)
    pos, neg = create_train_data(pos, neg, lambda x: x[25:75, 25:75])
    print("training SVM")
    svm = train_svm(pos, neg)
    rpos, rneg = load_core_data(train_data_dir, color=1)
    print("Conducting hard negative ")
    hard_negative(svm, rneg)
    return svm


#  Create an SVM using training data, hn_iteration and the first run of bootstrapped ata
def create_iter(color=0):
    pos_folders, neg_folders = [], []
    for iteration_folder in os.listdir(hn_dir):
        pos_folders.append(os.path.join(hn_dir, iteration_folder, "pos"))
        neg_folders.append(os.path.join(hn_dir, iteration_folder, "neg"))

    hn_pos, hn_neg = [], []
    cur_iter = len(neg_folders)

    print("Loading iterative data. {} iterations found".format(cur_iter))
    for fpath in neg_folders:
        for file_path in os.listdir(fpath):
            ipath = os.path.join(fpath, file_path)
            hn_neg.append(hog(cv2.imread(ipath, 0)))

    for fpath in pos_folders:
        for file_path in os.listdir(fpath):
            ipath = os.path.join(fpath, file_path)
            hn_pos.append(hog(cv2.imread(ipath, 0)))

    print("Loading original HOG data")
    pos, neg = load_core_data(train_data_dir, color=color)
    pos, neg = create_train_data(pos, neg, lambda x: x[25:75, 25:75])

    pos += hn_pos
    neg += hn_neg
    print("pos/neg len", len(pos), len(neg))

    hneg = []

    for path in [os.path.join(hard_neg_dir, i) for i in os.listdir(hard_neg_dir)]:
        hneg.append(cv2.imread(path, color))

    neg = neg + [hog_multi(i) for i in hneg]

    print("Fitting SVM to {} samples. ({} pos, {} neg)".format(len(pos) + len(neg), len(pos), len(neg)))
    svm = train_svm(pos, neg, prob=True)
    joblib.dump(svm, "../data/svm/SVM_HN_{}.pkl".format(str(cur_iter)))


# test on images
def run_iter(iter, idir="../data/beachVolleyball1", color=0):
    svm = joblib.load("../data/svm/SVM_HN_{}.pkl".format(str(iter)))
    image_paths = [os.path.join(idir, i) for i in os.listdir(idir)]
    out_path = "../data/training_data/hniter/i{}".format(str(iter))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for index in range(0, 200, 5):
        print("processing image ", index)
        im = cv2.imread(image_paths[index], color)
        height, width, _ = im.shape
        probas, images = [], []
        p_index = 0
        for i in range(0, height - 50, 3):
            for j in range(0, width - 50, 3):
                temp = im[i:i+50, j:j+50]
                features = hog_multi(temp).reshape(1, -1)
                a = svm.predict_proba(features)
                probas.append([p_index, a[0, 0]])
                images.append(temp)
                p_index += 1
        print(sorted(probas, key=lambda x: x[1])[0])
        print(sorted(probas, key=lambda x: x[1])[-1])
        for target_index, prob in sorted(probas, key=lambda x: x[1])[-40:]:
            cv2.imwrite(os.path.join(out_path, "_".join(["hneg_self", str(index), str(target_index)])) +".png", images[target_index])


if __name__ == "__main__":
    color = 1
    # first_run(color = color)
    create_iter(color=color)
    run_iter(0, color=color)
