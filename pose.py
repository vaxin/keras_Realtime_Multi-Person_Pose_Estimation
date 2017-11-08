# coding: utf-8
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import sys, random, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

vallabel = 'challenger/vallabel.json'
def read(path):
    f = open(path, 'r')
    c = f.read()
    f.close()
    return c

print '======= loading annotations ==========='
c = read(vallabel)
'''{"url": "http://www.sinaimg.cn/dy/slidenews/4_img/2013_24/704_997547_218968.jpg", "image_id": "d8eeddddcc042544a2570d4c452778b912726720", "keypoint_annotations": {"human3": [0, 0, 3, 0, 0, 3, 0, 0, 3, 67, 279, 1, 87, 365, 1, 65, 345, 1, 0, 0, 3, 0, 0, 3, 0, 0, 3, 40, 454, 1, 44, 554, 1, 0, 0, 3, 20, 179, 1, 17, 268, 1], "human2": [444, 259, 1, 474, 375, 2, 451, 459, 1, 577, 231, 1, 632, 396, 1, 589, 510, 1, 490, 538, 1, 0, 0, 3, 0, 0, 3, 581, 535, 2, 0, 0, 3, 0, 0, 3, 455, 78, 1, 486, 205, 1], "human1": [308, 306, 1, 290, 423, 1, 298, 528, 1, 433, 297, 1, 440, 404, 1, 447, 501, 2, 342, 530, 1, 0, 0, 3, 0, 0, 3, 417, 520, 1, 0, 0, 3, 0, 0, 3, 376, 179, 1, 378, 281, 1]}, "human_annotations": {"human3": [0, 169, 114, 633], "human2": [407, 59, 665, 632], "human1": [265, 154, 461, 632]}}, {"url": "http://www.sinaimg.cn/dy/slidenews/4_img/2013_47/704_1154733_789201.jpg", "image_id": "054d9ce9201beffc76e5ff2169d2af2f027002ca", "keypoint_annotations": {"human3": [144, 180, 1, 171, 325, 2, 256, 428, 2, 265, 196, 1, 297, 311, 1, 300, 412, 1, 178, 476, 2, 0, 0, 3, 0, 0, 3, 253, 474, 2, 0, 0, 3, 0, 0, 3, 220, 23, 1, 205, 133, 1], "human2": [637, 374, 2, 626, 509, 1, 0, 0, 3, 755, 347, 1, 728, 538, 1, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 604, 169, 1, 674, 290, 1]'''
import json
annos = json.loads(c)
ground_truth = {}
for anno in annos:
    im_id = anno['image_id']
    ground_truth[im_id] = anno

print '======= done ==========='

def get_pair(keypoints, i):
    return (keypoints[i * 3], keypoints[i * 3 + 1], keypoints[i * 3 + 2])

def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def get_head_box(x_top, x_neck, y_top, y_neck):
    d = distance(x_top, y_top, x_neck, y_neck)
    d = int(d)
    return (int(x_top - d / 2), y_top, d, d)

def get_box(x, y, w):
    w = int(w)
    return (int(x - w / 2), int(y - w / 2), w, w)

def get_human_box(keypoints):
    (x_top, y_top, v_top) = get_pair(keypoints, 12)
    (x_neck, y_neck, v_neck) = get_pair(keypoints, 13)
    head_box = get_head_box(x_top, x_neck, y_top, y_neck)
    
    min_x, min_y = 10000, 10000
    max_x, max_y = 0, 0
    arr = []
    for i in range(14):
        (x, y, v) = get_pair(keypoints, i)
        if v == 1:
            arr.append((x, y))
    arr.append((head_box[0], head_box[1]))
    arr.append((head_box[0] + head_box[2], head_box[1] + head_box[3]))
    
    padding = head_box[2] / 4
    
    for (x, y) in arr:
        if x > max_x:
            max_x = x
        elif x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        elif y < min_y:
            min_y = y
    return (min_x - padding, min_y - padding, max_x - min_x + 2 * padding, max_y - min_y + 2 * padding)

def drawAnno(im, ax, obj, color = 'y'):
    ax.imshow(im, aspect='equal')
    # 人体骨骼关键点共有14个
    # 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子
    # (x, y, v), v 表示是否可见，v=1可见，v=2 不可见，v=3 不在图内或不可推测
    keypoints_map = obj['keypoint_annotations']
    for key in keypoints_map:
        keypoints = keypoints_map[key]
        # 头部大小

        # 先假定头一定有
        #(x_top, y_top, v_top) = get_pair(keypoints, 12)
        #(x_neck, y_neck, v_neck) = get_pair(keypoints, 13)

        #head_box = get_head_box(x_top, x_neck, y_top, y_neck)
        #head_size = head_box[2]
        #rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2], head_box[3], linewidth = 2, edgecolor=color, facecolor='none')
        #ax.add_patch(rect)

        # 其他部位我们乘以一个系数
        # 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，
        #beta = [ 0.5, 0.5, 0.5, 0.5, 0.5,     0.5, 0.5, 0.5, 0.5, 0.5,      0.5, 0.5 ]
        for i in range(14):
            (x, y, v) = get_pair(keypoints, i)
            if v != 1:
                continue
            size = 5
            box = get_box(x, y, size)
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth = 2, edgecolor = color, facecolor='none')
            ax.add_patch(rect)

def process(input_image, params, model_params):
    oriImg = cv2.imread(input_image)  # B,G,R order
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return (subset, candidate, all_peaks) 

def draw(input_image, subset, candidate, all_peaks):
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def getPoint(raw_points, challenger_id, candidate):
    challenger_ai_reverse_index = [ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2 ]
    i = challenger_ai_reverse_index[challenger_id]
    keypoint_index = raw_points[i - 1]
    if -1 == keypoint_index:
        return None
    else:
        (x, y) = candidate[keypoint_index.astype(int), 0:2]
        return (x, y)

def computeHeadTop(neck_point, nose_point):
    if neck_point is None or nose_point is None:
        return None
    return (2 * nose_point[0] - neck_point[0], 2 * nose_point[1] - neck_point[1])

def getAnnotation(img_id, subsets, candidate):
    # find connection in the specified sequence, center 29 is in the position 15
    #   1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    16   17    18    19
    # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    # challenger.ai 0/右肩，1/右肘，2/右腕，3/左肩，4/左肘，5/左腕，6/右髋，7/右膝，8/右踝，9/左髋，10/左膝，11/左踝，12/头顶，13/脖子
    # 由于头顶与keras模型没有对应关系，所以将头顶换成鼻子进行映射
    # 0/右肩，1/右肘，2/右腕，3/左肩，4/左肘，5/左腕，6/右髋，7/右膝，8/右踝，9/左髋，10/左膝，11/左踝，12/鼻子，13/脖子。

    anno = { "image_id" : img_id, "keypoint_annotations": {} }
    for i_person in range(len(subsets)):
        points = []
        raw_points = subsets[i_person]
        for i in range(14):
            if i == 12:
                # 计算头顶
                neck_point = getPoint(raw_points, 13, candidate)
                nose_point = getPoint(raw_points, 12, candidate)
                point = computeHeadTop(neck_point, nose_point)
            else:
                point = getPoint(raw_points, i, candidate)

            if point is None:
                points.append(0)
                points.append(0)
                points.append(3)
            else:
                (x, y) = point
                points.append(x)
                points.append(y)
                points.append(1)
        anno['keypoint_annotations']['human' + str(i_person + 1)] = points

    return anno

def test():
    img_ids = []
    if len(sys.argv) < 2:
        for i in range(1):
            img_ids.append(random.choice(os.listdir('challenger/valimg')).split('.')[0])

    else:
        for i in range(len(sys.argv)):
            if i < 1:
                continue
            img_ids.append(sys.argv[i])

        # generate image with body parts

    for im_id in img_ids:
        im_name = 'challenger/valimg/' + im_id + '.jpg'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        fig, ax = plt.subplots(figsize=(12, 12))
        (subset, candidate, all_peaks) = process(im_name, params, model_params)
        anno = getAnnotation(im_id, subset, candidate)
        print('========my result==========')
        print(anno)
        print('======= ground truth =========')
        print(ground_truth[im_id])
        #canvas = draw(im_name, subset, candidate, all_peaks)
        #plt.imshow(canvas)
        im = cv2.imread(im_name)
        fig, ax = plt.subplots(figsize=(12, 12))

        drawAnno(im, ax, anno, 'y')
        drawAnno(im, ax, ground_truth[im_id], 'b')
        plt.show()


    plt.show()

def write(path, c):
    f = open(path, 'w+')
    f.write(c)
    f.close()

def doMatch():
    files = os.listdir('challenger/testimg')
    for f in files:
        im_id = f.split('.')[0].strip()
        if im_id == "":
            continue
        json_file = 'challenger/testlabel/' + im_id + '.json'
        if os.path.exists(json_file):
            continue

        im_name = 'challenger/testimg/' + im_id + '.jpg'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print '-0-0-0-0-----> anno for {:s}'.format(im_name)
        (subset, candidate, all_peaks) = process(im_name, params, model_params)
        anno = getAnnotation(im_id, subset, candidate)
        write(json_file, json.dumps(anno))

def val():
    files = os.listdir('challenger/valimg')
    c = 0
    annos = []
    labels = []

    for f in files:
        im_id = f.split('.')[0].strip()
        if im_id == "":
            continue

        if c > 100:
            break
        c += 1

        im_name = 'challenger/valimg/' + im_id + '.jpg'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print '-0-0-0-0-----> anno for {:s}'.format(im_name)
        (subset, candidate, all_peaks) = process(im_name, params, model_params)
        anno = getAnnotation(im_id, subset, candidate)
        annos.append(anno)
        labels.append(ground_truth[im_id])

    write('challenger/valpredict.json', json.dumps(annos))
    write('challenger/valgroundtruth.json', json.dumps(labels))

def mergeTestAnnos():
    files = os.listdir('challenger/testimg')
    annos = []
    c = 0 
    for f in files:
        im_id = f.split('.')[0].strip()
        if im_id == "":
            continue
        json_file = 'challenger/testlabel/' + im_id + '.json'
        if os.path.exists(json_file):
            annos.append(json.loads(read(json_file)))
            c += 1
            if c % 100 == 0:
                print(c)

    write('challenger/testpredict.json', json.dumps(annos))

def showRandomTestAnno():
    im_id = random.choice(os.listdir('challenger/testimg')).split('.')[0]
    im_path = 'challenger/testimg/' + im_id + '.jpg'
    json_file = 'challenger/testlabel/' + im_id + '.json'

    anno = json.loads(read(json_file))
    im = cv2.imread(im_name)
    fig, ax = plt.subplots(figsize=(12, 12))
    drawAnno(im, ax, anno, 'y')
    plt.show()

if __name__ == '__main__':
    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights('model/keras/model.h5')

    # load config
    params, model_params = config_reader()

    test()
    #doMatch()
    #val()
    #mergeTestAnnos()
