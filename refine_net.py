# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import optimizers
import numpy as np
import random, os, cv2

def makeModel():
    model = Sequential()
    model.add(Dense(64, input_dim = 14 * 2, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(2, activation = 'relu'))
    return model

import keras.backend as K

def customLoss(yTrue, yPred):
    return K.mean(K.exp((yTrue - yPred) ** 2))

def train(model, x_train, y_train, x_val, y_val):
    batch_size = 100
    epochs = 10000
    sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.99, nesterov = True)
    rmsp = optimizers.RMSprop()
    #model.compile(optimizer = rmsp, loss = customLoss)
    model.compile(optimizer = sgd, loss = 'mean_square_error')
    #early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 100)
    components = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data=(x_val, y_val)) #, callbacks = [ early_stopping ])

def trainModel(x_train, y_train, x_val, y_val):
    model = makeModel()
    try:
        train(model, x_train, y_train, x_val, y_val)
    except:
        pass
    model.save('model/adjust_points.h5')
    
def loadModel():
    from keras.models import load_model
    model = load_model('model/adjust_points.h5', custom_objects = { 'customLoss': customLoss })
    return model

def predict(model, x):
    return model.predict(np.asarray(x))

def distance(anno_points, predict_points):
    anno_points = np.reshape(np.asarray(anno_points), (14, 3))
    predict_points = np.reshape(np.asarray(predict_points), (14, 3))
    return np.mean(np.sum(np.exp((anno_points[:,:2] - predict_points[:,:2]) ** 2) - 1, axis = 1))

def con32(arr):
    return np.reshape(np.reshape(np.asarray(arr), (14, 3))[:,:2], (14*2))

def genPairs(anno, predict):
    anno_humans = anno['keypoint_annotations']
    predict_humans = predict['keypoint_annotations']

    anno_keys = anno_humans.keys()
    anno_count = len(anno_keys)

    predict_keys = predict_humans.keys()
    predict_count = len(predict_keys)

    if predict_count == 0 or anno_count == 0:
        return None

    if predict_count != anno_count:
        return None
    
    anno_set = set()
    predict_set = set()
    
    ls = []
    for anno_key, anno in anno_humans.iteritems():
        for predict_key, predict in predict_humans.iteritems():
            score = distance(anno, predict)
            ls.append([ anno_key, predict_key, score ])
    ls.sort(key = lambda x: x[2])

    pairs = []
    for item in ls:
        anno_key, predict_key, _ = item
        if anno_key in anno_set or predict_key in predict_set:
            continue
        pairs.append((con32(anno_humans[anno_key]), con32(predict_humans[predict_key])))
        anno_set.add(anno_key)
        predict_set.add(predict_key)

    return pairs

import zerox, json, pickle
def loadValDatasets():
    return pickle.load(open('test.np', 'r'))

def genAndSaveValDatasets():
    vallabel = 'challenger/vallabel.json'
    print '======= loading annotations ==========='
    c = zerox.read(vallabel)
    annos = json.loads(c)
    ground_truth = {}
    for anno in annos:
        im_id = anno['image_id']
        ground_truth[im_id] = anno

    print '======= done ==========='

    files = os.listdir('challenger/predictvallabel')
    X = []
    Y = []
    N = []
    c = 0
    for f in files:
        im_id = f.split('.')[0].strip()
        if im_id == "":
            continue
        json_file = 'challenger/predictvallabel/' + im_id + '.json'
        im_file = 'challenger/valimg/' + im_id + '.jpg'
        c += 1
        if c % 100 == 0:
            print 'load data: ', c

        h, w = cv2.imread(im_file).shape[:2]
        n = float(max(h, w))
        predict = json.loads(zerox.read(json_file))
        anno = ground_truth[im_id]
        pairs = genPairs(anno, predict)
        if pairs is None:
            continue

        for pair in pairs:
            X.append(pair[0])
            Y.append(pair[1])
            N.append((w, h))
    pickle.dump((np.asarray(X), np.asarray(Y), np.asarray(N)), open('test.np', 'wr'))

def test():
    x_train = generateX(1000)
    y_train = targetF(x_train)

    x_test = generateX(100)
    y_test = targetF(x_test)

    trainModel(x_train, y_train, x_test, y_test)

def trainRefineModel(X, Y):
    size = X.shape[0]
    divide = int(size * 4 / 5.0)
    trainX, valX = X[:divide], X[divide:]
    trainY, valY = Y[:divide], Y[divide:]
    trainModel(trainX, trainY, valX, valY)

def stat(X, Y):
    arr = np.mean(np.exp((X - Y) ** 2), axis = 0).tolist()
    names = [ "右肩", "右肘", "右腕", "左肩", "左肘", "左腕", "右髋", "右膝", "右踝", "左髋", "左膝", "左踝", "头顶", "脖子" ]
    for i, diff in enumerate(arr):
        print names[i/2], 'X' if i % 2 == 0 else 'Y', diff

    return np.mean(np.sum(np.exp((anno_points[:,:2] - predict_points[:,:2]) ** 2), axis = 1))
    print('total=', np.mean(np.exp((X - Y) ** 2)))

def normalize1(X, Y, N):
    N = np.max(N, axis = 1)[:,None].astype(float)
    return (X / N, (Y - X) / N / 2.0 + 0.5)

def normalize(X, Y, N):
    N = np.max(N, axis = 1)[:,None].astype(float)
    return (X / N, (Y / N)[:13])

def expeTrain():
    (oX, oY, N) = loadValDatasets()
    X, Y = normalize(oX, oY, N)
    trainRefineModel(X, Y)

def expeStat():
    (oX, oY, N) = loadValDatasets()
    X, Y = normalize(oX, oY, N)

    model = loadModel()
    predict_Y = predict(model, X)
    N = np.max(N, axis = 1)[:,None].astype(float)
    predict_Y = (predict_Y - 0.5) * 2 * N + oX

    print '---before---'
    stat(oX, oY)
    print '---after---'
    stat(predict_Y, oY)

def printAnnosDistance(anno, predict):
    anno_humans = anno['keypoint_annotations']
    predict_humans = predict['keypoint_annotations']

    for anno_key, anno in anno_humans.iteritems():
        for predict_key, predict in predict_humans.iteritems():
            score = distance(anno, predict)
            print("%s - %s : %f" % (anno_key, predict_key, score))

def predictAnnotations(ct):
    vallabel = 'challenger/vallabel.json'
    print '======= loading annotations ==========='
    c = zerox.read(vallabel)
    annos = json.loads(c)
    ground_truth = {}
    for anno in annos:
        im_id = anno['image_id']
        ground_truth[im_id] = anno

    print '======= done ==========='

    model = loadModel()

    files = os.listdir('challenger/' + ct + 'img')
    X = []
    N = []
    for f in files:
        im_id = f.split('.')[0].strip()
        if im_id == "":
            continue
        if im_id != "412d6879818e30b8448b4e89d7af57a7df88171f":
            continue
        json_file = 'challenger/predict' + ct + 'label/' + im_id + '.json'
        im_file = 'challenger/' + ct + 'img/' + im_id + '.jpg'

        h, w = cv2.imread(im_file).shape[:2]
        n = float(max(h, w))
        old_anno = json.loads(zerox.read(json_file))

        new_anno = { "image_id" : im_id, "keypoint_annotations": {} }

        for key in old_anno['keypoint_annotations']:
            human_anno = old_anno['keypoint_annotations'][key] # 14-d array
            oX = np.asarray([ con32(human_anno) ])
            X = oX / n
            predict_Y = predict(model, X)
            predict_Y = (predict_Y - 0.5) * 2 * n + oX
            new_predict = predict_Y[0].astype(int)
            tmp = []

            for item in np.reshape(new_predict, (14, 2)).tolist():
                tmp.append(item[0])
                tmp.append(item[1])
                tmp.append(3 if item[0] == 0 and item[1] == 0 else 1)

            new_anno['keypoint_annotations'][key] = tmp

        gt = ground_truth[im_id]   
        import eval
        old_score = eval.compare([ gt ], [ old_anno ])
        print("--------> old_score:", old_score)
        print("-----> old distance")
        printAnnosDistance(gt, old_anno)

        new_score = eval.compare([ gt ], [ new_anno ])
        print("--------> new_score:", new_score)
        print("-----> new distance")
        printAnnosDistance(gt, new_anno)

        print("======== ground truth==========")
        print(gt)
        print("========old anno===========")
        print(old_anno)
        print("========new anno===========")
        print(new_anno)

        #json_file = 'challenger/predictrefine' + ct + 'label/' + im_id + '.json'
        # write to new refine direct
        #zerox.write(json_file, json.dumps(new_anno))
        break

if __name__ == "__main__":
    predictAnnotations('val')
    #expeTrain()
    #expeStat()
