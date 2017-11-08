# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os, sys, cv2, random
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def read(file_path):
    f = open(file_path, 'r')
    c = f.read()
    f.close()
    return c

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


def load_anno():
    import json
    path = 'challenger/vallabel.json'

    objs = json.loads(read(path))
    
    m = {}
    for obj in objs:
        m[obj['image_id']] = obj
    return m

def anno(ax, obj):
    
    # 人体骨骼关键点共有14个
    # 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子
    # (x, y, v), v 表示是否可见，v=1可见，v=2 不可见，v=3 不在图内或不可推测
    keypoints_map = obj['keypoint_annotations']
    for key in keypoints_map:
        keypoints = keypoints_map[key]
        # 头部大小

        # 先假定头一定有
        (x_top, y_top, v_top) = get_pair(keypoints, 12)
        (x_neck, y_neck, v_neck) = get_pair(keypoints, 13)

        head_box = get_head_box(x_top, x_neck, y_top, y_neck)
        head_size = head_box[2]
        rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2], head_box[3], linewidth = 2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        # 其他部位我们乘以一个系数
        # 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，
        beta = [ 0.5, 0.5, 0.5, 0.5, 0.5,     0.5, 0.5, 0.5, 0.5, 0.5,      0.5, 0.5 ]
        for i in range(12):
            (x, y, v) = get_pair(keypoints, i)
            if v != 1:
                continue
            size = head_size * beta[i]
            box = get_box(x, y, size)
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth = 2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        human_box = get_human_box(keypoints)
        rect = patches.Rectangle((human_box[0], human_box[1]), human_box[2], human_box[3], linewidth = 2, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

def demo(sess, net, image_name, anno_obj):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    anno(ax, anno_obj)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        if cls != 'person':
            continue
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

if __name__ == '__main__':
    anno_obj_map = load_anno()
    img_ids = []
    if len(sys.argv) < 2:
        for i in range(10):
            img_ids.append(random.choice(os.listdir('challenger/valimg')).split('.')[0])

    else:
        for i in range(len(sys.argv)):
            if i < 1:
                continue
            img_ids.append(sys.argv[i])

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    demo_net = "VGGnet_test"
    net = get_network(demo_net)
    # load model
    print ('Loading network {:s}... '.format(demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, "model/VGGnet_fast_rcnn_iter_150000.ckpt")
    print (' done.')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    for im_id in img_ids:
        im_name = 'challenger/valimg/' + im_id + '.jpg'
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name, anno_obj_map[im_id])

    plt.show()

