import os
import argparse
import random


import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as cocomask


CLASS_MAP = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic",
    11: "fire",
    12: "street",
    13: "stop",
    14: "parking",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports",
    38: "kite",
    39: "baseball",
    40: "baseball",
    41: "skateboard",
    42: "surfboard",
    43: "tennis",
    44: "bottle",
    45: "plate",
    46: "wine",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted",
    65: "bed",
    66: "mirror",
    67: "dining",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy",
    89: "hair",
    90: "toothbrush",
    91: "hair",
    92: "banner",
    93: "blanket",
    94: "branch",
    95: "bridge",
    96: "building",
    97: "bush",
    98: "cabinet",
    99: "cage",
    100: "cardboard",
    101: "carpet",
    102: "ceiling",
    103: "ceiling",
    104: "cloth",
    105: "clothes",
    106: "clouds",
    107: "counter",
    108: "cupboard",
    109: "curtain",
    110: "desk",
    111: "dirt",
    112: "door",
    113: "fence",
    114: "floor",
    115: "floor",
    116: "floor",
    117: "floor",
    118: "floor",
    119: "flower",
    120: "fog",
    121: "food",
    122: "fruit",
    123: "furniture",
    124: "grass",
    125: "gravel",
    126: "ground",
    127: "hill",
    128: "house",
    129: "leaves",
    130: "light",
    131: "mat",
    132: "metal",
    133: "mirror",
    134: "moss",
    135: "mountain",
    136: "mud",
    137: "napkin",
    138: "net",
    139: "paper",
    140: "pavement",
    141: "pillow",
    142: "plant",
    143: "plastic",
    144: "platform",
    145: "playingfield",
    146: "railing",
    147: "railroad",
    148: "river",
    149: "road",
    150: "rock",
    151: "roof",
    152: "rug",
    153: "salad",
    154: "sand",
    155: "sea",
    156: "shelf",
    157: "sky",
    158: "skyscraper",
    159: "snow",
    160: "solid",
    161: "stairs",
    162: "stone",
    163: "straw",
    164: "structural",
    165: "table",
    166: "tent",
    167: "textile",
    168: "towel",
    169: "tree",
    170: "vegetable",
    171: "wall",
    172: "wall",
    173: "wall",
    174: "wall",
    175: "wall",
    176: "wall",
    177: "wall",
    178: "water",
    179: "waterdrops",
    180: "window",
    181: "window",
    182: "wood"
}

def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image

    Args:
        im: a 3-channel uint8 image in BGR (opencv default order)
        mask: a binary 1-channel image of the same size stores the mask information
        color: if None, will choose automatically, i.e., red
    """

    if color is None:
        color = np.array([0, 0, 255])

    im = np.where( 
        np.repeat((mask > 0)[:, :, None], 3, axis=2),
        im * (1 - alpha) + color * alpha, 
        im
    )
    return im


def parse_args():
    parser = argparse.ArgumentParser(description="Display random images from coco dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', default="/mnt/e/workspace/coco", help='specifying the root path of coco dataset')
    parser.add_argument('--ann', default="/mnt/e/workspace/coco/annotations/instances_train2017.json", help='specifying annotation file to use')
    parser.add_argument('--save', action='store_true', help='If set, it will save the image')
    parser.add_argument('--random', action="store_true", help='If set, it will get random images')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    # initialize coco
    coco = COCO(args.ann)

    # plot some coco dataset information
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in cats]
    print('coco categories : \n{}'.format(' '.join(cat_names)))

    supcat_names = set([cat['supercategory'] for cat in cats])
    print('coco supercategories : \n{}'.format(' '.join(supcat_names)))

    imageIds = coco.getImgIds()
    print('total image is {}'.format(len(imageIds)))

    if args.random:
        image_id = np.random.randint(0, len(imageIds))
    else:
        image_id = 0

    image_info = coco.loadImgs(imageIds[image_id])[0]

    imname = image_info['file_name']
    annIds = coco.getAnnIds(image_info['id'])

    im = cv2.imread(os.path.join(args.datadir, "train2017", imname))
    anns = coco.loadAnns(annIds)
    
    ann_cat_names = []
    for ann in anns:
        cat_name = CLASS_MAP[ann['category_id']]
        ann_cat_names.append(cat_name)
        # display bounding box
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        b, g, r = random.randint(0,255), random.randint(0,255), random.randint(0,255)
        cv2.putText(im, cat_name, (x,y-5), cv2.FONT_HERSHEY_PLAIN, 
                   0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.rectangle(im, (x, y), (x+w, y+h), (255,0,0), 1)

        # display mask
        if 'segmentation' in ann:
            mask_im = coco.annToMask(ann) * 255
            im = draw_mask(im, mask_im, color=np.array((b, g, r), dtype=np.float32))

    if args.save:
        cv2.imwrite('output.jpg', im)

    print('getting coco image{} with image id {} with {} annotations'.format(imname, image_id, len(anns)))
    print('The image contains the following annotations')
    print(ann_cat_names)

if __name__ == "__main__":
    main()









