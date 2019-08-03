"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class ComicConfig(Config):
    """Configuration for training on the eBDtheque dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "characters"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + character

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

###########################################################
#  Dataset
############################################################

class ComicDataset(utils.Dataset):

    def load_comic(self, dataset_dir, subset):
        """Load the eBDtheque dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only three classes to add.
        # self.add_class("comic", 1, "panel")
        # self.add_class("comic", 2, "character")
        self.add_class("comic", 1, "character")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        data = json.load(open(os.path.join(dataset_dir, "instances.json")))
        data = list(data.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]

        # Add images
        for i in data[2]:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            # if type(a['regions']) is dict:
            #     polygons = [r['shape_attributes'] for r in a['regions'].values()]
            # else:
            #     polygons = [r['shape_attributes'] for r in a['regions']]
            # p = a['segmentation'][0]
            # xcord = []
            # ycord = []
            # for i,j in enumerate(p):
            #     if(i%2 == 0):
            #         xcord.append(j)
            #     else:
            #         ycord.append(j)

            # shapes = {}
            # shapes['xcord'] = xcord
            # shapes['ycord'] = ycord

            # polygons = [shapes] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            # image_path = os.path.join(dataset_dir, a['filename'])
            # image = skimage.io.imread(image_path)
            # height, width = image.shape[:2]
            # img_id = a['image_id']
            # for i in annotations[2]:
            #     if(i['id'] == img_id):
            #         filename = i['file_name']
            #         height = i['height']
            #         width = i['width']

            filename = i['file_name']
            # height = i['height']
            # width = i['width']
            img_id = i['id']

            image_path = os.path.join(dataset_dir, filename)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            polygons = []
            for annotation in data[3]:
                if(img_id == annotation['image_id'] and annotation['category_id'] == 2):
                    p = annotation['segmentation'][0]
                    xcord = []
                    ycord = []
                    for i,j in enumerate(p):
                        if(i%2 == 0):
                            if(i == len(p)-2):
                                continue
                            else:
                                if(j > width):
                                    xcord.append(int(width)-1)
                                else:
                                    xcord.append(j)
                        else:
                            if(i == len(p)-1):
                                continue
                            else:
                                if(j > height):
                                    ycord.append(int(height)-1)
                                else:
                                    ycord.append(j)

                    shapes = {}
                    shapes['xcord'] = xcord
                    shapes['ycord'] = ycord
                    # shapes['cat_id'] = annotation['category_id']
                    polygons.append(shapes)


            self.add_image(
                "comic",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=int(width), height=int(height),
                polygons=polygons)



    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "comic":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]['polygons']

        m = np.zeros([image_info["height"], image_info["width"], len(annotations)], dtype=np.int32)
        
        for i, annotation in enumerate(annotations):
            # class_id = annotation['cat_id']

            rr, cc = skimage.draw.polygon(np.array(annotation['ycord']), np.array(annotation['xcord']), shape=(image_info["height"],image_info["width"]))
            m[rr,cc,i] = 1
            # print(image_info["height"], image_info["width"])
            # print(image_id)
            # for i in range(len(rr)):
            #     print(rr[i], cc[i])
            #     print("====================")

            # instance_masks.append(m)
            # class_ids.append(class_id)

        # mask = np.stack(instance_masks, axis=2).astype(np.bool)
        # class_ids = np.array(class_ids, dtype=np.int32)
        mask = m.astype(np.bool)
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)

        return mask, class_ids

        # comic = info["path"]
        # comic = ["panel", "character", "balloon"]
        # mask = np.zeros([annotations["height"], annotations["width"], len(annotations["polygons"])],
                        # dtype=np.int32)
        # for i, p in enumerate(annotations["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            # rr, cc = skimage.draw.polygon(p['xcord'], p['ycord'])
            # mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # class_ids = np.array([self.class_names.index(c) for c in comic])
        # return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "comic":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ComicDataset()
    dataset_train.load_comic(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ComicDataset()
    dataset_val.load_comic(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    # model.load_weights(filepath, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ComicConfig()
    else:
        class InferenceConfig(ComicConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        # model.load_weights(weights_path, by_name=True)
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
