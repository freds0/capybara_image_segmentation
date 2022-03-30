import os
import tensorflow as tf
import numpy as np
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import itertools

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def run_inference_for_single_image(model, image_np):
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))

    detections = dict(itertools.islice(detections.items(), num_detections))

    detections["num_detections"] = num_detections

    #image_np_with_detections = image_np.copy()

    # Handle models with masks:
    if "detection_masks" in detections:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detections["detection_masks"][0], detections["detection_boxes"][0],
               image_np.shape[0], image_np.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        detections["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return detections


def generate_inference(model, label_map, image_path, output_path):

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    image_np_with_detections = image_np.copy()
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np_with_detections)

    boxes = np.asarray(output_dict["detection_boxes"][0])
    classes = np.asarray(output_dict["detection_classes"][0]).astype(np.int64)
    scores = np.asarray(output_dict["detection_scores"][0])
    mask = np.asarray(output_dict["detection_masks_reframed"])

    # List of the strings that is used to add correct label for each box.
    category_index = label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=mask,
        use_normalized_coordinates=True,
        line_thickness=3)

    filename = os.path.basename(image_path)

    # Save to output file
    result = Image.fromarray(image_np_with_detections)
    output_filepath = os.path.join(output_path, filename)
    result.save(output_filepath)