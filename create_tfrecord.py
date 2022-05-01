#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.
Please note that this tool creates sharded output files.
Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --test_annotations_file="${TEST_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import ast
import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util



def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.
  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(category_index[category_id]['name'].encode('utf8'))
    area.append(object_annotations['area'])

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)
      if not object_annotations['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)
      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      encoded_mask_png.append(output_io.getvalue())
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def create_tf_record_from_coco_annotations(
    annotations_file, image_dir, output_path, include_masks):
  """Loads COCO annotation json files and converts to tf.Record format.
  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  """
  with tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf.python_io.TFRecordWriter(output_path)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = label_map_util.create_category_index(
        groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      tf.logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    tf.logging.info('%d images are missing annotations.',
                    missing_annotation_count)

    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        tf.logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      _, tf_example, num_annotations_skipped = create_tf_example(
          image, annotations_list, image_dir, category_index, include_masks)
      total_num_annotations_skipped += num_annotations_skipped
      output_tfrecords.write(tf_example.SerializeToString())
    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)

    print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('-y', '--yaml', default='config/parameters.yaml', help='Config file YAML format')
    parser.add_argument('--input_train_csv', help='Input train csv filepath.')
    parser.add_argument('--input_test_csv', help='Input test csv filepath.')
    parser.add_argument('--images_train_dir', help='Train images folder.')
    parser.add_argument('--images_test_dir', help='Test images folder.')
    parser.add_argument('--output_train_tfrecord', help='Output train TFRecord filepath.')
    parser.add_argument('--output_test_tfrecord', help='Output test TFRecord filepath.')
    parser.add_argument('--only_train', action='store_true', help='Execute only for train dataset')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.yaml))
        print(e)
        exit()

    config_input_train_csv = config['preprocess']['output_data_aug_csv'] if config['pipeline_config']['use_data_aug'] else config['pipeline_config']['input_train_csv']
    config_train_images_dir = config['preprocess']['output_data_aug_imgs_folder'] if config['pipeline_config']['use_data_aug'] else config['pipeline_config']['input_train_img_folder']
    labels_list = ast.literal_eval(config['pipeline_config']['classes_names'])
    train_csv_filepath = join(args.base_dir, args.input_train_csv) if args.input_train_csv else config_input_train_csv
    train_images_dir = join(args.base_dir, args.images_train_dir) if args.images_train_dir else config_train_images_dir
    train_output_path = join(args.base_dir, args.output_train_tfrecord) if args.output_train_tfrecord else config['pipeline_config']['train_record_path']

    create_tf_record_from_coco_annotations(
        train_csv_filepath, train_images_dir, train_output_path, True)

    if not (args.only_train):
        test_csv_filepath = join(args.base_dir, args.input_test_csv) if args.input_test_csv else config['pipeline_config']['input_test_csv']
        test_images_dir = join(args.base_dir, args.images_test_dir) if args.images_test_dir else config['pipeline_config']['input_test_img_folder']
        test_output_path = join(args.base_dir, args.output_test_tfrecord) if args.output_test_tfrecord else config['pipeline_config']['test_record_path']

        create_tf_record_from_coco_annotations(
            test_csv_filepath, test_images_dir, test_output_path, True)


