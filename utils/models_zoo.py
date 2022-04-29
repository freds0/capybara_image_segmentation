import os

class zoo:

    def __init__(self):
        '''
        Source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        '''
        self.models_dict = {
            'mask_rcnn_inception_resnet_v2_1024x1024': 'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz'
        }

    def get_link_model(self, model_name):
        return self.models_dict[model_name]

    def get_folder_model(self, model_name):
        url_model = self.get_link_model(model_name)
        model_tar_filename = os.path.basename(url_model)
        model_dir = model_tar_filename.replace('.tar.gz', '')
        return model_dir
