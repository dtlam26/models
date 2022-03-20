import os
import io
import sys
import pandas as pd
import random
import json
import requests
import tensorflow.compat.v1 as tf1
from utils import dataset_util
from collections import namedtuple, OrderedDict
import contextlib2
from utils import *
import matplotlib.pyplot as plt
from dataset_tools import tf_record_creation_util
import cv2

"""
Receiving CSV to turn into TFRecords
"""
class Tf_Encoder(object):
    def __init__(self,csv,obj_categories,img_dir=None, online=False, image_categories=None):
        self.df = pd.read_csv(csv)
        self.obj_categories = obj_categories
        self.image_categories = image_categories
        self.img_dir = img_dir
        self.online = online

    def create_tf_example(self,group_objects, path = None, image_format = b'jpg'):
        if self.online:
            encoded_jpg = requests.get(str(group_objects.filename)).content
        else:
            if path:
                with tf1.gfile.GFile(os.path.join(path, '{}'.format(group_objects.filename)), 'rb') as fid:
                    encoded_jpg = fid.read()
            else:
                if 'http' in group_objects.filename:
                    raise ValueError("Please turn on online mode")
                with tf1.gfile.GFile(group_objects.filename, 'rb') as fid:
                    encoded_jpg = fid.read()
        _ = tf1.image.decode_image(encoded_jpg)
        height,width,channel = _.shape


        filename = group_objects.filename.encode('utf8')

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        image_classes_text = []
        image_classes = []
        def _encode_value_to_group_text_n_id(value,encode_value_dict,groups):
            if type(value) is int:
                groups[0].append(encode_value_dict[int(value)].encode('utf8'))
                groups[1].append(int(value))
            else:
                groups[0].append(value.encode('utf8'))
                groups[1].append(int(dict((v,k) for k,v in encode_value_dict.items()).get(value)))

        for index, row in group_objects.objects.iterrows():
            if not row.x1 == -1:
                xmins.append(row['x1'] / width)
                xmaxs.append(row['x2'] / width)
                ymins.append(row['y1'] / height)
                ymaxs.append(row['y2'] / height)
                _encode_value_to_group_text_n_id(row['obj_class'],self.obj_categories,[classes_text,classes])

            if self.image_categories:
                _encode_value_to_group_text_n_id(row['image_class'],self.image_categories,[image_classes_text,image_classes])
        feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
        }
        if len(xmins) > 0:
            feature.update({
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                })
        if self.image_categories:
            feature.update({
                'image/class/text': dataset_util.bytes_feature(image_classes_text[0]),
                'image/class/label': dataset_util.int64_feature(image_classes[0])
                })
        tf_example = tf1.train.Example(features=tf1.train.Features(feature=feature))
        return tf_example

    def create_label_map(self,folder):
        msg = ''
        for idx, name in self.obj_categories.items():
            msg = msg + "item {\n"
            msg = msg + " id: " + str(idx) + "\n"
            msg = msg + " name: '" + name + "'\n}\n\n"
        with open(os.path.join(os.getcwd(),'tfrecords',folder,'obj_map.pbtxt'), 'w') as f:
            f.write(msg[:-1])
            f.close()
        if self.image_categories:
            msg = ''
            for idx, name in self.image_categories.items():
                msg = msg + "item {\n"
                msg = msg + " id: " + str(idx) + "\n"
                msg = msg + " name: '" + name + "'\n}\n\n"
            with open(os.path.join(os.getcwd(),'tfrecords',folder,'image_map.pbtxt'), 'w') as f:
                f.write(msg[:-1])
                f.close()

    def tfrecords_write(self,folder,file,shard=True,num_shards=10,image_format = b'jpg'):
        if not os.path.isdir('tfrecords/'+folder):
            os.mkdir('tfrecords/'+folder)

        self.create_label_map(folder)
        self.output_path = 'tfrecords/'+folder+'/'+file
        def obj_split(df):
            data = namedtuple('objects_in_img', ['filename', 'objects'])
            gb = df.groupby('image_path')
            return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
        if shard:
            print('--sharding--')
            self.output_path = self.output_path+'.record'
            num_shards = int(num_shards)
            tf_record_close_stack =  contextlib2.ExitStack()
            writer = tf_record_creation_util.open_sharded_output_tfrecords(
              tf_record_close_stack, self.output_path, num_shards = num_shards)
        else:
            self.output_path = self.output_path+'.tfrecord'
            writer = tf1.python_io.TFRecordWriter(self.output_path)

        path = self.img_dir
        self.output_path = os.path.join(os.getcwd(), self.output_path)
        grouped_images = obj_split(self.df)
        file_errors = 0

        for index,group_objects in enumerate(grouped_images):
            tf_example = self.create_tf_example(group_objects,image_format=image_format)
            if shard:
                output_shard_index = index % num_shards
                writer[output_shard_index].write(tf_example.SerializeToString())
            else:
                writer.write(tf_example.SerializeToString())
        if not shard:
            writer.close()
        print("FINISHED. There were %d errors" %file_errors)
        print('Successfully created the TFRecords: {}'.format(self.output_path))


exp = Tf_Encoder('/home/dtlam26/Documents/ASI/Fire_Smoke/test.csv',image_categories = {1: 'fire', 2: 'smoke', 3: 'normal'},obj_categories = {1: 'fire_local', 2: 'smoke_local'},online=True)

exp.tfrecords_write('test','test')
