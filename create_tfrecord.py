import tensorflow as tf
import os
from downsampler import get_images_in_folder

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_image_pair(original_image_path, downsampled_image_path, downsampling_rate):
    orig_im_string = open(original_image_path, 'rb').read()
    donwsamp_im_string = open(downsampled_image_path, 'rb').read()
    orig_im_shape = tf.image.decode_jpeg(orig_im_string).shape.as_list()
    donwsamp_im_shape=  tf.image.decode_jpeg(donwsamp_im_string).shape.as_list()
    print(downsampled_image_path)
    print(donwsamp_im_shape)
    feature = {
        'height_orig': _int64_feature(orig_im_shape[0]),
        'width_orig': _int64_feature(orig_im_shape[1]),
        'depth_orig': _int64_feature(orig_im_shape[2]),
        'image_orig': _bytes_feature(orig_im_string),
        'height_down': _int64_feature(donwsamp_im_shape[0]),
        'width_down': _int64_feature(donwsamp_im_shape[1]),
        'depth_down': _int64_feature(donwsamp_im_shape[2]),
        'image_down': _bytes_feature(donwsamp_im_string),
        'downsampling_rate': _float_feature(downsampling_rate)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

if __name__ == '__main__':
    downsample_rates = (2, 4, 8, 16, 32)

    for rate in downsample_rates:
        record_file = os.path.join('tfrecords', '{:03d}.tfrecords'.format(rate))
        image_paths = get_images_in_folder('raw_images')
        downsampled_paths = ['downsampled/{:03d}/'.format(rate)+os.path.basename(im_name) for im_name in image_paths]

        with tf.python_io.TFRecordWriter(record_file) as writer:
          for orig_file, downsamp_file in zip(image_paths,downsampled_paths):
            try:
                sample = serialize_image_pair(orig_file, downsamp_file, rate)
                writer.write(sample.SerializeToString())
            except Exception as e:
                print('oops')