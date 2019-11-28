
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
import pprint

from Super_resolution import super_resolution

pp = pprint.PrettyPrinter()

flags = tf.app.flags

#Setting
flags.DEFINE_string("Set_data_dir", "Data/basic_1_nocube/", "Set Training Data path")
flags.DEFINE_integer("full_side_size", 64, "data one side size")
flags.DEFINE_integer("full_side_size_x", 64, "data one side size")
flags.DEFINE_integer("full_side_size_y", 128, "data one side size")
flags.DEFINE_integer("full_side_size_z", 64, "data one side size")
flags.DEFINE_integer("start_frame", 51, "test_data start frame")
flags.DEFINE_integer("total_frame", 75, "test_data total frame")
flags.DEFINE_string("scene", "basic_1_nocube", "name of scene")


data_type = np.float32

FLAGS = flags.FLAGS

def main(_):

    pp.pprint(flags.FLAGS.__flags)

    import timeit
    start_time = timeit.default_timer()
    Set = super_resolution(
        Set_data_dir=FLAGS.Set_data_dir,
        full_side_size=FLAGS.full_side_size,
        full_side_size_x=FLAGS.full_side_size_x,
        full_side_size_y=FLAGS.full_side_size_y,
        full_side_size_z=FLAGS.full_side_size_z,
        start_frame=FLAGS.start_frame,
        total_frame=FLAGS.total_frame,
        scene=FLAGS.scene,
        data_type=data_type
    )
    print("Set_GPU_Octree_training_data_start")
    Set.Set_Octree_test_data_GPU_nocube(8)
    end_time = timeit.default_timer() - start_time
    print("total time is = ", end_time)

if __name__ == '__main__':
    tf.app.run()
