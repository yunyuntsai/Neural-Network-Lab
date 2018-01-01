from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy import misc

import sys
import os
import time
import pdb

import model

#========================#
#    Global Parameter    #
#========================#
Student_ID = sys.argv[1]
model_call = getattr(model, Student_ID)
BATCH_SIZE = 1
CLASS_NUM  = 12
Dataset_Path = '../datasets/CamVid'

#--------------#
#    Define    #
#--------------#
def CamVid_dataset_parser(
    Dataset_Path,
    data_index,
    target_index
    ):
    
    """
    Simultaneously read out the data and target.
    Also, do some image pre-processing. 
    (You can add your own image pre-processing)
    
    Args :
    (1) Dataset_Path : Path to the CamVid directory.
    (2) data_index   : The array of the data images name.
    (3) target_index : The array of the target images name.
    
    Returns:
    (1) data   : The data images. Array Shape = [Total_Image_Num, Image_Height, Image_Width, Image_Depth (3 for RGB)]
    (2) target : The target images. Array Shape = [Total_Image_Num, Image_Height, Image_Width, Image_Depth (3 for RGB)]
    """
    
    # data_index : The name of all the images
    for iter in range(len(data_index)):
        # get each image name
        data_name   = data_index[iter]
        target_name = target_index[iter]
    
        # Read Image
        data_tmp   = misc.imread(Dataset_Path + data_name)
        target_tmp = misc.imread(Dataset_Path + target_name)
        
        # Data Preprocessing : 
        # You can add your own image preprocessing below.
        """
        """
        # If you want to resize the input size, you can uncomment the following code.
        # - H_resize: The height of the image you want to resize.
        # - W_resize: The width of the image you want to resize.
        """
        H_resize = 224
        W_resize = 224
        data_tmp   = scipy.misc.imresize(data_tmp,   (H_resize, W_resize))
        target_tmp = scipy.misc.imresize(target_tmp, (H_resize, W_resize))
        """
        
        # Concatenate each data to a big array 
        # Final shape of array : [Total_Image_Num, Image_Height, Image_Width, Image_Depth (3 for RGB)]
        if iter==0:
            data   = np.expand_dims(data_tmp  , axis=0)
            target = np.expand_dims(target_tmp, axis=0)
        else:
            data   = np.concatenate([data  , np.expand_dims(data_tmp  , axis=0)], axis=0)
            target = np.concatenate([target, np.expand_dims(target_tmp, axis=0)], axis=0)
            
    return data, target

def one_hot(
    target,
    class_num
    ):
    
    """
    Modify the value to the one-of-k type.
    
    i.e. array([[1, 2]
                [3, 4]])
    
    args:
    (1) target    : The 4-D array of the image. Array Shape = [Image_Num, Image_Height, Image_Width, Image_Depth]
    (2) class_num : The number of the class. (i.e. 12 for the CamVid dataset; 10 for the mnist dataset)
    
    Returns:
    (1) one_hot_target: The 4-D array of the one-of-k type target. Array Shape = [Image_Num, Image_Height, Image_Width, class_num]
    """
    
    target.astype('int64')
    one_hot_target = np.zeros([np.shape(target)[0], np.shape(target)[1], np.shape(target)[2], class_num])
    
    meshgrid_target = np.meshgrid(np.arange(np.shape(target)[1]), np.arange(np.shape(target)[0]), np.arange(np.shape(target)[2]))
    
    one_hot_target[meshgrid_target[1], meshgrid_target[0], meshgrid_target[2], target] = 1
    
    return one_hot_target
	
def compute_accuracy(
    xs,
    ys,
    is_training,
    prediction,
    v_xs,
    v_ys, 
    batch_size,
    sess
    ):
    
    """
    Compute the accuray of the semantic segmentation classification problem.
    
    Args:
    (1) xs          : A 4-D input placeholder tensor. Shape = [Batch_Size, Image_Height Image_Width, Image_Depth]
    (2) ys          : A target placeholder tensor. Shape = [Batch_Size, [output_shape]] (output_shape is user-defined)
    (3) is_training : A 1-D bool type placeholder tensor.
    (4) prediction  : An output placeholder tensor. Shape = Must be same as the ys shape.
    (5) v_xs        : An 4-D array of the data. Shape = [Image_Num, Image_Height Image_Width, Image_Depth]
    (6) v_ys        : An array of the target. Shape = [Image_Num, [target_shape]]
    (7) batch_size  : The data number in one batch.
    (8) sess        : Tensorflow Session.
    
    Returns:
    (1) total_accuracy_top1: Top1 Accuracy.
    """
    
    total_iter = int(len(v_xs)/batch_size)
    total_accuracy_top1 = 0
    
    for iter in range(total_iter):
        v_xs_part = v_xs[iter*batch_size:(iter+1)*batch_size, :]
        v_ys_part = v_ys[iter*batch_size:(iter+1)*batch_size, :]
        
        y_pre = sess.run( prediction, 
                          feed_dict={ xs: v_xs_part, 
                                      ys: v_ys_part,											
                                      is_training: False})
        
        
        prediction_top1 = np.argsort(-y_pre, axis=-1)[:, :, :, 0]
        
        # Calculate the Accuracy
        correct_prediction_top1 = np.equal(prediction_top1, np.argmax(v_ys_part, -1))
        accuracy_top1 = np.mean(correct_prediction_top1.astype(float))
        total_accuracy_top1 = total_accuracy_top1 + accuracy_top1
        
        # Save the result as array
        if iter==0:
            result = prediction_top1
        else:
            result = np.concatenate([result, prediction_top1], axis=0)
        
        
    total_accuracy_top1 = total_accuracy_top1 / total_iter
    
    return result, total_accuracy_top1


#============#
#    main    #
#============#
def main(argv=None):
    #---------------#
    #    Dataset    #
    #---------------#
    #*****************#
    #    test data    #
    #*****************#
    print("Loading testing data ... ")
    # read index
    test_data_index    = open(Dataset_Path + '/test.txt'      , 'r').read().splitlines()
    test_target_index  = open(Dataset_Path + '/testannot.txt' , 'r').read().splitlines()
    # read images
    test_data, test_target = CamVid_dataset_parser(Dataset_Path, test_data_index, test_target_index)
    # one-hot target
    test_target = one_hot(target = test_target, class_num = CLASS_NUM)
    print("\033[0;32mTest Data Number\033[0m  = {}" .format( np.shape(test_data)[0]   ))
    print("\033[0;32mTest Data Shape \033[0m  = {}" .format( np.shape(test_data)[1:4] )) # [Height, Width, Depth]
    
    #-------------------#
    #    Placeholder    #
    #-------------------#
    data_shape = np.shape(test_data)
    xs = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, data_shape[1], data_shape[2], data_shape[3]], name = 'input')
    ys = tf.placeholder(dtype = tf.float32, shape = [BATCH_SIZE, data_shape[1], data_shape[2], CLASS_NUM], name = 'output')
    lr = tf.placeholder(dtype = tf.float32)
    is_training = tf.placeholder(dtype = tf.bool)
    
    #-------------#
    #    Model    #
    #-------------#
    net = xs
    prediction = model_call( net, 
                             is_training, 
                             initializer = tf.contrib.layers.variance_scaling_initializer(), 
                             class_num   = CLASS_NUM, 
                             scope       = Student_ID)
    
    #-------------------------#
    #    Get All Variables    #
    #-------------------------#
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=Student_ID)
    # Model Size
    Model_Size = 0
    for iter, variable in enumerate(all_variables):
        Model_Size += reduce(lambda x, y: x*y, variable.get_shape().as_list())
        # See all your variables	
        """
        print(variable)
        """
    print("\033[0;36m=======================\033[0m")
    print("\033[0;36m Model Size\033[0m = {}" .format(Model_Size))
    print("\033[0;36m=======================\033[0m")
    
    #-------------#
    #    Saver    #
    #-------------#
    saver = tf.train.Saver()
    
    #---------------#
    #    Session    #
    #---------------#
    with tf.Session() as sess:
        # Initial all tensor variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Loading trained weights
        print("Loading trained weights ...")
        print(os.path.join(os.getcwd(), Student_ID + ".ckpt"))
        saver.restore(sess, os.path.join(os.getcwd(), Student_ID + ".ckpt"))
        
        print("Testing ... ")
        test_result, test_accuracy = compute_accuracy( xs, ys, is_training, prediction, 
                                                       v_xs       = test_data,
                                                       v_ys       = test_target, 
                                                       batch_size = BATCH_SIZE, 
                                                       sess       = sess)
                                                            
        print("\033[0;32mTesting Accuracy\033[0m = {}" .format(test_accuracy))
        
        # Save the test result to the image
        # If you use this code, you can see the predict images at 
        # ->  /nets/CamVid_Y_pre/testannot/
        # (Optional)
        """
        save_CamVid_result_as_image(
            result     = test_result, 
            path       = 'CamVid_Y_pre',
            file_index = test_target_index)
        """
			
if __name__ == "__main__":
    tf.app.run(main, sys.argv)
	
	
	
	
	
	
