from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy import misc

import sys
import os
import time
import pdb


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
        
        H_resize = 224
        W_resize = 224
        data_tmp   = scipy.misc.imresize(data_tmp,   (H_resize, W_resize))
        target_tmp = scipy.misc.imresize(target_tmp, (H_resize, W_resize))
        
        
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

# (Optinal Function)
def shuffle_data(
    data, 
    target
    ):
    
    """
    Shuffle the data.
    
    Args:
    (1) data   : An 4-D array of the data. Shape = [Image_Num, Image_Height Image_Width, Image_Depth]
    (2) target : An array of the target. Shape = [Image_Num, [target_shape]]
    Returns:
    (1) shuffle_data   : Shuffle data. Shape = [Image_Num, Image_Height Image_Width, Image_Depth]
    (2) shuffle_target : Shuffle target. Shape = [Image_Num, [target_shape]]
    """
    
    shuffle_index = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffle_index)
    
    shuffle_data   = data  [shuffle_index, :, :, :]
    shuffle_target = target[shuffle_index, :, :, :]
    
    return shuffle_data, shuffle_target

def per_class_accuracy(
    prediction, 
    batch_ys
    ):
    
    """
    Show the accuracy of each class.
    
    Args:
    (1) prediction : An output placeholder tensor. Shape = Must be same as the batch_ys shape.
    (2) batch_ys   : An array of the target. Shape = [Batch_Size, [target_shape]]
    
    Returns:
    (None)
    """
    
    print("Per Class Accuracy")
    [BATCH, HEIGHT, WIDTH, CLASS_NUM] = np.shape(batch_ys)
    correct_num = np.zeros([CLASS_NUM, 1])
    total_num = np.zeros([CLASS_NUM, 1])
    
    print_per_row = 10
    cn = np.zeros([print_per_row], np.int32)
    tn = np.zeros([print_per_row], np.int32)
    
    for i in range(CLASS_NUM):
        y_tmp = np.equal(np.argmax(batch_ys, -1), i)
        p_tmp = np.equal(np.argmax(prediction, -1), i)
        total_num = np.count_nonzero(y_tmp)
        zeros_num = np.count_nonzero( (p_tmp+y_tmp) == 0)
        correct_num = np.count_nonzero(np.equal(y_tmp, p_tmp)) - zeros_num
        if total_num == 0:
            accuracy = -1
        else:
            accuracy = float(correct_num) / float(total_num)
        
        if CLASS_NUM <= 15:
            print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=correct_num, target=total_num))
        else:
            iter = i%print_per_row
            cn[iter] = correct_num
            tn[iter] = total_num
            if i%print_per_row==0:
                print("    Class{Iter}	: {predict} / {target}".format(Iter = i, predict=np.sum(cn), target=np.sum(tn)))

def save_CamVid_result_as_image(
    result,
    path, 
    file_index
    ):
    
    """
    Save the CamVid result to the image for visualing.
    
    Args:
    (1) result     : Image to be Save. An 4D array. Shape=[Image_Num, Image_Height, Image_Width, Image_Depth]
    (2) Path       : Path to save the image.
    (3) file_index : Index of each images.
    
    Returns:
    (None)
    """
    
    # -- Color the result --
    print("Coloring the results ... ")
    #***************************************#
    #	class0 : (	128 	128 	128	)	#
    #	class1 : (	128 	0 		0	)	#
    #	class2 : (	192 	192 	128	)	#
    #	class3 : (	128 	64 		128	)	#
    #	class4 : (	0 		0 		192	)	#
    #	class5 : (	128 	128 	0	)	#
    #	class6 : (	192 	128 	128	)	#
    #	class7 : (	64 		64 		128	)	#
    #	class8 : (	64 		0 		128	)	#
    #	class9 : (	64 		64 		0	)	#
    #	class10 : (	0		128 	192	)	#
    #	class11 : (	0		0		0	)	#
    #***************************************#
    shape = np.shape(result)
    RGB = np.zeros([shape[0], shape[1], shape[2], 3], np.uint8)
    for i in range(shape[0]):
        for x in range(shape[1]):
            for y in range(shape[2]):
                if result[i][x][y] == 0:
                    RGB[i][x][y][0] = np.uint8(128)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 1:
                    RGB[i][x][y][0] = np.uint8(128) 
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(0) 
                elif result[i][x][y] == 2:
                    RGB[i][x][y][0] = np.uint8(192)
                    RGB[i][x][y][1] = np.uint8(192)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 3:
                    RGB[i][x][y][0] = np.uint8(128)
                    RGB[i][x][y][1] = np.uint8(64)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 4:
                    RGB[i][x][y][0] = np.uint8(0)
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(192)
                elif result[i][x][y] == 5:
                    RGB[i][x][y][0] = np.uint8(128)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(0)
                elif result[i][x][y] == 6:
                    RGB[i][x][y][0] = np.uint8(192)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 7:
                    RGB[i][x][y][0] = np.uint8(64)
                    RGB[i][x][y][1] = np.uint8(64)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 8:
                    RGB[i][x][y][0] = np.uint8(64)
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(128)
                elif result[i][x][y] == 9:
                    RGB[i][x][y][0] = np.uint8(64)
                    RGB[i][x][y][1] = np.uint8(64)
                    RGB[i][x][y][2] = np.uint8(0)
                elif result[i][x][y] == 10:
                    RGB[i][x][y][0] = np.uint8(0)
                    RGB[i][x][y][1] = np.uint8(128)
                    RGB[i][x][y][2] = np.uint8(192)
                elif result[i][x][y] == 11:
                    RGB[i][x][y][0] = np.uint8(0)
                    RGB[i][x][y][1] = np.uint8(0)
                    RGB[i][x][y][2] = np.uint8(0)
    
    # -- Save the result into image --
    # Create the directory if it is not exist
    if not os.path.exists(path):
        print("\033[1;35;40m%s\033[0m is not exist!" %path)
        os.mkdir(path)
        print("\033[1;35;40m%s\033[0m is created" %path)
        
    for i, target in enumerate(RGB):
        # Create the directory if it is not exist
        dir = file_index[i].split('/')
        dir_num = len(dir)
        for iter in range(1, dir_num-1):
            if not os.path.exists(path + '/' + dir[iter]):
                print("\033[1;35;40m%s\033[0m is not exist!" %path + '/' + dir[iter])
                os.mkdir(path + '/' + dir[iter])
                print("\033[1;35;40m%s\033[0m is created" %path + '/' + dir[iter])
        
        # save
        scipy.misc.imsave(path + file_index[i], target)
