from face_list import face_fname
from PIL import Image
import numpy as np
import sys

data_dir = '../data/'


def load_face_data(face_id_list,scale,test_amount):
    num_classes=len(face_id_list)

    bad_image_list = []
    face_data = []
    face_class = []
    face_data_test = []
    face_class_test = []

    img_width=168
    img_height=192
    for (index,face_id) in enumerate(face_id_list):

        face_data_current = []
        face_class_current = []

        face_str = 'B' + ("%02d"%face_id)
        cur_class = np.zeros((1,num_classes))
        cur_class[0,index] = 1.0
        for f in face_fname:
            f_new = data_dir + f.replace('BXX',face_str)
            try:
                img = Image.open(f_new)
                sys.stdout.write(".")
            except IOError:
                bad_image_list.append(f_new)
                sys.stdout.write('X')
                continue
            img = img.resize((int(img_width/scale),int(img_height/scale)),Image.ANTIALIAS)
            img_arr = np.array(img)
            img_arr = img_arr.reshape(1,(int(img_width/scale)*int(img_height/scale)))
            face_data_current.append(img_arr)
            face_class_current.append(np.copy(cur_class))
        #cast to proper numpy array
        face_data_current = np.array(face_data_current)
        face_data_current = face_data_current.reshape(face_data_current.shape[0],face_data_current.shape[2])

        face_class_current = np.array(face_class_current)
        face_class_current = face_class_current.reshape(face_class_current.shape[0],face_class_current.shape[2])

        #shuffle data
        rng_state = np.random.get_state();
        np.random.shuffle(face_data_current)
        np.random.set_state(rng_state)
        np.random.shuffle(face_class_current)

        for i in range(test_amount):
            face_data_test.append(face_data_current[i,:])
            face_class_test.append(face_class_current[i,:])

        for i in range(test_amount+1,face_data_current.shape[0]):
            face_data.append(face_data_current[i,:])
            face_class.append(face_class_current[i,:])

    #cast to proper numpy array
    face_data = np.array(face_data)
    #face_data = face_data.reshape(face_data.shape[0],face_data.shape[2])

    face_class = np.array(face_class)
    #face_class = face_class.reshape(face_class.shape[0],face_class.shape[2])

    face_data_test = np.array(face_data_test)
    #face_data_test = face_data_test.reshape(face_data_test.shape[0],face_data_test.shape[2])

    face_class_test = np.array(face_class_test)
    #face_class_test = face_class_test.reshape(face_class_test.shape[0],face_class_test.shape[2])
    
    #normalize
    face_data = np.float64(face_data)
    face_data_test = np.float64(face_data_test)

    f_mean = np.mean(np.append(face_data,face_data_test,axis=0))
    f_std = np.std(np.append(face_data,face_data_test,axis=0))

    face_data = face_data - f_mean
    face_data = face_data/f_std
    face_data_test = face_data_test - f_mean
    face_data_test = face_data_test/f_std

    face_data = np.asarray(face_data,np.float32)
    face_class = np.asarray(face_class,np.float32)
    face_data_test = np.asarray(face_data_test,np.float32)
    face_class_test = np.asarray(face_class_test,np.float32)

    return (face_data,face_class,face_data_test,face_class_test,bad_image_list)

if __name__ == '__main__':

    #there is no 14
    face_id_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18, \
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    (face_data,face_class,face_data_test,face_class_test,bad_image_list) = load_face_data(face_id_list,p['scale'],5)

    print("bad images: ")
    for f in bad_image_list:
        print(f)

    print(face_data.shape)
    print(face_class.shape)
    print(face_data_test.shape)
    print(face_class_test.shape)

    print(np.sum(face_class,axis=0))

