from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = "D:/pydemo/my_cnn_model_animal/data/datasets/animal_photos/bird/002.jpg"
path2 = "D:/pydemo/my_cnn_model_animal/data/datasets/animal_photos/cat/102.jpg"
path3 = "D:/pydemo/my_cnn_model_animal/data/datasets/animal_photos/dog/202.jpg"
path4 = "D:/pydemo/my_cnn_model_animal/data/datasets/animal_photos/duck/303.jpg"
path5 = "D:/pydemo/my_cnn_model_animal/data/datasets/animal_photos/rabbit/404.jpg"

animal_dict = {0:'bird',1:'cat',2:'dog',3:'duck',4:'rabbit'}

w=100
h=100
c=3

def read_one_image(path):  #训练数据中含有大量的三维图片，在训练中这类图片直接跳过了，但执行模型的时候拿出来的是一张，所以要注意
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)




with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)

    saver = tf.train.import_meta_graph('D:/pydemo/my_cnn_model_animal/data/model/animal/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('D:/pydemo/my_cnn_model_animal/data/model/animal/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"动物预测:"+animal_dict[output[i]])