from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = "D:/pydemo/my_cnn_model/data/datasets/flower_photos/daisy/5547758_eea9edfd54_n.jpg"
path2 = "D:/pydemo/my_cnn_model/data/datasets/flower_photos/dandelion/7355522_b66e5d3078_m.jpg"
path3 = "D:/pydemo/my_cnn_model/data/datasets/flower_photos/roses/12240303_80d87f77a3_n.jpg"
path4 = "D:/pydemo/my_cnn_model/data/datasets/flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "D:/pydemo/my_cnn_model/data/datasets/flower_photos/tulips/11746080_963537acdc.jpg"

flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

w=100
h=100
c=3

def read_one_image(path):
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

    saver = tf.train.import_meta_graph('D:/pydemo/my_cnn_model/data/model/flower/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('D:/pydemo/my_cnn_model/data/model/flower/'))

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
        print("第",i+1,"朵花预测:"+flower_dict[output[i]])