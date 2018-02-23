import tensorflow as tf
matrix1 = tf.constant([1,2,3,3,5,6],shape=[2,3])#构造矩阵(1,2,3
                                                #         3,5,6)

matrix2 = tf.constant([2,3,2,4,3,1,4,5,4],shape=[3,3])#构造矩阵(2,3,2
                                                      #         4,3,1
                                                      #         4,5,4)
product = tf.matmul(matrix1, matrix2)#矩阵乘法 a23*b33=c23 (1*2+2*4+3*4,1*3+2*3+3*5,1*2+2*1+3*4
                                     #                      3*2+5*4+6*4,3*3+5*3+6*5,3*2+5*1+6*4)
sess = tf.Session()
result=sess.run(product)
print(result)
