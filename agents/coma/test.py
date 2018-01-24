import tensorflow as tf

s = tf.placeholder(dtype=tf.float32, shape=[1,3])
en = tf.nn.softmax_cross_entropy_with_logits(labels=s, logits=s)
with tf.Session() as sess:
    # print(sess.run(tf.reduce_sum(s, axis=1, keep_dims=True),feed_dict={
    #     s: [[1,2,3],[4,5,6]]
    # }))
    print(sess.run(en, feed_dict={
        s: [[0.9, 0.1, 0]]
    }))