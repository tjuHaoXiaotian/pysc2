
import tensorflow as tf
import numpy as np
from collections import deque
from pysc2.lib import point
from pysc2.lib import transform

array = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,0,0,0,0,0,0]
                  ])

array2 = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,1,0,0,2,0,0],
    [0,0,0,0,0,0,0,0]
                  ])

def test():
    with tf.Session() as sess:
        one_hot_player_id = tf.one_hot(array, depth=3, on_value=1, off_value=0, dtype=tf.int32)
        print(sess.run(one_hot_player_id))
        channels = tf.unstack(one_hot_player_id, axis=2)
        channel1_2 = channels[1:]
        print(channel1_2)

        result = tf.stack([channels[1],channels[2],array2])
        print(result.get_shape())
        result = sess.run(result)

        print(result)

def test2():
    layer1 = np.copy(array)
    layer2 = np.copy(array)
    layer1[layer1==2] = 0
    layer2[layer2==1] = 0
    layer2[layer2==2] = 1
    layer3 = array2
    result = np.stack([layer1,layer2, layer3], axis=0)
    print(result)
class F(object):
    def __init__(self, placeholder, placeholder2):
        self.placeholder = placeholder
        # [
        # [1,2]
        # ]
        self.placeholder2 = placeholder2
        #[
        # [2],
        # [3]
        # ]
        # tf.truncated_normal([2, 3], mean=0., stddev=1.)

        self.w1 = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32) # 2,3
        # [
        # [1, 2, 3],
        # [4, 5, 6]
        # ]
        self.result1 = tf.matmul(self.placeholder, self.w1) # 1,3
        # [9, 12, 15]

        self.result2 = tf.matmul(self.placeholder2, self.result1) # 2,3
        # result 1:  自己计算
        # [
        #   [18, 24, 30]
        #   [27, 36, 45]
        # ]

        # result 2:  用了右边的赋值
        # [
        #   [9, 12, 15]
        #   [18, 24, 30]
        #   [27, 36, 45]
        # ]

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run([self.result2],feed_dict={
                self.placeholder:[[1,2]],
                self.placeholder2:[[2],[3]],
                self.result1:[[1,2,3]]
            })
            print(result)

def test_tf_where():
    pred = tf.placeholder(dtype=tf.bool, name='bool')
    x = tf.placeholder(tf.int32,[1,2],name="input")
    y = tf.cond(pred, lambda: x, lambda: x - 1)
    z = tf.where(pred, x, x - 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y1, z1 = sess.run([y, z], feed_dict={pred: True, x:[[1,1]]})
        y2, z2 = sess.run([y, z], feed_dict={pred: False, x:[[1,1]]})
        print(y1, z1)
        print(y2, z2)






class World_To_Screen(object):
    def __init__(self):
        self._feature_layer_screen_size = point.Point(84.0,84.0)
        self._camera_width_world_units = 24.0
        self._map_size = point.Point(64, 64)
        self._camera_center = point.Point(33.0,25.0)

        self._world_to_screen = transform.Linear(point.Point(1, -1),
                                             point.Point(0, self._map_size.y))
        self._screen_to_fl_screen = transform.Linear(
            self._feature_layer_screen_size / self._camera_width_world_units)
        self._world_to_fl_screen = transform.Chain(
            self._world_to_screen,
            self._screen_to_fl_screen,
            transform.Floor())
        self._update_camera(self._camera_center)

    def _update_camera(self, camera_center):
        """Update the camera transform based on the new camera center."""
        camera_radius = (self._feature_layer_screen_size /
                         self._feature_layer_screen_size.x *
                         self._camera_width_world_units / 2)
        center = camera_center.bound(camera_radius, self._map_size - camera_radius)
        self._camera = point.Rect(
            (center - camera_radius).bound(self._map_size),
            (center + camera_radius).bound(self._map_size))
        self._world_to_screen.offset = (-self._camera.bl *
                                        self._world_to_screen.scale)

    def word_to_screen(self):
        print(self._world_to_fl_screen.fwd_pt(point.Point(43.873016,19.841270)))
        return self._world_to_fl_screen.fwd_pt(point.Point(43.873016,19.841270))


# t = F(tf.placeholder(tf.float32,[1,2]),tf.placeholder(tf.float32,[2,1]))
# t.train()
# #
# test2()

# test_tf_where()

w = World_To_Screen()

w.word_to_screen()


