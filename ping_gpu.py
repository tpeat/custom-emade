import tensorflow as tf
# from tensorflow.compat.v1 import Session

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# I think this might have been moved
tf.compat.v1.disable_eager_execution()

def pingGPU():
    """Tries to execute a single multiply function on a GPU in order to
    determine if Tensorflow is GPU-enabled"""
    try:
        with tf.device('/gpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)

        # with tf.Session() as sess:
        sess = tf.compat.v1.Session()
        sess.run(c)
        print("Tensorflow is running with GPU optimization")
        sess.close()
    except Exception as e:
        print(e)
        print("Tensorflow was unable to run GPU optimization.")

pingGPU()

# from apple dev page
def verify():
    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)

# verify()
# this does enable GPU usage so it must work
