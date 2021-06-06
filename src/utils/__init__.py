def disable_tensorflow_gpu():
    try:
        import tensorflow as tf
    except ImportError:
        print("Cannot import Tensorflow")
    else:
        tf.config.set_visible_devices([], 'GPU')
        print("Successfully disable all gpus")
