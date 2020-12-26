import tensorflow as tf
import ray
import time

print(tf.__version__)
ray.init(num_gpus=1)

@ray.remote(num_gpus=1)
def test_gpu():
    tf.Variable(1)
    return ray.get_gpu_ids()

fn_id = test_gpu.remote()
gpu_ids = ray.get(fn_id)
print('gpu IDs:', ray.get_gpu_ids(), gpu_ids)
