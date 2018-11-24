import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA = '../data/flower_photos'
OUTPUT_FILE = './flower_processed_data.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_list(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获取一个子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print('processing:', dir_name)
        i = 0
        # 处理图片数据
        for file_name in file_list:
            i += 1
            # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()    # 读取图像
            image = tf.image.decode_jpeg(image_raw_data)                # 图像解码,三维矩阵[w, h, 3]
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)       # 改变图像数据类型
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            # 随机划分数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            if i % 200 == 0:
                print(i, 'images processed')
        current_label += 1
    # 将训练数据随机打乱以获得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    print('finish process data')
    return np.asarray([training_images, training_labels, validation_images, validation_labels,
                       testing_images, testing_labels])


# 数据整理主函数
def main():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session() as sess:
        processed_data = create_image_list(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        print('Prepare write to file')
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()
