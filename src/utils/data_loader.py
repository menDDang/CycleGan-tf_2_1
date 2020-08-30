import random
import os

from image import read_image
import tensorflow as tf


class DataLoader:
    def __init__(self, file_list_A, file_list_B, hp):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B

        # Set configuration
        self.apply_normalization = hp["data"]["apply_normalization"]
        self.input_width = hp["data"]["width"]
        self.input_height = hp["data"]["height"]
        self.input_channel_num = hp["data"]["channel_num"]

        # Create dataset
        img_shape = [self.input_width, self.input_height, self.input_channel_num]
        self.dataset = tf.data.Dataset.from_generator(
            self.batch_generator,
            output_types=(tf.float32, tf.float32)
            output_shapes=(img_shape, img_shape)
        )

    def batch_generator(self):
        while True:
            # Read image from domain A
            idx = random.randint(0, len(self.file_list_A))
            file_name = self.file_list_A[idx]
            if not os.path.isfile(file_name): 
                continue
            image_A = read_image(file_name, apply_normalization=self.apply_normalization)

            # Raad image from domain B
            idx = random.randint(0, len(self.file_list_B))
            file_name = self.file_list_B[idx]
            if not os.path.isfile(file_name): 
                continue
            image_B = read_image(file_name, apply_normalization=self.apply_normalization)

            # Yield batch
            yield image_A, image_B
            