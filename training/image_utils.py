import os
import cv2
import logging
import tag_creator as tag
"""
Module that holds all image utility classes and methods
"""


class Image:
    """
    Image class that enables easy path manipulation and data storage for all system parts while training.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    def __init__(self, path_to_image: str, database='Unknown', segmented=0):
        if path_to_image.startswith(('.', '..')):
            self.path = self._process_relative_path(path_to_image)
        elif path_to_image.startswith('/'):
            self.path = path_to_image
        else:
            raise Exception('Invalid file path')
        self.directory, self.file_name = self._extract_from_absolute_path(self.path)
        self.image_format = self._extract_format(self.file_name)
        self.image, self.depth, self.width, self.height = self._extract_from_image(self.path)
        logging.debug('Loaded image{abs_path=%s, directory=%s, file_name=%s, format=%s, '
                      'width=%d, height=%d, depth=%d}',
                      self.path, self.directory, self.file_name,
                      self.image_format, self.width, self.height, self.depth)

        self.database = database
        self.segmented = segmented

    @staticmethod
    def _extract_from_absolute_path(path: str):
        path_parts = path.split('/')
        file_name = path_parts.pop(-1)
        folder = path_parts.pop(-1)
        return folder, file_name

    @staticmethod
    def _extract_from_image(path_to_image: str):
        image = cv2.imread(path_to_image)
        height, width, depth = image.shape
        return image, depth, width, height

    @staticmethod
    def _process_relative_path(relative_path):
        path_parts = relative_path.split('/')
        cwd_parts = os.getcwd().split('/')

        for i in range(len(path_parts)):
            if path_parts[i] == '..':
                cwd_parts.pop(-1)
            elif path_parts[i] == '.':
                continue
            else:
                break
        cwd_parts += path_parts[i:]
        result = ''
        for part in cwd_parts:
            result += f'{part}/'
        return result[:-1]

    @staticmethod
    def _extract_format(file_name: str):
        return file_name.split('.')[-1]

    def generate_xml(self, objects):
        return tag.create_xml(self.directory, self.file_name, self.path,
                              self.width, self.height, objects,
                              self.database, self.depth, self.segmented)


if __name__ == '__main__':
    # example
    Image('../tablice/pozicije/0050.png')
