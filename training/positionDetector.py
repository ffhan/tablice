import os
import sys

import cv2
import numpy as np
import argparse
import logging
import tag_creator as tag
import image_utils as img

from os import listdir

# argument parser helper text
PROGRAM_DESCRIPTION = 'Detects corners of a table and computes its\' center and returns it in a nice format.'
FILES_HELP = '''List all files you want the program to detect. 
                Files have to be exclusively directories or exclusively images.'''
FILE_TYPE_HELP = 'Indicates that specified files are images, not directories.'
RECURSIVE_HELP = '''Indicates image searching in subdirectories of specified directories. 
                    Ignored when images_provided flag is present.'''
FORMATS_HELP = 'Indicates which image formats should be filtered (those that are not specified will be ignored).' \
               'Example input: --formats png jpeg jpg'
DESTINATION_HELP = 'Indicates where processed files should be stored. ' \
                   'If destination isn\'t provided it will be stored in the same directory where the original file is located.'

# dot colours for corner detection
GREEN_DOT = (0, 146, 0)
RED_DOT = (213, 0, 0)
BLUE_DOT = (0, 0, 230)
VIOLET_DOT = (213, 0, 230)

GREEN_BOUNDS = [(0, 100, 0), (25, 150, 25)]
RED_BOUNDS = [(200, 0, 0), (250, 0, 0)]
BLUE_BOUNDS = [(0, 0, 100), (20, 20, 250)]
VIOLET_BOUNDS = [(180, 0, 180), (250, 0, 250)]

supported_formats = ['png', 'jpg', 'jpeg']
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
argument_parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION)

argument_parser.add_argument('files', metavar='N', type=str, nargs='*', help=FILES_HELP)
argument_parser.add_argument('-f', dest='images_provided', const=True, nargs='?', default=False, help=FILE_TYPE_HELP)
argument_parser.add_argument('-r', dest='recursive_search', const=True, nargs='?', default=False, help=RECURSIVE_HELP)
argument_parser.add_argument('--dest', dest='destination', type=str, nargs='?', help=DESTINATION_HELP)
argument_parser.add_argument('--formats', metavar='N', type=str, nargs='+', help=FORMATS_HELP)

args = argument_parser.parse_args()


def get_all_images_from_directory(directory):
    if directory[-1] != '/':
        directory += '/'
    images = []
    for f in listdir(directory):
        if '.' in f:
            file_format = f.split('.')[-1]
            if file_format.lower() in supported_formats:
                images.append(directory + f)
        else:
            if recursive_search:
                images.append(get_all_images_from_directory(directory + f))
            else:
                continue
    return images


def return_all_images():
    if images_provided:
        return files
    images = []
    for directory in files:
        images += get_all_images_from_directory(directory)
    return list(map(lambda im: img.Image(im), images))


def find_corners(image_object, *boundaries):
    final_points = []
    image = image_object.image
    for boundary in boundaries:
        # loop over the boundaries
        lower, upper = boundary
        logging.info("Boundaries: %s %s", lower, upper)
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        mask = np.array(mask)
        points = np.where(mask > 0)
        logging.debug("Found points: %s", points)

        central_point = [0, 0]
        x_points, y_points = points
        if x_points.size in [None, 0]:
            raise Exception('Didn\'t find a point!')
        for i in range(x_points.size):
            central_point[0] += x_points[i]
            central_point[1] += y_points[i]
        central_point[0] /= x_points.size
        central_point[1] /= y_points.size

        central_point = tuple(map(lambda t: int(round(t)), central_point))

        logging.debug("Central point: %s", central_point)
        final_points.append(central_point)
    return final_points


def find_center(corners):
    center = [0, 0]
    counter = 0
    for x, y in corners:
        center[0] += x
        center[1] += y
        counter += 1
    return tuple(map(lambda t: round(t / counter), center))


def process_image(image):
    corners = find_corners(image,
                           RED_BOUNDS, BLUE_BOUNDS,
                           VIOLET_BOUNDS, GREEN_BOUNDS)
    center = find_center(corners)
    return corners, center


def generate_scene_object(image, corners):
    obj = tag.SceneObject('registration_plate',
                          tag.Coords(corners[0][1],
                                     corners[0][0],

                                     corners[1][1],
                                     corners[1][0],

                                     corners[2][1],
                                     corners[2][0],

                                     corners[3][1],
                                     corners[3][0]))
    return tag.create_xml(image.directory, image.file_name,
                          image.path, image.width, image.height, [obj])


def show_points_on_image(image_path, corners, center):
    image = cv2.imread(image_path)
    corners.append(center)
    for x, y in map(lambda i: (int(i[0]), int(i[1])), corners):
        image[x, y] = (255, 255, 255)
    cv2.imshow('image', image)
    cv2.waitKey(0)


def save_scene_object_to_xml(image):
    global destination
    corners, center = process_image(image)
    xml = generate_scene_object(image, corners)

    if destination is None:
        dest = image.path.replace(image.image_format, 'xml')
    else:
        dest = os.path.realpath(destination)
        if dest[-1] != '/':
            dest += '/'
        dest += image.file_name
        dest = dest.replace(image.image_format, 'xml')

    with open(dest, 'w') as file:
        file.write(repr(xml))


images_provided = args.images_provided
if images_provided:
    args.recursive_search = False
files = args.files
recursive_search = args.recursive_search
destination = args.destination if args.destination else None
if args.formats is not None:
    supported_formats = list(map(lambda t: t.lower(), args.formats))
all_images = return_all_images()

if __name__ == '__main__':
    logging.debug('Run configuration: %s', args)
    logging.info('Discovered %d images: %s', len(all_images), all_images)
    for img in all_images:
        save_scene_object_to_xml(img)

