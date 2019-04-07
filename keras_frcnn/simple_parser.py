import cv2
import numpy as np
from functools import reduce
from training.tag_utils import parse_xml_tags

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line = line.strip().replace('pozicije', 'render')
            line_split = line.split('.')
            img_format = line_split[-1]

            xml_filepath = line.replace(f'.{img_format}', '.xml')

            with open(xml_filepath, 'r') as file:
                xml_content = reduce(lambda a, b : a + b, file.readlines())

            xml_data = parse_xml_tags(xml_content)

            class_name = xml_data['name']
            filename = xml_data['path']
            x1 = xml_data['x1']
            y1 = xml_data['y1']
            x2 = xml_data['x2']
            y2 = xml_data['y2']
            x3 = xml_data['x3']
            y3 = xml_data['y3']
            x4 = xml_data['x4']
            y4 = xml_data['y4']

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name,
                 'x1': int(float(x1)), 'y1': int(float(y1)),
                 'x2': int(float(x2)), 'y2': int(float(y2)),
                 'x3': int(float(x3)), 'y3': int(float(y3)),
                 'x4': int(float(x4)), 'y4': int(float(y4))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping

