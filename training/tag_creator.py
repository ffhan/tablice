from typing import List
'''
Module created to handle easy XML (and possibly HTML) creation.
'''


class Tag:
    '''
    Tag defines a html/xml-like "tag" (<tag>...</tag>) and contains objects inside of itself.
    Can be used to easily construct dynamic XML or HTML files.
    '''

    def __init__(self, tag, *args):
        '''
        Defines an inner tag name and populates item list with arguments.

        Example:
            tag = Tag('test', 'testItem')
            print(tag)

        Output:
            <test>testItem</test>

        :param str tag: Tag name
        :param args: Items this tag holds.
        '''
        self.tag = tag
        self.items = list()
        for item in args:
            self.items.append(item)

    def add_item(self, tag):
        '''
        Adds a single item to the item list.
        :param tag: Any object that can be turned into a string (includes a Tag itself).
        :return:
        '''
        self.items.append(tag)

    def add_items(self, *args):
        '''
        Add a variable number of items.
        :param args: Items to be added to the item list.
        :return:
        '''
        for item in args:
            self.add_item(item)

    def __repr__(self):
        '''
        Constructs a Tag with all its' inner elements. It takes care of tabs and newlines.

        Example:

            t = Tag('test', 'testItem')
            print(t)
            tag_string = str(t)
            print(t)

        Output:

            <test>testItem</test>
            <test>testItem</test>

        :return str: Final string output of tag and its' contents.
        '''
        result = ''

        if len(self.items) <= 1 and type(self.items[0]) is not Tag:
            result = '<{0}>{1}</{0}>'.format(self.tag, self.items[0])
            return result

        result += '<{}>\n'.format(self.tag)
        for item in self.items:
            result += '\t' + str(item).replace('\n', '\n\t') + '\n'
        result += '</{}>'.format(self.tag)

        return result

    def save(self, path, filename):
        '''
        Saves the string output of a tag. This creates a needed output (HTML/XML).

        :param str path: File destination
        :param filename: File name. It HAS to have a format (.xml/.html/etc.)
        :return:
        '''

        with open(path + filename, 'w') as out:
            out.write(str(self))


class Coords:
    def __init__(self,
                 x1, y1, x2, y2, x3, y3, x4, y4):
        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

        self.x3 = x3
        self.y3 = y3

        self.x4 = x4
        self.y4 = y4


class SceneObject:
    def __init__(self, name, coords, pose='Unspecified', truncated=0,
                 difficult=0, occluded=0):
        self.name = name
        self.coords = coords
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.occluded = occluded


def create_xml(folder, filename, path, width, height, objects: List[SceneObject], database='Unknown', depth=3,
               segmented=0):
    xml = Tag('annotation',
              Tag('folder', folder),
              Tag('filename', filename),
              Tag('path', path),
              Tag('source',
                  Tag('database', database)),
              Tag('size',
                  Tag('width', width),
                  Tag('height', height),
                  Tag('depth', depth)),
              Tag('segmented', segmented))
    for obj in objects:
        xml.add_item(Tag('object',
                         Tag('name', obj.name),
                         Tag('pose', obj.pose),
                         Tag('truncated', obj.truncated),
                         Tag('difficult', obj.difficult),
                         Tag('corners',
                             Tag('x1', obj.coords.x1),
                             Tag('y1', obj.coords.y1),

                             Tag('x2', obj.coords.x2),
                             Tag('y2', obj.coords.y2),

                             Tag('x3', obj.coords.x3),
                             Tag('y3', obj.coords.y3),

                             Tag('x4', obj.coords.x4),
                             Tag('y4', obj.coords.y4)
                             ),
                         )
                     )
    return xml


def create_pascal_voc_xml(folder, filename, path,
                          width, height,
                          name,
                          xmin, ymin, xmax, ymax,
                          database='Unknown', segmented=0, pose='Unspecified',
                          truncated=0, difficult=0, depth=3):
    '''
    Defined structure for PASCAL VOC 2008. XML format (http://host.robots.ox.ac.uk/pascal/VOC/voc2008/htmldoc/).
    Currently supports only one detected object inside an image. This suffices for my needs, could be easily expanded
    if needed.

    Example:
        print(create_pascal_voc_xml('folder', 'test.jpg', 'folder\\test.jpg', 600, 300, 'registrationplate', 155, 9, 565, 133))

    :param str folder: Defines the folder that contains the specified image.
    :param str filename: Defines the filename for the specified image.
    :param str path: Defines the path to the specified image.
    :param int width: Defines the image width.
    :param int height: Defines the image height.
    :param str name: Describes the classification name of an object inside of the image
    :param int xmin: Object bounding box x-min coordinate
    :param int ymin: Object bounding box y-min coordinate
    :param int xmax: Object bounding box x-max coordinate
    :param int ymax: Object bounding box y-max coordinate
    :param str database: Name of the database from which the image was pulled
    :param int segmented: Image segmentation
    :param str pose: Object pose name
    :param int truncated: Object truncation
    :param int difficult: Object classfication difficulty
    :param int depth: Image depth (RGB for example has depth 3, RGBA has depth 4)
    :return: Tag object with all necessary information ready to be turned into a string (use str() method)
    '''
    return Tag('annotation',
               Tag('folder', folder),
               Tag('filename', filename),
               Tag('path', path),
               Tag('source',
                   Tag('database', database)),
               Tag('size',
                   Tag('width', width),
                   Tag('height', height),
                   Tag('depth', depth)),
               Tag('segmented', segmented),
               Tag('object',
                   Tag('name', name),
                   Tag('pose', pose),
                   Tag('truncated', truncated),
                   Tag('difficult', difficult),
                   Tag('bndbox',
                       Tag('xmin', xmin),
                       Tag('ymin', ymin),
                       Tag('xmax', xmax),
                       Tag('ymax', ymax)),

                   )
               )
