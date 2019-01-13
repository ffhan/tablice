from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from scipy.stats import itemfreq
from transforms import RGBTransform
import cv2, os, math, random, urllib.request, datetime, tag_creator
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import threading

urls = sio.loadmat("SUN397_urls.mat")


def rand_char():
    c = chr(random.randint(65, 90))

    if c == 'W':
        return rand_char()

    return c


def rand_int_char():
    return chr(random.randint(48, 57))


def create_char_pair():
    return rand_char() + rand_char()


def _create_chars(func, min_len=1, max_len=2):
    chars = ''
    for i in range(random.randint(min_len, max_len)):
        chars += func()  # use a delegate function to abstract string creation rules.
    return chars


def create_chars():
    return _create_chars(rand_char, 1, 2)


def create_num_chars():
    return _create_chars(rand_int_char, 3, 4)


def registration():
    return create_char_pair() + '  ' + create_num_chars() + '-' + create_chars()


def visit_links(func, skip = 0, limit = 0, report = False):
    # tells us if limit has been set in the beginning, otherwise we wouldn't be able to
    # tell when limit hits 0.
    limited = limit > 0

    cap = limit

    count = 0

    end = False

    for arr in urls['SUN'][0]:
        if end:
            return
        for link in arr[2][0]:
            if skip > 0:
                skip -= 1
                continue
            if limit > 0:
                limit -= 1
            elif limit <= 0 and limited:
                end = True
                break
            if not func(link[0]):
                limit += 1
                count += 1
            if report: # implicitly asserted that limit is <= 0.
                print(100 * (cap - limit) / cap)

            print("PID {} has generated {} images!".format(threading.current_thread(), count))

def visit_links_async(func, num_threads = 5, skip = 0, limit = 0):

    images_per_thread = limit / num_threads

    threads = [threading.Thread(target = visit_links, args = (func, skip + index * images_per_thread, images_per_thread)) for index in range(num_threads)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def open_link(link):
    def inner(img):
        cv2.imshow('Image from link', img)
        if cv2.waitKey() & 0xff == 27: quit()

    _link(link, inner)

def open_pil_link(link):
    def inner(img):
        pil_img = Image.fromarray(img)
        pil_img.show()

    _link(link, inner)

def get_link(link):
    def inner(img):
        pil_img = Image.fromarray(img)
        return pil_img

    return _link(link, inner)

def _link(link, func):
    req = urllib.request.urlopen(link)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    return func(img)

def find_rotation_coeffs(theta, x0, y0):
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array([ct, -st, x0*(1-ct) + y0*st, st, ct, y0*(1-ct)-x0*st,0,0])

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def get_perspective_point(x, y, coefs):

    new_x = round((coefs[0] * x + coefs[1] * y + coefs[2]) / (coefs[6] * x + coefs[7] * y + 1))
    new_y = round((coefs[3] * x + coefs[4] * y + coefs[5]) / (coefs[6] * x + coefs[7] * y + 1))

    return new_x, new_y

def create_perspective_transform_matrix(src, dst):
    """ Creates a perspective transformation matrix which transforms points
        in quadrilateral ``src`` to the corresponding points on quadrilateral
        ``dst``.

        Will raise a ``np.linalg.LinAlgError`` on invalid input.
        """
    # See:
    # * http://xenia.media.mit.edu/~cwren/interpolator/
    # * http://stackoverflow.com/a/14178717/71522
    in_matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(af).reshape(8)

def get_point(pt, transform_matrix):

    res = np.dot(transform_matrix.reshape((3, 3)), ((pt[0], ), (pt[1], ), (1, )))
    res = res / res[2]
    return int(round(res[0][0])), int(round(res[1][0]))

def dominant_color(img):
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]

    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    return palette[np.argmax(itemfreq(labels)[:, -1])]


class Plate:
    def __init__(self):

        dt = datetime.datetime.utcnow()
        date = dt.date()
        micros = int(dt.microsecond)

        self.dtset = ['train', 'test'][random.randint(1, 10) // 10]
        self.text = registration()

        self.city, self.suffix = self.text.split('  ')

        self.inner_padding_x = 73
        self.inner_padding_y = 17

        self.name = '{}_{}_{}_{}'.format(self.dtset, self.text, micros, random.randint(0, 1000))

        self.path = 'images\\{}'.format(self.dtset)

        self.size_x = 750
        self.size_y = 400
        # self.padding_x = 60
        # self.padding_y = 60
        #
        # self.outer_padding_x = 30
        # self.outer_padding_y = 30

        # define possible perspective rotations
        # rot_x = 30
        # rot_y = 20
        rot_ran = 30  # rotation range

        #todo: random inner color, random outer color
        # define colors
        background = (0, 0, 0, 0)
        plate_edge = (0x15, 0x15, 0x15, 255)
        # outer_col = (0xF3, 0x2E, 0x2E, 255)
        outer_col = self.color_randomize(255)
        plate_inner = (0xFF, 0xFF, 0xFF, 255)
        font_col = (0x15, 0x15, 0x15, 255)

        frame_width = 10

        # define objects needed for drawing text
        self.image = Image.new("RGBA", (self.size_x, self.size_y), color = background)
        # self.image = Image.open('plate.png', 'RGBA')

        image = Image.open('plate_small.png').convert('RGBA')
        # background = Image.new('RGBA', self.image.size, (255, 255, 255))

        self.padding_x = round((self.image.size[0] - image.size[0]) / 2)
        self.padding_y = round((self.image.size[1] - image.size[1]) / 2)

        rot_x = round(self.padding_x / 4)
        rot_y = rot_x

        self.image.paste(image, (self.padding_x, self.padding_y), image)

        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.truetype("tahoma", 75)

        # self.size_x = self.image.size[0]
        # self.size_y = self.image.size[1]

        # write centered text
        width, height = self.font.getsize(self.suffix)
        width = (310 - width) / 2 + 150

        # drawing outer color
        # self.draw_rectangle(self.outer_padding_x, self.outer_padding_y, outer_col)

        #drawing plate edge
        # self.draw_rectangle(self.padding_x, self.padding_y, plate_edge)
        #
        # self.draw_rectangle(self.padding_x + frame_width, self.padding_y + frame_width, plate_inner)

        self.draw.text((self.padding_x + self.inner_padding_x, self.padding_y + self.inner_padding_y), self.city, fill=font_col, font=self.font)
        self.draw.text((self.padding_x + self.inner_padding_x + width, self.padding_y + self.inner_padding_y), self.suffix, fill=font_col, font=self.font)

        self.image.save('clean_table.png')

        original = [[self.padding_x, self.padding_y],
                    [self.size_x - 1 - self.padding_x, self.padding_y],
                    [self.size_x - 1 - self.padding_x, self.size_y - 1 - self.padding_y],
                    [self.padding_x, self.size_y - 1 - self.padding_y]
                    ]

        rotation = [[self.padding_x + random.randint(-rot_x, rot_x), self.padding_y + random.randint(-rot_y, rot_y)],
                    [self.size_x - 1 - self.padding_x - random.randint(-rot_x, rot_x), self.padding_y + random.randint(-rot_y, rot_y)],
                    [self.size_x - 1 - self.padding_x - random.randint(-rot_x, rot_x), self.size_y - 1 - self.padding_y - random.randint(-rot_y, rot_y)],
                    [self.padding_x + random.randint(-rot_x, rot_x), self.size_y - 1 - self.padding_y - random.randint(-rot_y, rot_y)]]

        coefs = cv2.getPerspectiveTransform(np.float32(original), np.float32(rotation))

        im = np.array(self.image)
        # noise = cv2.randn(im2, (m,m,m,0), (sgm,sgm,sgm,0))
        # im = im + noise
        im = cv2.warpPerspective(im, coefs, (self.size_x, self.size_y))


        # self.image.show()

        self.image = Image.fromarray(im, mode = 'RGBA')
        self.draw = ImageDraw.Draw(self.image)

        self.image.save('warped_plate.png')

        random_function = lambda limit, sigma : abs(limit - abs(limit - random.normalvariate(limit, sigma)))

        self.image = ImageEnhance.Brightness(self.image).enhance(random_function(1, 0.01))
        self.image = ImageEnhance.Contrast(self.image).enhance(random_function(1, 0.3))
        self.image = ImageEnhance.Sharpness(self.image).enhance(random.uniform(0.5, 2))

        # self.image.show()

        #coefs = create_perspective_transform_matrix(original, rotation)
        #coefs = find_rotation_coeffs(random.uniform(- math.pi / 3, math.pi / 3), self.size_x / 2, self.size_y / 2)
        '''
        self.image = self.image.transform((self.size_x, self.size_y),
                                          Image.PERSPECTIVE,
                                          coefs
                                          )

        self.draw = ImageDraw.Draw(self.image)
        '''
        #warp1, warp2 = get_perspective_point(original[0][0], original[0][1], coefs), get_perspective_point(original[3][0], original[3][1], coefs)
        (wx1, wy1), (wx2, wy2), (wx3, wy3), (wx4, wy4) = get_point((original[0][0], original[0][1]), coefs), get_point((original[1][0], original[1][1]), coefs), \
                                                         get_point((original[2][0], original[2][1]), coefs), get_point((original[3][0], original[3][1]), coefs)
        #print(get_perspective_point(original[0][0], original[0][0], coefs))

        self.xmin = min(wx1, wx2, wx3, wx4)
        self.ymin = min(wy1, wy2, wy3, wy4)
        self.xmax = max(wx1, wx2, wx3, wx4)
        self.ymax = max(wy1, wy2, wy3, wy4)

        # print(self.xmin, self.ymin)
        # print(self.xmax, self.ymax)
        # self.draw = ImageDraw.Draw(self.image)
        # self.draw.rectangle((self.xmin, self.ymin, self.xmax, self.ymax))

        #self.image.show()

    def draw_rectangle(self, padding_x, padding_y, color):

        self.draw.rectangle(
            [
                (padding_x, padding_y),
                (self.size_x - padding_x),
                (self.size_y - padding_y)
            ],
            color
        )

    def color_randomize(self, alpha):

        def rnd_col():
            return random.randint(0, 255)

        return rnd_col(), rnd_col(), rnd_col(), alpha

    def _center(self, width, height):
        return (self.size_x - width) / 2, (self.size_y - height) / 2 - 5

    def __repr__(self):
        return self.text

    def show(self):
        self.image.show()

    def save(self):
        self.image.save("{}\\{}.png".format(self.path, self.name))

    def paste(self, img):
        # perc = random.uniform(0.05, 0.05)
        new_width = random.uniform(45, 60)
        new_height = new_width / self.size_x * self.size_y
        x, y = map(round, [new_width, new_height])

        width, height = img.size

        if width > 600:
            ratio = height / width
            width = 600 #set width to 600, recalculate height as to maintain the aspect ratio

            height = round( ratio * width )

            img = img.resize((width, height), Image.ANTIALIAS)

        # x = round(width * perc)
        # y = round(height * perc)

        xperc = x / self.size_x
        yperc = y / self.size_y
        # print(x,y,width,height)
        new_x = random.randint(0, width - x)
        new_y = random.randint(0, height - y)

        minx = round(self.xmin * xperc) + new_x - 1
        miny = round(self.ymin * yperc) + new_y - 1
        maxx = round(self.xmax * xperc) + new_x + 1
        maxy = round(self.ymax * yperc) + new_y + 1

        self.xml = tag_creator.create_pascal_voc_xml(self.dtset, self.name + '.png', '{}\\{}'.format(self.path, self.name),
                                                width, height, 'registrationplate', minx, miny, maxx, maxy)


        # get dominant colour of a background image, add it to the plate (simulate lighting)
        # dominant_tint = dominant_color(np.array(img))
        # print(dominant_tint)
        # self.image = RGBTransform().mix_with(dominant_tint, factor=0.1).applied_to(self.image)

        plate = self.image.resize((x,y), Image.ANTIALIAS)
        #TODO: make sure the result image is in RGB, not RGBA (alpha channel redundant at this point.

        img.paste(plate, (new_x, new_y), plate)

        # drw = ImageDraw.Draw(img)
        # drw.rectangle((minx, miny, maxx, maxy))
        return img
        #img.save("test_plates\\test.png")

    def save_xml(self):
        self.xml.save('{}\\'.format(self.path), '{}.xml'.format(self.name))

    @staticmethod
    def worsen(image):
        width, height = image.size
        ratio = width / height
        x = round(width * 2 / 3)
        y = round(x * ratio)
        img = image.resize((x, y), Image.ANTIALIAS)
        return img.resize((width, height), Image.ANTIALIAS)
'''
for arr in urls['SUN'][0]:
    for link in arr[2][0]:
        print(link[0])


'''
#visit_links(print, skip = 1000, limit = 8) # - how to use it in visit_links

#open_pil_link('http://labelme.csail.mit.edu/Images/users/antonio/static_sun_database/y/youth_hostel/sun_drwspcgwoiyirann.jpg')
# urls[database][must be 0][category][.jpg links (must be 2)][must be 0][choose link][must be 0]
# print(urls['SUN'][0][50][2][0][10][0])
'''
for i in range(150):
    plate = Plate()
    plate.save()
'''

def get_components(link):
    plate = Plate()
    ##plate.paste(get_link('http://labelme.csail.mit.edu/Images/users/antonio/static_sun_database/y/youth_hostel/sun_drwspcgwoiyirann.jpg'))

    im = get_link(link)

    return plate, im

def final_step(image, plate):
    return plate.worsen(image).convert('RGB')

def save_img(link):

    plate, im = get_components(link)

    if im.size[0] < 600:
        return False

    img = plate.paste(im)

    # final_step(img, plate).save('{}\\{}.png'.format(plate.path, plate.name))
    img.save('{}\\{}.png'.format(plate.path, plate.name))
    plate.save_xml()

    return True

def show_image(link):

    plate, im = get_components(link)

    if im.size[0] < 600:
        return False

    img = plate.paste(im)

    # final_step(img, plate).show()
    img.show()

    return True

if __name__ == '__main__':
    visit_links(save_img, limit=1)
    # visit_links(show_image, limit = 1)
# visit_links(show_image, limit = 1)
# visit_links(save_img, limit = 1, report=True)
