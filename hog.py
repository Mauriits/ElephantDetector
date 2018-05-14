import cv2


class HOGDescriptor:
    win_size = (96, 80) #(48, 40) #(64, 56) # #(144, 112)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    bins = 8

    def __init__(self):
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.bins)

    def calc_hog(self, image, location=None):

        if location is not None:
            if location[0] + self.win_size[0] > image.shape[1] or location[1] + self.win_size[1] > image.shape[0]:
                print("calc_hog: Window out of image range")

            x = location[1]
            y = location[0]
            image = image[x:(x + self.win_size[1]), y:(y + self.win_size[0])]

        else:
            image = cv2.resize(image, self.win_size)
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return self.hog.compute(image)