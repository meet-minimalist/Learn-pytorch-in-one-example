
import imgaug.augmenters as iaa


class PaddedResize(object):

    def __init__(self, size=224):
        self.seq_resize = iaa.Sequential([
                    iaa.CropToSquare(position='center'),
                    iaa.Resize({'height': size, 'width':'keep-aspect-ratio'})
                ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = self.seq_resize(image=image)

        return {'image':image, 'label':label}
