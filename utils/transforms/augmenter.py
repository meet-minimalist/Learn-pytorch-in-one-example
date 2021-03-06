
import imgaug.augmenters as iaa


class Augmenter(object):

    def __init__(self):
        
        self.seq_aug = iaa.SomeOf((1, 2), [
                        iaa.OneOf([
                            #iaa.Dropout(p=(0.1, 0.2)),
                            iaa.CoarseDropout(0.05, size_percent=0.1, per_channel=0.5),
                            iaa.SaltAndPepper(0.05),
                            iaa.CoarseSaltAndPepper(0.03, size_percent=(0.1, 0.2))
                            ]),
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=(0.5, 1.0)),
                            iaa.MedianBlur(k=(3, 5)),
                            iaa.MotionBlur(k=5, angle=[-45, 45])
                            ]),
                        iaa.OneOf([
                            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                            iaa.Grayscale(alpha=(0.5, 1.0)),
                            iaa.AddToHueAndSaturation((-50, 50))
                            ]),
                        iaa.OneOf([
                            iaa.Fliplr(0.5),
                            iaa.Affine(scale=(0.8, 1.2)),
                            iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}),
                            iaa.Affine(rotate=(-30, 30)),
                            iaa.Affine(shear={'x': (-15, 15), 'y': (-15, 15)})
                            ])
                        ], random_order=True)


    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = self.seq_aug(image=image)

        return {'image':image, 'label':label}
