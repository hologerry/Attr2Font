import os
import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


class ImageAttr(data.Dataset):
    """Dataset class for the ImageAttr dataset."""
    def __init__(self, image_dir, attr_path, transform, mode,
                 binary=False, n_style=4,
                 char_num=52, unsuper_num=968, train_num=120, val_num=28):
        """Initialize and preprocess the ImageAttr dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.n_style = n_style

        self.transform = transform
        self.mode = mode
        self.binary = binary

        self.super_train_dataset = []
        self.super_test_dataset = []
        self.unsuper_train_dataset = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.char_num = char_num
        self.unsupervised_font_num = unsuper_num
        self.train_font_num = train_num
        self.val_font_num = val_num

        self.test_super_unsuper = {}
        for super_font in range(self.train_font_num+self.val_font_num):
            self.test_super_unsuper[super_font] = random.randint(0, self.unsupervised_font_num - 1)

        self.char_idx_offset = 10

        self.chars = [c for c in range(self.char_idx_offset, self.char_idx_offset+self.char_num)]

        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.super_train_dataset) + len(self.unsuper_train_dataset)
        else:
            self.num_images = len(self.super_test_dataset)

    def preprocess(self):
        """Preprocess the font attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[0].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[1:]

        train_size = self.char_num * self.train_font_num
        val_size = self.char_num * self.val_font_num

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            target_char = filename.split('/')[1].split('.')[0]
            char_class = int(target_char) - self.char_idx_offset
            font_class = int(i / self.char_num)

            attr_value = []
            for val in values:
                if self.binary:
                    attr_value.append(val == '1')
                else:
                    attr_value.append(eval(val) / 100.0)

            # print(filename, char_class, font_class, attr_value)

            if i < train_size:
                self.super_train_dataset.append([filename, char_class, font_class, attr_value])
            elif i < train_size + val_size:
                self.super_test_dataset.append([filename, char_class, font_class, attr_value])
            else:
                self.unsuper_train_dataset.append([filename, char_class, font_class, attr_value])

        print('Finished preprocessing the Image Attribute (Explo) dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        # dataset = self.super_train_dataset if self.mode == 'train' else self.super_test_dataset

        if self.mode == 'train':
            if index < len(self.super_train_dataset):
                filename_A, charclass_A, fontclass_A, attr_A = self.super_train_dataset[index]
                label_A = 1.0
                font_embed_A = self.unsupervised_font_num  # dummy id 968
                # B is supervised or unsupervised
                sample_p = random.random()
                if sample_p < 0.5:
                    # Unsupervise
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.unsupervised_font_num - 1)
                    filename_B, charclass_B, fontclass_B, attr_B = self.unsuper_train_dataset[index_B]
                    label_B = 0.0
                    font_embed_B = fontclass_B - self.train_font_num - self.val_font_num  # convert to [0, 967]
                else:
                    # Supervise
                    # get B from supervise train !!
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.train_font_num - 1)
                    filename_B, charclass_B, fontclass_B, attr_B = self.super_train_dataset[index_B]
                    label_B = 1.0
                    font_embed_B = self.unsupervised_font_num  # dummy id 968

            else:
                # get A from unsupervise train !!
                index = index - len(self.super_train_dataset)
                filename_A, charclass_A, fontclass_A, attr_A = self.unsuper_train_dataset[index]
                label_A = 0.0
                font_embed_A = fontclass_A - self.train_font_num - self.val_font_num
                # B is supervised or unsupervised
                sample_p = random.random()
                if sample_p < 0.5:
                    # Unsupervise
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.unsupervised_font_num - 1)  # noqa
                    filename_B, charclass_B, fontclass_B, attr_B = self.unsuper_train_dataset[index_B]
                    label_B = 0.0
                    font_embed_B = fontclass_B - self.train_font_num - self.val_font_num  # convert to [0, 967]
                else:
                    # Supervise
                    # get B from supervise train !!
                    index_B = index % self.char_num + self.char_num * random.randint(0, self.train_font_num - 1)
                    filename_B, charclass_B, fontclass_B, attr_B = self.super_train_dataset[index_B]
                    label_B = 1.0
                    font_embed_B = self.unsupervised_font_num  # dummy id 968

        else:
            # load the random one from unsupervise data as the reference aka A
            # unsuper to super
            font_index_super = index // self.char_num + self.train_font_num
            font_index_unsuper = self.test_super_unsuper[font_index_super]
            char_index_unsuper = index % self.char_num + self.char_num * font_index_unsuper
            filename_A, charclass_A, fontclass_A, attr_A = self.unsuper_train_dataset[char_index_unsuper]
            label_A = 0.0
            font_embed_A = fontclass_A - self.train_font_num - self.val_font_num  # convert to [0, 967]

            filename_B, charclass_B, fontclass_B, attr_B = self.super_test_dataset[index]
            label_B = 1.0
            font_embed_B = self.unsupervised_font_num  # dummy id 968

        # Get style samples
        random.shuffle(self.chars)
        style_chars = self.chars[:self.n_style]
        styles_A = []
        if self.n_style == 1:
            styles_A.append(filename_A)
        else:
            for char in style_chars:
                styles_A.append(rreplace(filename_A, str(charclass_A+10), str(char), 1))

        random.shuffle(self.chars)
        style_chars = self.chars[:self.n_style]
        styles_B = []
        if self.n_style == 1:
            styles_B.append(filename_B)
        else:
            for char in style_chars:
                styles_B.append(rreplace(filename_B, str(charclass_B+10), str(char), 1))

        image_A = Image.open(os.path.join(self.image_dir, filename_A)).convert('RGB')
        image_B = Image.open(os.path.join(self.image_dir, filename_B)).convert('RGB')
        # Open and transform style images
        style_imgs_A = []
        for style_A in styles_A:
            style_imgs_A.append(self.transform(Image.open(os.path.join(self.image_dir, style_A)).convert('RGB')))
        style_imgs_A = torch.cat(style_imgs_A)
        style_imgs_B = []
        for style_B in styles_B:
            style_imgs_B.append(self.transform(Image.open(os.path.join(self.image_dir, style_B)).convert('RGB')))
        style_imgs_B = torch.cat(style_imgs_B)

        return {"img_A": self.transform(image_A), "charclass_A": torch.LongTensor([charclass_A]),
                "fontclass_A": torch.LongTensor([fontclass_A]), "attr_A": torch.FloatTensor(attr_A),
                "styles_A": style_imgs_A,
                "fontembed_A": torch.LongTensor([font_embed_A]),
                "label_A": torch.FloatTensor([label_A]),
                "img_B": self.transform(image_B), "charclass_B": torch.LongTensor([charclass_B]),
                "fontclass_B": torch.LongTensor([fontclass_B]), "attr_B": torch.FloatTensor(attr_B),
                "styles_B": style_imgs_B,
                "fontembed_B": torch.LongTensor([font_embed_B]),
                "label_B": torch.FloatTensor([label_B])}

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, image_size=256,
               batch_size=16, dataset_name='explor_all', mode='train', num_workers=8,
               binary=False, n_style=4,
               char_num=52, unsuper_num=968, train_num=120, val_num=28):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset_name == 'explor_all':
        dataset = ImageAttr(image_dir, attr_path, transform,
                            mode, binary, n_style,
                            char_num=52, unsuper_num=968,
                            train_num=120, val_num=28)
    data_loader = data.DataLoader(dataset=dataset,
                                  drop_last=True,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)

    return data_loader
