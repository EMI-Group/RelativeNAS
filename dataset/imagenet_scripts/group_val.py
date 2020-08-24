import os
import shutil

id_format = '00000000'

names = "/raid/huangsh/imagenet/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.txt"
labels = "/raid/huangsh/imagenet/ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

src_dir = "/raid/huangsh/imagenet/ILSVRC2012/val_imgs"
des_dir = "/raid/huangsh/imagenet/ILSVRC2012/images/val"

f1 = open(names, 'r')
f2 = open(labels, 'r')

line1 = f1.readline()

names_list = []
while line1:
    list1 = line1.split('\t')[1]
    names_list.append(list1[:-1])
    line1 = f1.readline()
f1.close()

line2 = f2.readline()
img_id = 1
while line2:
    label = line2.split('\n')[0]
    sub_name = names_list[int(label)-1]
    sub_dir = '{}/{}'.format(des_dir, sub_name)
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    # else:
    #     for file in os.listdir(sub_dir):
    #         os.remove('{}/{}'.format(sub_dir, file))
    leng = len(str(img_id))
    img_name = 'ILSVRC2012_val_{}{}.JPEG'.format(id_format[:-leng], img_id)
    shutil.copyfile('{}/{}'.format(src_dir, img_name), '{}/{}'.format(sub_dir, img_name))
    line2 = f2.readline()
    img_id += 1


