import scipy.io as scio

src_file = "/raid/huangsh/imagenet/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.mat"
des_file = "/raid/huangsh/imagenet/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.txt"

f = open(des_file, 'w')

val_names = scio.loadmat(src_file)
ids = val_names['synsets']['ILSVRC2012_ID']
names = val_names['synsets']['WNID']

for i in range(len(ids)):
    id = ids[i][0][0][0]
    name = names[i][0][0]
    f.write('{}\t{}\n'.format(id, name))
f.close()