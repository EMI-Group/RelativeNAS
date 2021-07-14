import numpy as np
from genotypes import Genotype, PRIMITIVES
from model_search import NetworkCIFAR

steps = 4  # define the number of intermediate nodes in a cell
multiplier = 4


def init_pop(args):
    m = args.pop_size
    # define the upper boundary
    upper = np.zeros((2 * 2 * 2 * steps, 1))  # two type of cells, two input nodes and two operations

    for i in range(2):  # iterate each type of cell
        n = 2  # the two input nodes

        for j in range(steps):
            # one node connection and the corresponding operation
            upper[i * 2 * 2 * steps + 4 * j] = n
            upper[i * 2 * 2 * steps + 4 * j + 1] = len(PRIMITIVES)

            # another node connection and the corresponding operation
            upper[i * 2 * 2 * steps + 4 * j + 2] = n
            upper[i * 2 * 2 * steps + 4 * j + 3] = len(PRIMITIVES)

            n += 1

    # lu: define the upper and lower bounds of the variables
    lu = np.concatenate((np.zeros((1, 2 * 2 * 2 * steps)), upper.T))
    XRRmin = np.tile(lu[0, :], (m, 1))
    XRRmax = np.tile(lu[1, :], (m, 1))
    # randomly initial the population
    p = XRRmin + (XRRmax - XRRmin) * np.random.rand(m, 2 * 2 * 2 * steps)

    return p, lu


# calculate the center point and variation of all the architecture vectors in the population
def cal_center(args, p):
    m = args.pop_size
    mean = np.mean(p, axis=0).reshape((1, len(np.mean(p, axis=0))))
    center = np.dot(np.ones((m // 2, 1)), mean)
    var = np.mean(np.sqrt(np.sum(np.abs(p-np.mean(p, axis=0))**2, axis=1)))

    return center, var


# generate randomly pairs
def gen_pairs(args):
    m = args.pop_size
    rlist = np.random.permutation(np.arange(m))
    rpairs = rlist.reshape((m // 2, 2))

    return rpairs


# convert the architecture vector into genotype
# genotype is text description of the architecture
def genotype(arch):
    def _parse(weights):
        # weights = np.floor(weights)
        gene = []
        for i in range(len(weights)//2):
            # variable = a if exper else b
            node = int(weights[2*i]) if weights[2*i] % 1 != 0 else int(weights[2*i]-0.1)
            op = int(weights[2*i+1]) if weights[2*i+1] % 1 != 0 else int(weights[2*i+1]-0.1)

            gene.append((PRIMITIVES[op], node))

        return gene

    gene_normal = _parse(arch[0: len(arch)//2])
    gene_reduce = _parse(arch[len(arch)//2:])

    concat = range(2 + steps - multiplier, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


# convert the architecture into model
def decode(args, CIFAR_CLASSES, arch, epoch):
    model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, genotype(arch))
    model = model.cuda()
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    return model


def update_state_dict(state_dict, model_a, model_b):    # model_a is the teacher while model_b is the student
    # arch_info contains True/False to indicate whether the operation is in the model
    a_info = model_a.arch_info
    b_info = model_b.arch_info
    model_a_dict = model_a.state_dict()
    model_b_dict = model_b.state_dict()

    for item in state_dict:
        if "cells" in item and "_ops" in item:

            info = item.split(".")
            layer_index = int(info[1])
            op_index = int(info[3])
            # if only model_a (teacher) contains the operation
            if a_info[layer_index][op_index] is True and b_info[layer_index][op_index] is False:
                # update with model_a (teacher)
                state_dict['{}'.format(item)] = model_a_dict['{}'.format(item)]

            # if only model_b (student) contains the operation
            elif a_info[layer_index][op_index] is False and b_info[layer_index][op_index] is True:
                # update with model_b (student)
                state_dict['{}'.format(item)] = model_b_dict['{}'.format(item)]

            # if both model_a (teacher) and model_b (student) contain the operation
            elif a_info[layer_index][op_index] is True and b_info[layer_index][op_index] is True:
                # update with model_a (teacher)
                state_dict['{}'.format(item)] = model_a_dict['{}'.format(item)]
            else:
                state_dict['{}'.format(item)] = model_a_dict['{}'.format(item)]

        else:
            # other weights are updated with model_a (teacher)
            state_dict['{}'.format(item)] = model_a_dict['{}'.format(item)]

    return state_dict


