from genotypes import PRIMITIVES
from operations import *
from utils import drop_path


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(op_names)//2

        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.cell_info = [False] * k * len(PRIMITIVES)

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for _ in range(2):
                for name in PRIMITIVES:
                    stride = 2 if reduction else 1
                    op = OPS[name](C, stride, True)
                    self._ops += [op]

            for _ in range(i):
                for name in PRIMITIVES:
                    stride = 1
                    op = OPS[name](C, stride, True)
                    self._ops += [op]

            cur = sum(1 for cnt in range(i) for n in range(2 + cnt))
            op1_id = PRIMITIVES.index(op_names[2 * i])
            op2_id = PRIMITIVES.index(op_names[2 * i + 1])

            self.cell_info[(cur + indices[2 * i])*len(PRIMITIVES) + op1_id] = True
            self.cell_info[(cur + indices[2 * i + 1])*len(PRIMITIVES) + op2_id] = True

        self._indices = indices
        self._op_names = op_names

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]

        for i in range(self._steps):

            cur = sum(1 for cnt in range(i) for n in range(2 + cnt))

            h1 = states[self._indices[2 * i]]
            op1_id = PRIMITIVES.index(self._op_names[2 * i])
            op1 = self._ops[(cur + self._indices[2 * i])*len(PRIMITIVES) + op1_id]

            h2 = states[self._indices[2 * i + 1]]
            op2_id = PRIMITIVES.index(self._op_names[2 * i + 1])
            op2 = self._ops[(cur + self._indices[2 * i + 1])*len(PRIMITIVES) + op2_id]

            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self.arch_info = []

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            self.arch_info.append(cell.cell_info)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


