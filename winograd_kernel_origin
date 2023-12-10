import sys
import torch
import torch.nn as nn
from torch import tensor
import math
import time
import torch.nn.functional as F
import torch.nn.init as init


def get_transform_matrices(m=2):
    if m == 2:
        # store 3 transform matrices G, B, A for F(2x2, 3x3)
        B = tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, -1.0, 1.0],
             [-1.0, 1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, -1.0]])
        B_T = B.transpose(1, 0)
        G = tensor(
            [[1.0, 0.0, 0.0],
             [0.5, 0.5, 0.5],
             [0.5, -0.5, 0.5],
             [0.0, 0.0, 1.0]])
        G_T = G.transpose(1, 0)
        A = tensor([[1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                    [0.0, -1.0]])
        A_T = A.transpose(1, 0)

    if m == 4:
        # store 3 transform matrices G, B, A for F(4x4, 3x3)
        B_T = tensor(
            [[4.0, 0.0, -5.0, 0.0, 1.0, 0.0],
             [0.0, -4.0, -4.0, 1.0, 1.0, 0.0],
             [0.0, 4.0, -4.0, -1.0, 1.0, 0.0],
             [0.0, -2.0, -1.0, 2.0, 1.0, 0.0],
             [0.0, 2.0, -1.0, -2.0, 1.0, 0.0],
             [0.0, 4.0, 0.0, -5.0, 0.0, 1.0]]
        )
        B = B_T.transpose(1, 0)
        G = tensor(
            [[1/4, 0.0, 0.0],
             [-1/6, -1/6, -1/6],
             [-1/6, 1/6, -1/6],
             [1/24, 1/12, 1/6],
             [1/24, -1/12, 1/6],
             [0.0, 0.0, 1.0]]
        )
        G_T = G.transpose(1, 0)
        A_T = tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
             [0.0, 1.0, -1.0, 2.0, -2.0, 0.0],
             [0.0, 1.0, 1.0, 4.0, 4.0, 0.0],
             [0.0, 1.0, -1.0, 8.0, -8.0, 1.0]]
        )
        A = A_T.transpose(1, 0)

    # print('A:', A.shape, 'B:', B.shape, 'G:', G.shape)
    return A, A_T, B, B_T, G, G_T


class Winograd(nn.Module):

    def __init__(self, m):
        super(Winograd, self).__init__()

        self.m = m

    def forward(self, input, weight):
        """
        Compute Winograd convolution (general condition for DNNs).
        F(mxm, rxr)

        :param input:
        :param filter:
        :return: output
        """
        
        N, C, H, W = input.size()
        K, _, r, _ = weight.size()
        assert H == W

        m = self.m  # the size of output tile (optional)
        print('the size of output tile:', m)
        a = m + r - 1  # the size of input tile
        print('the size of input tile:', a)

        if (H >= 4 and H % 2 == 0) is False:
            raise Exception("Only input for perfect tiling is supported.")  # H/W should be even numbers

        # compute the number of tiles
        T = int(math.ceil((H - r + 1) / m))  # the number of tiles per channel = ceil(H_output / m)
        P = N * T * T  # number of tiles
        print('number of input tiles:', T * T)

        A, A_T, B, B_T, G, G_T = get_transform_matrices(m)

        # Winograd transformation: Y = A^T((GgG^T) * (B^TdB))A
        U = torch.zeros(K, C, a, a)
        V = torch.zeros(C, P, a, a)

        # step 1: U = GgG^T (for each channel in the filter)
        for k in range(K):
            for c in range(C):
                U[k, c] = torch.matmul(G, torch.matmul(weight[k, c], G_T))  

        # step 2: V = B^TdB
        for n in range(N):  # batch
            for tH in range(T):  # number of tiles per channel
                for tW in range(T):  # number of tiles per channel
                    for c in range(C):
                        b = n * (T * T) + tH * T + tW
                        vH = tH * m  # tH * (r-1) is wrong, should be stride=m
                        vW = tW * m  # tW * (r-1) is wrong, should be stride=m
                        V[c, b] = torch.matmul(B_T, torch.matmul(input[n, c, vH:vH + a, vW:vW + a], B))

        # step 3: M = U * V (multi-channel should be added together at the same position or convert EWMM to GEMM!)
        M = torch.zeros(K, P, a, a)
        for k in range(K):
            for b in range(P):
                for c in range(C):
                    M[k, b] += U[k, c] * V[c, b]  # way 1: EWMM and then add cross channel C

        # U, V = U.permute(2, 3, 0, 1), V.permute(2, 3, 0, 1)  # U: (a, a, K, C), V: (a, a, C, P)
        # M = torch.matmul(U, V)  # way 2: M = UV (faster than way 1): (a, a, K, P)
        # M = M.permute(2, 3, 0, 1)  # M: (K, P, a, a)

        # step 4: Y = A^T(U * V)A
        out_size = H - r + 1
        Y = torch.zeros(N, K, out_size, out_size)
        for k in range(K):
            for n in range(N): 
                for tH in range(T):
                    for tW in range(T):
                        b = n * (T * T) + tH * T + tW
                        oH = tH * m
                        oW = tW * m
                        Y[n, k, oH:oH + m, oW:oW + m] = torch.matmul(A_T, torch.matmul(M[k, b], A))

        return Y
    
