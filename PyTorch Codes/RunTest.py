import os.path
import torch
import time
import numpy as np
from tqdm import tqdm, trange
import torchgeometry as tgm


def getInput(bS, device):
    src = torch.randint(low=10, high=30, size=[bS, 2], device=device).float()
    src = src.unsqueeze(1).repeat(1, 4, 1)
    src[:, 1, 0] += 128
    src[:, 2, 1] += 128
    src[:, 3, 0] += 128
    src[:, 3, 1] += 128
    return src


def getTar(bs, src):
    wave = torch.randint(low=0, high=32, size=[bs, 4, 2], device=src.device).float()
    return src + wave


def adjust(device, bs):
    mydevice = device
    src_ps = getInput(bs, mydevice)
    dst_p = getTar(bs, src_ps)
    src_ps_mid = src_ps.transpose(1, 2)
    dst_p_mid = dst_p.transpose(1, 2)
    ones = torch.ones((bs, 1, 4), device=mydevice)
    src_ps_new = torch.cat((src_ps_mid, ones), dim=1)
    dst_p_new = torch.cat((dst_p_mid, ones), dim=1)
    scale = src_ps_new[0, 0, 1:2] - src_ps_new[0, 0, 0:1]
    scale_y = src_ps_new[0, 1, 2:3] - src_ps_new[0, 1, 0:1]
    div = scale / scale_y

    return src_ps, dst_p, src_ps_new, dst_p_new, scale, div


def TensorDLT(bs, src, tar, loops):
    print("TensorDLT \n")
    if src.device == 'cuda':
        for i in range(10):
            warm = torch.inverse(torch.randn((bs, 3, 3)).cuda())

    time_list = []
    for i in trange(loops):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        ones = torch.ones((bs, 4, 1), device=src.device)
        xy1 = torch.cat((src, ones), 2)
        zeros = torch.zeros_like(xy1, device=src.device)

        xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
        M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
        M2 = torch.matmul(
            tar.reshape(-1, 2, 1),
            src.reshape(-1, 1, 2),
        ).reshape(bs, -1, 2)

        A = torch.cat((M1, -M2), 2)
        b = tar.reshape(bs, -1, 1)

        Ainv = torch.inverse(A)
        h8 = torch.matmul(Ainv, b).reshape(bs, 8)
        H = torch.cat((h8, ones[:, 0, :]), 1).reshape(bs, 3, 3)

        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        time_list.append(end - start)
    mean = np.mean(np.array(time_list)[10:])

    return mean


def TensorACA_rec(bs, src, tar, scale, div):
    print("TensorACA_rec \n")
    if src.device == 'cuda':
        for i in range(10):
            warm = torch.inverse(torch.randn((bs, 3, 3)).cuda())
    time_list = []
    for i in trange(loops):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        H = torch.zeros((bs, 3, 3), device=src.device)
        MN_MP_MQ_P2 = tar[:, :, 1:] - tar[:, :, 0:1]
        Q4 = torch.cross(MN_MP_MQ_P2[:, 1:2, :], MN_MP_MQ_P2[:, 0:1, :], dim=2)
        h_temp = torch.sum(Q4, dim=2, keepdim=True) * tar[:, :, 0:1]
        H[:, :, 0:1] = tar[:, :, 1:2] * Q4[:, :, 0:1] - h_temp
        H[:, :, 1:2] = torch.mul(div, tar[:, :, 2:3] * Q4[:, :, 1:2] - h_temp)
        H[:, :, 2:3] = scale * h_temp - src[:, 0:1, 0:1] * H[:, :, 0:1] - src[:, 1:2, 0:1] * H[:, :, 1:2]

        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        time_list.append(end - start)
    mean = np.mean(np.array(time_list)[10:])

    return mean


def TensorACA_C(bs, src, tar, loops):
    print("TensorACA_C \n")
    if src.device == 'cuda':
        for i in range(10):
            warm = torch.inverse(torch.randn((bs, 3, 3)).cuda())
    time_list = []

    for i in trange(loops):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        MN_MP_MQ_P1 = src[:, :, 1:] - src[:, :, 0:1]
        MN_MP_MQ_P2 = tar[:, :, 1:] - tar[:, :, 0:1]
        Q3 = torch.cross(MN_MP_MQ_P1[:, 1:2, :], MN_MP_MQ_P1[:, 0:1, :], dim=2)
        Q3[:, :, -1:] = torch.sum(Q3, dim=2, keepdim=True)
        Q4 = torch.cross(MN_MP_MQ_P2[:, 1:2, :], MN_MP_MQ_P2[:, 0:1, :], dim=2)
        Q4[:, :, -1:] = torch.sum(Q4, dim=2, keepdim=True)
        a_vec = torch.div(Q4, Q3)
        N_P_M_2 = torch.cat((tar[:, :, 1:2], tar[:, :, 2:3], tar[:, :, 0:1]), dim=2)
        N_P_M_1 = torch.cat((src[:, :, 1:2], src[:, :, 2:3], src[:, :, 0:1]), dim=2)
        H_P1 = a_vec * N_P_M_2
        H = torch.matmul(H_P1, N_P_M_1.inverse())

        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        time_list.append(end - start)
    mean = np.mean(np.array(time_list)[10:])
    return mean


def ACA_C_Python(bs, src, tar, loops):
    print("ACA_C_Python \n")
    if src.device == 'cuda':
        for i in range(10):
            warm = torch.inverse(torch.randn((bs, 3, 3)).cuda())
    time_list = []
    for i in trange(loops):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        M1N1_X = src[:, 1, 0] - src[:, 0, 0]
        M1N1_Y = src[:, 1, 1] - src[:, 0, 1]

        M1P1_X = src[:, 2, 0] - src[:, 0, 0]
        M1P1_Y = src[:, 2, 1] - src[:, 0, 1]

        M1Q1_X = src[:, 3, 0] - src[:, 0, 0]
        M1Q1_Y = src[:, 3, 1] - src[:, 0, 1]

        fA1 = M1N1_X * M1P1_Y - M1N1_Y * M1P1_X
        Q3_x = M1P1_Y * M1Q1_X - M1P1_X * M1Q1_Y
        Q3_y = M1N1_X * M1Q1_Y - M1N1_Y * M1Q1_X

        M2N2_X = tar[:, 1, 0] - tar[:, 0, 0]
        M2N2_Y = tar[:, 1, 1] - tar[:, 0, 1]

        M2P2_X = tar[:, 2, 0] - tar[:, 0, 0]
        M2P2_Y = tar[:, 2, 1] - tar[:, 0, 1]

        M2Q2_X = tar[:, 3, 0] - tar[:, 0, 0]
        M2Q2_Y = tar[:, 3, 1] - tar[:, 0, 1]

        fA2 = M2N2_X * M2P2_Y - M2N2_Y * M2P2_X
        Q4_x = M2P2_Y * M2Q2_X - M2P2_X * M2Q2_Y
        Q4_y = M2N2_X * M2Q2_Y - M2N2_Y * M2Q2_X

        tt1 = fA1 - Q3_x - Q3_y
        C11 = Q3_y * Q4_x * tt1
        C22 = Q3_x * Q4_y * tt1
        C33 = Q3_x * Q3_y * (fA2 - Q4_x - Q4_y)
        C31 = C11 - C33
        C32 = C22 - C33

        tt3 = tar[:, 0, 0] * C33
        tt4 = tar[:, 0, 1] * C33
        H1_11 = tar[:, 1, 0] * C11 - tt3
        H1_12 = tar[:, 2, 0] * C22 - tt3
        H1_21 = tar[:, 1, 1] * C11 - tt4
        H1_22 = tar[:, 2, 1] * C22 - tt4

        res_0 = H1_11 * M1P1_Y - H1_12 * M1N1_Y
        res_1 = H1_12 * M1N1_X - H1_11 * M1P1_X
        res_3 = H1_21 * M1P1_Y - H1_22 * M1N1_Y
        res_4 = H1_22 * M1N1_X - H1_21 * M1P1_X
        res_6 = C31 * M1P1_Y - C32 * M1N1_Y
        res_7 = C32 * M1N1_X - C31 * M1P1_X
        res_2 = tt3 * fA1 - res_0 * src[:, 0, 0] - res_1 * src[:, 0, 1]
        res_5 = tt4 * fA1 - res_3 * src[:, 0, 0] - res_4 * src[:, 0, 1]
        res_8 = C33 * fA1 - res_6 * src[:, 0, 0] - res_7 * src[:, 0, 1]

        H = torch.ones((bs, 9), device=src.device)
        H[:, 0] = res_0
        H[:, 1] = res_1
        H[:, 2] = res_2
        H[:, 3] = res_3
        H[:, 4] = res_4
        H[:, 5] = res_5
        H[:, 6] = res_6
        H[:, 7] = res_7
        H[:, 8] = res_8
        H = H.reshape(bs, 3, 3)

        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        time_list.append(end - start)
    mean = np.mean(np.array(time_list)[10:])
    return mean


def TensorIHN(bs, src, tar, loops):
    print("TensorIHN \n")
    if src.device == 'cuda':
        for i in range(10):
            warm = torch.inverse(torch.randn((bs, 3, 3)).cuda())
    time_list = []
    for i in trange(loops):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        H = tgm.get_perspective_transform(src, tar)

        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        time_list.append(end - start)
    mean = np.mean(np.array(time_list)[10:])

    return mean


def TensorGE(bs, src, tar, loops):
    print("TensorGE \n")
    if src.device == 'cuda':
        for i in range(10):
            warm = torch.inverse(torch.randn((bs, 3, 3)).cuda())
    time_list = []

    for i in trange(loops):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        x1 = src[:, 0:1, 0:1]
        y1 = src[:, 0:1, 1:]
        x0 = src[:, 1:2, 0:1]
        y0 = src[:, 1:2, 1:]
        x2 = src[:, 2:3, 0:1]
        y2 = src[:, 2:3, 1:]
        x3 = src[:, 3:, 0:1]
        y3 = src[:, 3:, 1:]
        X1 = tar[:, 0:1, 0:1]
        Y1 = tar[:, 0:1, 1:]
        X0 = tar[:, 1:2, 0:1]
        Y0 = tar[:, 1:2, 1:]
        X2 = tar[:, 2:3, 0:1]
        Y2 = tar[:, 2:3, 1:]
        X3 = tar[:, 3:, 0:1]
        Y3 = tar[:, 3:, 1:]

        x0X0 = x0 * X0
        x1X1 = x1 * X1
        x2X2 = x2 * X2
        x3X3 = x3 * X3
        x0Y0 = x0 * Y0
        x1Y1 = x1 * Y1
        x2Y2 = x2 * Y2
        x3Y3 = x3 * Y3
        y0X0 = y0 * X0
        y1X1 = y1 * X1
        y2X2 = y2 * X2
        y3X3 = y3 * X3
        y0Y0 = y0 * Y0
        y1Y1 = y1 * Y1
        y2Y2 = y2 * Y2
        y3Y3 = y3 * Y3

        minor = torch.cat(
            (torch.cat((x0 - x2, x1 - x2, x2, x3 - x2), dim=2), torch.cat((y0 - y2, y1 - y2, y2, y3 - y2), dim=2)),
            dim=1)
        major = torch.cat((torch.cat(
            (x2X2 - x0X0, x2X2 - x1X1, -x2X2, x2X2 - x3X3, x2Y2 - x0Y0, x2Y2 - x1Y1, -x2Y2, x2Y2 - x3Y3), dim=2),
                           torch.cat((y2X2 - y0X0, y2X2 - y1X1, -y2X2, y2X2 - y3X3, y2Y2 - y0Y0, y2Y2 - y1Y1, -y2Y2,
                                      y2Y2 - y3Y3), dim=2),
                           torch.cat(((X0 - X2), (X1 - X2), X2, (X3 - X2), (Y0 - Y2), (Y1 - Y2), Y2, (Y3 - Y2)),
                                     dim=2)), dim=1)

        scalar1 = minor[:, 0:1, 0:1]
        scalar2 = minor[:, 0:1, 1:2]
        minor[:, 1:2, 1:2] = minor[:, 1:2, 1:2] * scalar1 - minor[:, 1:2, 0:1] * scalar2

        major[:, 0:1, 1:2] = major[:, 0:1, 1:2] * scalar1 - major[:, 0:1, 0:1] * scalar2
        major[:, 1:2, 1:2] = major[:, 1:2, 1:2] * scalar1 - major[:, 1:2, 0:1] * scalar2
        major[:, 2:3, 1:2] = major[:, 2:3, 1:2] * scalar1 - major[:, 2:3, 0:1] * scalar2

        major[:, 0:1, 5:6] = major[:, 0:1, 5:6] * scalar1 - major[:, 0:1, 4:5] * scalar2
        major[:, 1:2, 5:6] = major[:, 1:2, 5:6] * scalar1 - major[:, 1:2, 4:5] * scalar2
        major[:, 2:3, 5:6] = major[:, 2:3, 5:6] * scalar1 - major[:, 2:3, 4:5] * scalar2

        scalar2 = minor[:, 0:1, 3:4]
        minor[:, 1:2, 3:4] = minor[:, 1:2, 3:4] * scalar1 - minor[:, 1:2, 0:1] * scalar2

        major[:, 0:1, 3:4] = major[:, 0:1, 3:4] * scalar1 - major[:, 0:1, 0:1] * scalar2
        major[:, 1:2, 3:4] = major[:, 1:2, 3:4] * scalar1 - major[:, 1:2, 0:1] * scalar2
        major[:, 2:3, 3:4] = major[:, 2:3, 3:4] * scalar1 - major[:, 2:3, 0:1] * scalar2

        major[:, 0:1, 7:8] = major[:, 0:1, 7:8] * scalar1 - major[:, 0:1, 4:5] * scalar2
        major[:, 1:2, 7:8] = major[:, 1:2, 7:8] * scalar1 - major[:, 1:2, 4:5] * scalar2
        major[:, 2:3, 7:8] = major[:, 2:3, 7:8] * scalar1 - major[:, 2:3, 4:5] * scalar2

        scalar1 = minor[:, 1:2, 1:2]
        scalar2 = minor[:, 1:2, 3:4]
        major[:, 0:1, 3:4] = major[:, 0:1, 3:4] * scalar1 - major[:, 0:1, 1:2] * scalar2
        major[:, 1:2, 3:4] = major[:, 1:2, 3:4] * scalar1 - major[:, 1:2, 1:2] * scalar2
        major[:, 2:3, 3:4] = major[:, 2:3, 3:4] * scalar1 - major[:, 2:3, 1:2] * scalar2

        major[:, 0:1, 7:8] = major[:, 0:1, 7:8] * scalar1 - major[:, 0:1, 5:6] * scalar2
        major[:, 1:2, 7:8] = major[:, 1:2, 7:8] * scalar1 - major[:, 1:2, 5:6] * scalar2
        major[:, 2:3, 7:8] = major[:, 2:3, 7:8] * scalar1 - major[:, 2:3, 5:6] * scalar2

        scalar2 = minor[:, 1:2, 0:1]
        minor[:, 0:1, 0:1] = minor[:, 0:1, 0:1] * scalar1 - minor[:, 0:1, 1:2] * scalar2

        major[:, 0:1, 0:1] = major[:, 0:1, 0:1] * scalar1 - major[:, 0:1, 1:2] * scalar2
        major[:, 1:2, 0:1] = major[:, 1:2, 0:1] * scalar1 - major[:, 1:2, 1:2] * scalar2
        major[:, 2:3, 0:1] = major[:, 2:3, 0:1] * scalar1 - major[:, 2:3, 1:2] * scalar2

        major[:, 0:1, 4:5] = major[:, 0:1, 4:5] * scalar1 - major[:, 0:1, 5:6] * scalar2
        major[:, 1:2, 4:5] = major[:, 1:2, 4:5] * scalar1 - major[:, 1:2, 5:6] * scalar2
        major[:, 2:3, 4:5] = major[:, 2:3, 4:5] * scalar1 - major[:, 2:3, 5:6] * scalar2

        scalar1 = 1.0 / minor[:, 0:1, 0:1]
        major[:, 0:1, 0:1] *= scalar1
        major[:, 1:2, 0:1] *= scalar1
        major[:, 2:3, 0:1] *= scalar1
        major[:, 0:1, 4:5] *= scalar1
        major[:, 1:2, 4:5] *= scalar1
        major[:, 2:3, 4:5] *= scalar1

        scalar1 = 1.0 / minor[:, 1:2, 1:2]
        major[:, 0:1, 1:2] *= scalar1
        major[:, 1:2, 1:2] *= scalar1
        major[:, 2:3, 1:2] *= scalar1
        major[:, 0:1, 5:6] *= scalar1
        major[:, 1:2, 5:6] *= scalar1
        major[:, 2:3, 5:6] *= scalar1

        scalar1 = minor[:, 0:1, 2:3]
        scalar2 = minor[:, 1:2, 2:3]
        major[:, 0:1, 2:3] -= major[:, 0:1, 0:1] * scalar1 + major[:, 0:1, 1:2] * scalar2
        major[:, 1:2, 2:3] -= major[:, 1:2, 0:1] * scalar1 + major[:, 1:2, 1:2] * scalar2
        major[:, 2:3, 2:3] -= major[:, 2:3, 0:1] * scalar1 + major[:, 2:3, 1:2] * scalar2

        major[:, 0:1, 6:7] -= major[:, 0:1, 4:5] * scalar1 + major[:, 0:1, 5:6] * scalar2
        major[:, 1:2, 6:7] -= major[:, 1:2, 4:5] * scalar1 + major[:, 1:2, 5:6] * scalar2
        major[:, 2:3, 6:7] -= major[:, 2:3, 4:5] * scalar1 + major[:, 2:3, 5:6] * scalar2

        scalar1 = major[:, 0:1, 7:8]

        major[:, 1:2, 7:8] /= scalar1
        major[:, 2:3, 7:8] /= scalar1

        scalar1 = major[:, 0:1, 0:1]
        major[:, 1:2, 0:1] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 0:1] -= scalar1 * major[:, 2:3, 7:8]
        scalar1 = major[:, 0:1, 1:2]
        major[:, 1:2, 1:2] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 1:2] -= scalar1 * major[:, 2:3, 7:8]
        scalar1 = major[:, 0:1, 2:3]
        major[:, 1:2, 2:3] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 2:3] -= scalar1 * major[:, 2:3, 7:8]
        scalar1 = major[:, 0:1, 3:4]
        major[:, 1:2, 3:4] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 3:4] -= scalar1 * major[:, 2:3, 7:8]
        scalar1 = major[:, 0:1, 4:5]
        major[:, 1:2, 4:5] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 4:5] -= scalar1 * major[:, 2:3, 7:8]
        scalar1 = major[:, 0:1, 5:6]
        major[:, 1:2, 5:6] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 5:6] -= scalar1 * major[:, 2:3, 7:8]
        scalar1 = major[:, 0:1, 6:7]
        major[:, 1:2, 6:7] -= scalar1 * major[:, 1:2, 7:8]
        major[:, 2:3, 6:7] -= scalar1 * major[:, 2:3, 7:8]

        scalar1 = major[:, 1:2, 3:4]
        major[:, 2:3, 3:4] /= scalar1

        scalar1 = major[:, 1:2, 0:1]
        major[:, 2:3, 0:1] -= scalar1 * major[:, 2:3, 3:4]
        scalar1 = major[:, 1:2, 1:2]
        major[:, 2:3, 1:2] -= scalar1 * major[:, 2:3, 3:4]
        scalar1 = major[:, 1:2, 2:3]
        major[:, 2:3, 2:3] -= scalar1 * major[:, 2:3, 3:4]
        scalar1 = major[:, 1:2, 4:5]
        major[:, 2:3, 4:5] -= scalar1 * major[:, 2:3, 3:4]
        scalar1 = major[:, 1:2, 5:6]
        major[:, 2:3, 5:6] -= scalar1 * major[:, 2:3, 3:4]
        scalar1 = major[:, 1:2, 6:7]
        major[:, 2:3, 6:7] -= scalar1 * major[:, 2:3, 3:4]
        scalar1 = major[:, 1:2, 7:8]
        major[:, 2:3, 7:8] -= scalar1 * major[:, 2:3, 3:4]

        H = torch.cat((major[:, 2:3, :3], major[:, 2:3, 4:7],
                       torch.cat((major[:, 2:3, 7:8], major[:, 2:3, 3:4], torch.ones_like(scalar1)), dim=2)), dim=1)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        time_list.append(end - start)
    mean = np.mean(np.array(time_list)[10:])
    return mean


if __name__ == "__main__":
    batchSize = 1
    device = 'cuda'

    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    loops = 500000
    src, tar, src_our, tar_our, scale, div = adjust(device, batchSize)

    IHN_time = TensorIHN(batchSize, src, tar, loops)
    DLT_time = TensorDLT(batchSize, src, tar, loops)
    ACA_rec_time = TensorACA_rec(batchSize, src_our, tar_our, scale, div)
    ACA_mat_time = TensorACA_C(batchSize, src_our, tar_our, loops)
    ACA_C_Python_time = ACA_C_Python(batchSize, src, tar, loops)
    GE_time = TensorGE(batchSize, src, tar, loops)

    print(f"IHN: {IHN_time / 1e3}\n")
    print(f"DLT: {DLT_time / 1e3}\n")
    print(f"ACA_rec: {ACA_rec_time / 1e3}\n")
    print(f"ACA_Mat: {ACA_mat_time / 1e3}\n")
    print(f"ACA_C_Python: {ACA_C_Python_time / 1e3}\n")
    print(f"GE: {GE_time / 1e3}\n")
