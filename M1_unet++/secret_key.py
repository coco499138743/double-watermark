import torch
def generater_key(output, A):
    k = torch.tensor([1.0, 1.0, 0,1.0])
    for n in range(len(k)):
        if k[n]==1:
            k[n]=-0.95
        else: k[n]=-1
    t = torch.rand((12))
    for i in range(len(t)):
        t[i] = k[i % 4]
    # t = torch.unsqueeze(t, 0)
    # a = torch.ones((95, 96))
    # tc = torch.cat([a, t], 0)  # 列数不增加，行数增加
    # tc = torch.unsqueeze(torch.unsqueeze(tc, 0), 0)
    t = t.to(device=torch.device("cuda:0"))
    S_M = A
    l = output.size(0)
    for i in range(l):
        j = 0
        r = 0
        for M in S_M:
            output[i][0][j][M] = t[r]
            j = j + 8
            r = r + 1
    return output


def clean_list(lst1):
    lst1 = []
    return lst1