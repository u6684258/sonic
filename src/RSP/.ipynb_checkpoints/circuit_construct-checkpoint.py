import numpy as np


def solar_energy(fts, lts, sig_0, sig_1, G_ct, S_GT, G_p, G_e, epi, d, N):
    T = 1
    M = 16
    # d = 3
    D1 = []
    D2 = []
    K = []
    D3 = []
    Pt = []
    PCoeff = []
    for t in range(len(fts)):
        D1t = fts[t] * lts[t]
        # print(D1t)
        D2t = D1t * sig_0
        Kt = (10**d)**3 * np.ones_like(D2t) - D2t + sig_1
        # print(Kt)
        D3t = Kt * G_ct[t]
        D1.append(D1t)
        D2.append(D2t)
        K.append(Kt)
        D3.append(D3t)
        P = G_p * S_GT * G_ct[t]
        Pt.append(P)
        PCoeff.append((10**d)**3 * np.ones_like(D2t) * P + P * sig_1)
    D4 = np.sum(D3, axis=0)
    D5 = D4 * S_GT
    D6 = D5 * G_p
    G = np.sum(D6)
    G_e  = G+10
    # epi = 1.01
    d_index = int(G_e * epi - G)
    print(f"G={G}")
    print(f"index={d_index}")
    B = np.array([int(x) for x in format(d_index, f'0{M}b')])
    print(f"bit length: {B.shape}")

    k_final = np.sum(np.concatenate(PCoeff))



    a_upper = np.concatenate(lts + fts).astype(int)
    b_upper = np.zeros_like(a_upper).astype(int)
    c_upper = np.zeros_like(a_upper).astype(int)

    assert(a_upper.shape[0] == N*(2*T))
    assert(b_upper.shape[0] == N*(2*T))
    assert(c_upper.shape[0] == N*(2*T))
    assert(((a_upper * b_upper).astype(int)  == c_upper).all())

    a_middle = np.concatenate(lts +
                                [B] +
                                [[G]]
                            ).astype(int)

    b_middle = np.concatenate(fts +
                                [1 - B] + 
                                [[0]]
                            ).astype(int)
    c_middle = np.concatenate(D1 +
                                [np.zeros(M)] +
                                [[0]]
                            ).astype(int)

    assert(a_middle.shape[0] == N*(T)+M+1)
    assert(b_middle.shape[0] == N*(T)+M+1)
    assert(c_middle.shape[0]  == N*(T)+M+1)
    assert(((a_middle * b_middle).astype(int)  == c_middle).all())
    # print(a_middle)
    # print(b_middle)
    # print(c_middle)
    # print((a_middle * b_middle  // (10**d)).astype(int))

    a = np.concatenate([a_upper, a_middle])
    b = np.concatenate([b_upper, b_middle])
    c = np.concatenate([c_upper, c_middle])
    # Q1
    K_1 = []
    u_1 = []
    v_1 = []
    w_1 = []
    for i in range(N*T):
        # copy f_i
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[i] = 1
        u1[2*N*T + i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        # copy l_i
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[N*T + i] = 1
        v1[2*N*T + i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # Q6
    # check G_e * epsilon = 2 * B + G
    k1 = G_e * epi
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[-1] = 1
    u1[3*N*T:3*N*T+M] = np.array([2**(M-i-1) for i in range(M)])
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    # Q7
    # check 1-B+B=1
    for i in range(M):
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[3*N*T+i] =1
        v1[3*N*T+i] =1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # Q6
    # check G_e * epsilon = 2 * B + G
    k1 = k_final
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[-1] = 1
    for t in range(T):
        w1[2*N*T+t*N:2*N*T+(t+1)*N] = Pt[t]*sig_0
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)
    
    for i in range(len(K_1)):
        # print(i)
        # print(f"K:{K_1[i]}")
        # print(f"u:{u_1[i]@a}")
        # print(f"v:{v_1[i]@b}")
        # print(f"w:{w_1[i]@c}")
        # print()

        assert(K_1[i]==u_1[i]@a+v_1[i]@b+w_1[i]@c)

    print(np.concatenate([a_upper, a_middle]).shape)
    print(np.array(K_1).shape)  


    return ([a_upper, a_middle], [b_upper, b_middle], [c_upper, c_middle], 
            K_1, u_1, v_1, w_1)


def bushfire(a0, b0, a1, b1, sig, epi, cd, N):

    M = 16

    D0 = cd * (a0-b0) // (a0+b0)
    r0 = cd * (a0-b0) - D0 * (a0+b0)
    D1 = cd * (a1-b1) // (a1+b1)
    r1 = cd * (a1-b1) - D1 * (a1+b1)
    for i in range(len(r1)):
        if r0[i] < 0:
            r0[i] = r0[i] + (a0+b0)[i]
            D0[i] = D0[i] - 1
        if r1[i] < 0:
            r1[i] = r1[i] + (a1+b1)[i]
            D1[i] = D1[i] - 1
    Dd = D0 - D1
    C = Dd - sig
    I = (C >= 0).astype(int)
    # print(I)
    C = C * I
    s = np.sum(I)
    sdepi = s - epi
    
    assert((C >= 0).all())
    print(f"G={s}")
    print(f"index={sdepi}")

    Bis = [np.array([int(x) for x in format(C[i], f'0{M}b')]) for i in range(len(C))]
    # print(Bis)
    Bs = np.array([int(x) for x in format(sdepi, f'0{M}b')])
    Br = [np.array([int(x) for x in format(r0[i], f'0{M}b')]) for i in range(len(r0))] +\
            [np.array([int(x) for x in format(r1[i], f'0{M}b')]) for i in range(len(r1))]
    Bdr = [np.array([int(x) for x in format((a0+b0)[i]-r0[i], f'0{M}b')]) for i in range(len(r0))] +\
            [np.array([int(x) for x in format((a1+b1)[i]-r1[i], f'0{M}b')]) for i in range(len(r1))]

    a_upper = np.concatenate([a0, b0, a1, b1]).astype(int)
    b_upper = np.zeros_like(a_upper).astype(int)
    c_upper = np.zeros_like(a_upper).astype(int)

    assert(a_upper.shape[0] == N*4)
    assert(b_upper.shape[0] == N*4)
    assert(c_upper.shape[0] == N*4)
    assert(((a_upper * b_upper).astype(int)  == c_upper).all())

    a_middle = np.concatenate( [D0,
                                D1,
                                Dd,
                                I,
                                I] +
                                Bis + 
                                [Bs,
                                r0,
                                r1,
                                (a0+b0)-r0,
                                (a1+b1)-r1] +
                                Br +
                                Bdr
                            ).astype(int)

    b_middle = np.concatenate( [a0+b0,
                                a1+b1,
                                np.ones_like(Dd),
                                1-I,
                                Dd-sig] +
                                [1-x for x in Bis] + 
                                [1-Bs,
                                np.zeros(4*N)] +
                                [1-x for x in Br] +
                                [1-x for x in Bdr]
                            ).astype(int)
    c_middle = np.concatenate( [cd*(a0-b0)-r0,
                                cd*(a1-b1)-r1,
                                D0-D1,
                                np.zeros(N),
                                C,
                                np.zeros(N*M) ,
                                np.zeros(M),
                                np.zeros(4*N),
                                np.zeros(4*N*M)]
                            ).astype(int)

    assert(a_middle.shape[0] == 9*N+5*N*M+M)
    assert(b_middle.shape[0] == 9*N+5*N*M+M)
    assert(c_middle.shape[0] == 9*N+5*N*M+M)
    assert(((a_middle * b_middle).astype(int)  == c_middle).all())

    a_lower = np.array([s, sdepi]).astype(int)
    b_lower = np.zeros_like(a_lower).astype(int)
    c_lower = np.zeros_like(a_lower).astype(int)

    assert(a_lower.shape[0] == 2)
    assert(b_lower.shape[0] == 2)
    assert(c_lower.shape[0] == 2)
    assert(((a_lower * b_lower).astype(int)  == c_lower).all())

    a = np.concatenate([a_upper, a_middle, a_lower])
    b = np.concatenate([b_upper, b_middle, b_lower])
    c = np.concatenate([c_upper, c_middle, c_lower])

    assert(((a * b).astype(int)  == c).all())
    
    # Q1
    K_1 = []
    u_1 = []
    v_1 = []
    w_1 = []

    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[i] = 1
        u1[N+i] = 1
        v1[4*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    #Q2
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[2*N+i] = 1
        u1[3*N+i] = 1
        v1[5*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    #Q3
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[i] = cd
        u1[N+i] = -cd
        u1[9*N+M*N+M+i] = -1
        w1[4*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    #Q4
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[2*N+i] = cd
        u1[3*N+i] = -cd
        u1[9*N+M*N+M+N+i] = -1
        w1[5*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q5
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[4*N+i] = 1
        u1[5*N+i] = -1
        w1[6*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q6
    for i in range(N):
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[7*N+i] = 1
        v1[7*N+i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q7
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[7*N+i] = 1
        u1[8*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q8
    for i in range(N):
        k1 = sig
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[6*N+i] = 1
        v1[8*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q9
    for i in range(M*N):
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[9*N+i] = 1
        v1[9*N+i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q10
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[9*N+i*M:9*N+(i+1)*M] = np.array([2**(M-i-1) for i in range(M)])
        w1[8*N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q11
    for i in range(M):
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[9*N+M*N+i] = 1
        v1[9*N+M*N+i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q12
    k1 = 0
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[9*N+N*M:9*N+(N+1)*M] = np.array([2**(M-i-1) for i in range(M)])
    u1[13*N+(5*N+1)*M+1] = -1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    #Q13
    k1 = 0
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[7*N:8*N] = 1
    u1[13*N+(5*N+1)*M] = -1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    #Q14
    k1 = epi
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[13*N+(5*N+1)*M] = 1
    u1[13*N+(5*N+1)*M+1] = -1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    #Q18
    for i in range(4*M*N):
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[13*N+M*N+M+i] = 1
        v1[13*N+M*N+M+i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    #Q19
    for i in range(4*N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[13*N+M*N+M+i*M:13*N+M*N+M+(i+1)*M] = np.array([2**(M-i-1) for i in range(M)])
        u1[9*N+M*N+M+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    
    for i in range(len(K_1)):
        assert(K_1[i]==u_1[i]@a+v_1[i]@b+w_1[i]@c)

    print(f"length: {np.concatenate([a_upper, a_middle, a_lower]).shape}")
    print(f"constraints: {np.array(K_1).shape}")  

    return ([a_upper, a_middle, a_lower], [b_upper, b_middle, b_lower], [c_upper, c_middle, c_lower], 
            K_1, u_1, v_1, w_1)


N = 8

# sample_f = np.load("input/inputData/sample_f.npy").astype("int")
# sample_l = np.load("input/inputData/sample_l.npy").astype("int")
# d = 3
# a, b, c, K, u, v, w = solar_energy([sample_f], [sample_l], 
#                         np.ones_like(sample_f), 
#                         np.ones_like(sample_f), 
#                         [np.ones_like(sample_f)], 
#                         np.ones_like(sample_f),
#                         np.ones_like(sample_f),
#                         0.00000001 * (10**d)**6, 1, d, N)

a0 = np.load("input/inputData/a0.npy").astype("int")
b0 = np.load("input/inputData/b0.npy").astype("int")
a1 = np.load("input/inputData/a1.npy").astype("int")
b1 = np.load("input/inputData/b1.npy").astype("int")

a, b, c, K, u, v, w = bushfire(a0, b0, 
                                a1, 
                                b1, 
                                1, 1, 100, N)

np.savetxt("input/aL.txt", np.concatenate(a).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("input/aO.txt", np.concatenate(c).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("input/aR.txt", np.concatenate(b).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("input/cs.txt", np.array(K).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("input/wL.txt", np.array(u).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("input/wO.txt", np.array(w).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("input/wR.txt", np.array(v).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")

print("constraints generation successful!")