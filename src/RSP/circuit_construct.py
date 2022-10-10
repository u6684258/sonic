import numpy as np

N = 2
T = 1
M = 32
d = 3


def solar_energy(fts, lts, sig_0, sig_1, G_ct, S_GT, G_p, G_e, epi, d):
    D1 = []
    D2 = []
    K = []
    D3 = []
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
    D4 = np.sum(D3, axis=0)
    D5 = D4 * S_GT
    D6 = D5 * G_p
    G = np.sum(D6)
    d_index = int(G_e * epi - G)
    print(f"G={G}")
    print(f"index={d_index}")
    B = np.array([int(x) for x in format(d_index, '032b')])
    print(B.shape)
    a_upper = np.concatenate(fts + lts).astype(int)
    b_upper = np.zeros_like(a_upper).astype(int)
    c_upper = np.zeros_like(a_upper).astype(int)

    assert(a_upper.shape[0] == N*(2*T))
    assert(b_upper.shape[0] == N*(2*T))
    assert(c_upper.shape[0] == N*(2*T))
    assert(((a_upper * b_upper).astype(int)  == c_upper).all())

    a_middle = np.concatenate(fts +
                                D1 +
                                K +
                                K +
                                [D4,
                                D5,
                                B]
                            ).astype(int)

    b_middle = np.concatenate(lts +
                                [sig_0 for _ in range(T)] +
                                [np.ones(T*N)] +
                                G_ct +
                                [S_GT,
                                G_p,
                                1 - B]
                            ).astype(int)
    c_middle = np.concatenate(D1 +
                                D2 +
                                K +
                                D3 +
                                [D5,
                                D6,
                                np.zeros(M)]
                            ).astype(int)

    assert(a_middle.shape[0] == N*(4*T+2)+M)
    assert(b_middle.shape[0] == N*(4*T+2)+M)
    assert(c_middle.shape[0] == N*(4*T+2)+M)
    assert(((a_middle * b_middle).astype(int)  == c_middle).all())
    # print(a_middle)
    # print(b_middle)
    # print(c_middle)
    # print((a_middle * b_middle  // (10**d)).astype(int))

    a_lower = np.array([G, d_index]).astype(int)
    b_lower = np.zeros_like(a_lower).astype(int)
    c_lower = np.zeros_like(a_lower).astype(int)

    assert(a_lower.shape[0] == 2)
    assert(b_lower.shape[0] == 2)
    assert(c_lower.shape[0] == 2)
    assert(((a_lower * b_lower).astype(int)  == c_lower).all())

    a = np.concatenate([a_upper, a_middle, a_lower])
    b = np.concatenate([b_upper, b_middle, b_lower])
    c = np.concatenate([c_upper, c_middle, c_lower])
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
        u1[i] = -1
        u1[2*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        # copy l_i
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[N*T + i] = -1
        v1[2*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        # copy D1
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[3*N*T + i] = -1
        w1[2*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        # copy D2 and 1 - D2 + sigma_1
        k1 = (10**d)**3+sig_1[i]
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[4*N*T + i] = 1
        w1[3*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        # copy Kt
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[5*N*T + i] = -1
        w1[4*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    for i in range(N):
        # copy D5
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[6*N*T + N + i] = -1
        w1[6*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # Q2
    for i in range(N):
        # check D4 == sum D3t
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[6*N*T + i] = -1
        w1[5*N*T + i * T: 5*N*T + (i+1) * T] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    
    # Q3
    for i in range(N):
        # check constants sigma_0
        k1 = sig_0[i]
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        for j in range(T):
            v1[3*N*T + i + j*N] = 1
            K_1.append(k1)
            u_1.append(u1)
            v_1.append(v1)
            w_1.append(w1)
        # check constants 1
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        for j in range(T):
            v1[4*N*T + i + j*N] = 1
            K_1.append(k1)
            u_1.append(u1)
            v_1.append(v1)
            w_1.append(w1)
        # check constants G_ct
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        for j in range(T):
            k1 = G_ct[j][i]
            v1[4*N*T + i + j*N] = 1
            K_1.append(k1)
            u_1.append(u1)
            v_1.append(v1)
            w_1.append(w1)
        # check constants S_Gt
        k1 = S_GT[i]
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        v1[6*N*T + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        # check constants G_p
        k1 = G_p[i]
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        v1[6*N*T+N + i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # Q4
    # check G == sum D6
    k1 = 0
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[6*N*T + 2*N + M] = -1
    w1[6*N*T + N: 6*N*T + 2*N] = 1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    # Q5
    # check G_e * epsilon - G
    k1 = G_e * epi
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[-1] = 1
    u1[-2] = 1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    # Q6
    # check G_e * epsilon - G = 2 * B
    k1 = 0
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[-1] = -1
    u1[6*N*T+2*N:6*N*T+2*N+M] = np.array([2**(M-i-1) for i in range(M)])
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
        u1[6*N*T+2*N+i] =1
        v1[6*N*T+2*N+i] =1
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


    return ([a_upper, a_middle, a_lower], [b_upper, b_middle, b_lower], [c_upper, c_middle, c_lower], 
            K_1, u_1, v_1, w_1)


sample_f = np.load("data/sample_f.npy").astype("int")
sample_l = np.load("data/sample_l.npy").astype("int")


a, b, c, K, u, v, w = solar_energy([sample_f], [sample_l], 
                        np.ones_like(sample_f), 
                        np.ones_like(sample_f), 
                        [np.ones_like(sample_f)], 
                        np.ones_like(sample_f),
                        np.ones_like(sample_f),
                        0.00000001 * (10**d)**6, 0.5, d)

np.savetxt("../../input/aL.txt", np.concatenate(a).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("../../input/aO.txt", np.concatenate(c).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("../../input/aR.txt", np.concatenate(b).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("../../input/cs.txt", np.array(K).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("../../input/wL.txt", np.array(u).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("../../input/wO.txt", np.array(w).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
np.savetxt("../../input/wR.txt", np.array(v).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
