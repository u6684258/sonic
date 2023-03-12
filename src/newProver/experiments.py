from scipy.interpolate import lagrange
import galois
# from tate_bilinear_pairing import eta, ecc, f3m
from py_ecc.bn128 import bn128_curve, bn128_pairing
from py_ecc.fields import (
    bn128_FQ as FQ,
    bn128_FQ2 as FQ2,
    bn128_FQ12 as FQ12,
    bn128_FQP as FQP,
)
import random
import math
import numpy as np
# from poly_utils import PrimeField
from scipy.interpolate import lagrange
from fractions import Fraction
from sklearn.preprocessing import normalize
import time
import pickle

class PrimeField():
    def __init__(self, modulus):
        assert pow(2, modulus, modulus) == 2
        self.modulus = modulus

    def add(self, x, y):
        return (x+y) % self.modulus

    def sub(self, x, y):
        return (x-y) % self.modulus

    def mul(self, x, y):
        return (x*y) % self.modulus

    def exp(self, x, p):
        if p < 0:
            return self.inv(pow(x, -p, self.modulus))
        else:
            return pow(x, p, self.modulus)
    
    def toField(self, x):
        return x % self.modulus
    
    def neg(self, x):
        return -x % self.modulus
    
    
    def moduloMultiplication(self, a, b):
        
        mod = self.modulus
        res = 0; # Initialize result

        # Update a if it is more than
        # or equal to mod
        a = a % mod;
        b = b % mod;
        
        while (b):

            # If b is odd, add a with result
            if (b & 1):
                res = (res + a) % mod;

            # Here we assume that doing 2*a
            # doesn't cause overflow
            a = (2 * a) % mod;

            b >>= 1; # b = b / 2

        return res;

    # Modular inverse using the extended Euclidean algorithm
    def inv(self, a):
        if a == 0:
            return 0
        lm, hm = 1, 0
        low, high = a % self.modulus, self.modulus
        while low > 1:
            r = high//low
            nm, new = hm-lm*r, high-low*r
            lm, low, hm, high = nm, new, lm, low
        return lm % self.modulus

    def multi_inv(self, values):
        partials = [1]
        for i in range(len(values)):
            partials.append(self.mul(partials[-1], values[i] or 1))
        inv = self.inv(partials[-1])
        outputs = [0] * len(values)
        for i in range(len(values), 0, -1):
            outputs[i-1] = self.mul(partials[i-1], inv) if values[i-1] else 0
            inv = self.mul(inv, values[i-1] or 1)
        return outputs

    def div(self, x, y):
        return self.mul(x, self.inv(y))

    # Evaluate a polynomial at a point. The poly must have consecutive orders starting from "init_order".
    def eval_poly_at(self, p, x, init_order = 0):
        y = 0
        power_of_x = 1
        for i, p_coeff in enumerate(p):
            new_y = self.add(self.moduloMultiplication(power_of_x, p_coeff), y)
            y = new_y
            power_of_x = self.mul(power_of_x, x)
#             print(f"power_of_x:{power_of_x}")
#             print(f"p_coeff:{p_coeff}")
#             print(f"mul:{self.moduloMultiplication(power_of_x, p_coeff)+y}")
#             print(f"y:{y}")
#             print(f"new y: {new_y}")  
            
        change_order = self.exp(x, int(init_order))
#         print(f"change_order")

        return self.moduloMultiplication(y, change_order)
    
    def eval_poly_Y(self, coeffs, order, y):
        new_coeffs = []
        for poly in coeffs:
            new_coeffs.append([[self.eval_poly_at(poly[0], y, poly[1])], 0])
        return new_coeffs, order
    
    def eval_poly_X(self, coeffs, init_order, x):
        y = [[0],0]
        power_of_x = 1
        for i, p_coeff in enumerate(coeffs):
            part_poly = self.mul_by_const(p_coeff, power_of_x)
            y = self.add_polys(y[0], part_poly[0], y[1], part_poly[1])
            power_of_x = (power_of_x * x) % self.modulus
        if init_order < 0:
            change_order = self.inv(x ** -init_order)
        else:
            change_order = x ** init_order % self.modulus
        return self.mul_by_const(y, change_order)
        
    # Arithmetic for polynomials
    def add_polys(self, a, b, init_order_a = 0, init_order_b = 0):
        init_order_result = min(init_order_a, init_order_b)
        return ([((a[i-init_order_a+init_order_result] if (i-init_order_a+init_order_result < len(a) and i-init_order_a+init_order_result >= 0) else 0) + 
                  (b[i-init_order_b+init_order_result] if (i-init_order_b+init_order_result < len(b) and i-init_order_b+init_order_result >= 0) else 0))
                % self.modulus for i in range(max(len(a)+init_order_a, len(b)+init_order_b)-init_order_result)], init_order_result)
    
    def sub_polys(self, a, b, init_order_a = 0, init_order_b = 0):
        neg_b = self.mul_by_const([b,init_order_b], -1)
        return self.add_polys(a, neg_b[0], init_order_a, neg_b[1])
    
    
        
        
        # Arithmetic for polynomials
    def add_polys_bivar(self, a, b, init_order_a = 0, init_order_b = 0):
        init_order_result = min(init_order_a, init_order_b)
        result_poly = []
        for i in range(max(len(a)+init_order_a, len(b)+init_order_b)-init_order_result):
            poly_a = a[i-init_order_a+init_order_result] if (i-init_order_a+init_order_result < len(a) and i-init_order_a+init_order_result >= 0) else [[0], 0]
            poly_b = b[i-init_order_b+init_order_result] if (i-init_order_b+init_order_result < len(b) and i-init_order_b+init_order_result >= 0) else [[0], 0]
#             print(poly_a)
#             print(poly_b)
            result_poly.append(field.add_polys(poly_a[0], poly_b[0], poly_a[1], poly_b[1]))
        return result_poly, init_order_result

    
    def mul_by_const(self, a, c):
        return [(x*c) % self.modulus for x in a[0]], a[1]
    
    def mul_by_const_bivar(self, a, c):
        return [self.mul_by_const(x, c) for x in a[0]], a[1]
    
    def mul_polys(self, a, b, init_order_a = 0, init_order_b = 0):
        o = [0] * (len(a) + len(b) - 1)
        for i, aval in enumerate(a):
            for j, bval in enumerate(b):
                addon = self.moduloMultiplication(a[i], b[j])
                o[i+j] = (o[i+j] + addon) % self.modulus
        return [x % self.modulus for x in o], init_order_a + init_order_b
    
    def mul_polys_bivar(self, a, b, init_order_a = 0, init_order_b = 0):
        o = [[[0], 0]] * (len(a) + len(b) - 1)
        for i, aval in enumerate(a):
            for j, bval in enumerate(b):
                mul_result = self.mul_polys(a[i][0], b[j][0], a[i][1], b[j][1])
                o[i+j] = self.add_polys(o[i+j][0], mul_result[0], o[i+j][1], mul_result[1])

        return o, init_order_a + init_order_b
    
    
    def div_polys(self, a, b, init_order_a = 0, init_order_b = 0):
        assert len(a) >= len(b)
        a = [x for x in a]
        o = []
        apos = len(a) - 1
        bpos = len(b) - 1
        diff = apos - bpos
        while diff >= 0:
            quot = self.div(a[apos], b[bpos])
            o.insert(0, quot)
            for i in range(bpos, -1, -1):
                a[diff+i] -= self.moduloMultiplication(b[i], quot)
            apos -= 1
            diff -= 1
        return [x % self.modulus for x in o], init_order_a - init_order_b

#     def mod_polys(self, a, b):
#         return self.sub_polys(a, self.mul_polys(b, self.div_polys(a, b)))[:len(b)-1]

    # Build a polynomial from a few coefficients, together with init_order
    def sparse(self, coeff_dict):
        degree = max(coeff_dict.keys()) - min(coeff_dict.keys())
        o = [0] * (degree + 1)
        for k, v in coeff_dict.items():
            o[k - min(coeff_dict.keys())] = v % self.modulus
        return (o, min(coeff_dict.keys()))
    
    def sparse_bivar(self, coeff_dict):
        degree = max(coeff_dict.keys()) - min(coeff_dict.keys())
        o = [[[0], 0]] * (degree + 1)
#         print(o)
        for k, v in coeff_dict.items():
            o[k - min(coeff_dict.keys())] = v
        return (o, min(coeff_dict.keys()))
    
    def lagrange(self, xs, ys):
        fn = None
        for i, y in enumerate(ys):
            xlist = xs[:i]+xs[i+1:]
            xpoly = [list(np.array(np.poly1d(xlist, True).c)[::-1].astype(int)), 0]
            denominator = 1
            for x in xlist:
                denominator *= xs[i] - x
                
            pi = self.mul_by_const(xpoly, self.inv(denominator))
            pi = self.mul_by_const(pi, y)
            
            if fn is None:
                fn = pi
            else:
                fn = self.add_polys(fn[0], pi[0], fn[1], pi[1])
                
        return fn
    
    def isZero(self, a):
        return np.all(np.array(a) == 0)
    
    def dimension_change(self, a, init_order):
        newPoly = []
        for i in a:
            newPoly.append(i[0][0])
        return newPoly, init_order

def request_data(size):
    nir = np.load("../../data/nir.npy").astype("int")
    swir = np.load("../../data/swir.npy").astype("int")
    nir = nir - nir.min()
    swir = swir - swir.min()
    sample_f = nir[50,:size]
    sample_l = swir[50,:size]
    a1 = nir[100,:size]
    b1 = swir[100,:size]

    np.save("../../input/inputData/sample_f.npy", sample_f)
    np.save("../../input/inputData/sample_l.npy", sample_l)

    np.save("../../input/inputData/a0.npy", sample_f)
    np.save("../../input/inputData/b0.npy", sample_l)
    np.save("../../input/inputData/a1.npy", a1)
    np.save("../../input/inputData/b1.npy", b1)

def bushfire(a0, b0, a1, b1, sig, epi, cd, N):

    M = 32

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
    
    sum_r = np.sum(r0**2+r1**2)
    limit_r = sum_r + 1
    
    assert((C >= 0).all())
    print(f"G={s}")
    print(f"index={sdepi}")

    Bis = [np.array([int(x) * 2**(M-j-1) for j, x in enumerate(format(C[i], f'0{M}b'))]) for i in range(len(C))]
#     Bs = np.array([int(x) for x in format(sdepi, f'0{M}b')])
    Br = np.array([int(x) * 2**(M-j-1) for j, x in enumerate(format(limit_r - sum_r, f'0{M}b'))])
    BG = np.array([int(x) * 2**(M-j-1) for j, x in enumerate(format(sdepi, f'0{M}b'))])
        
    array2 = np.array([2**(M-i-1) for i in range(M)])
#     print(format(np.sum(r0**2+r1**2), f'0{M}b'))
#     Bdr = [np.array([int(x) for x in format((a0+b0)[i]-r0[i], f'0{M}b')]) for i in range(len(r0))] +\
#             [np.array([int(x) for x in format((a1+b1)[i]-r1[i], f'0{M}b')]) for i in range(len(r1))]

    a_lower = np.concatenate([a0, b0, a1, b1]).astype(int)
    b_lower = np.zeros_like(a_lower).astype(int)
    c_lower = np.zeros_like(a_lower).astype(int)

    assert(a_lower.shape[0] == N*4)
    assert(b_lower.shape[0] == N*4)
    assert(c_lower.shape[0] == N*4)
    assert(((a_lower * b_lower).astype(int)  == c_lower).all())

    a_middle = np.concatenate( [D0,
                                D1,
                                I] +
                                Bis + 
                                [I,
                                 Br,
                                 BG,
                                r0,
                                r1]
                            ).astype(int)
    b_middle = np.concatenate( [a0+b0,
                                a1+b1,
                                1-I] +
                                [array2-x for x in Bis] + 
                                [Dd-sig,
                                 array2-Br,
                                 array2-BG,
                                r0,
                                r1]
                            ).astype(int)
    c_middle = np.concatenate( [cd*(a0-b0)-r0,
                                cd*(a1-b1)-r1,
                                np.zeros(N),
                                np.zeros(N*M) ,
                                C,
                                np.zeros(M),
                                np.zeros(M),
                                r0**2,
                                r1**2]
                            ).astype(int)

    assert(a_middle.shape[0] == 6*N+N*M+2*M)
    assert(b_middle.shape[0] == 6*N+N*M+2*M)
    assert(c_middle.shape[0] == 6*N+N*M+2*M)
    assert(((a_middle * b_middle).astype(int)  == c_middle).all())


    a = np.concatenate([a_middle, a_lower])
    b = np.concatenate([b_middle, b_lower])
    c = np.concatenate([c_middle, c_lower])

    assert(((a * b).astype(int)  == c).all())
    
    # Q1
    K_1 = []
    u_1 = []
    v_1 = []
    w_1 = []
    
    # r- + s-
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[6*N+(N+2)*M+i] = 1
        u1[7*N+(N+2)*M+i] = 1
        v1[i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    # r- - s-
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[4*N+(N+2)*M+i] = -1
        u1[6*N+(N+2)*M+i] = cd
        u1[7*N+(N+2)*M+i] = -cd
        w1[i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    # r+ + s+
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[8*N+(N+2)*M+i] = 1
        u1[9*N+(N+2)*M+i] = 1
        v1[N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
    # r+ - s+
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[5*N+(N+2)*M+i] = -1
        u1[8*N+(N+2)*M+i] = cd
        u1[9*N+(N+2)*M+i] = -cd
        w1[N+i] = -1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # i + (1-i) = 1
    for i in range(N):
        k1 = 1
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[2*N+i] = 1
        v1[2*N+i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        
    # e_{i,j} - (e_{i,j}-2^{j-1}) = 2^{j-1}
    for i in range(N):
        for j in range (M):
            k1 = array2[j]
            u1 = np.zeros_like(a)
            v1 = np.zeros_like(b)
            w1 = np.zeros_like(c)
            u1[3*N+M*i+j] = 1
            v1[3*N+M*i+j] = 1
            K_1.append(k1)
            u_1.append(u1)
            v_1.append(v1)
            w_1.append(w1)

    # n- - n+ + \kappa 
    for i in range(N):
        k1 = -sig
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[i] = -1
        u1[N+i] = 1
        v1[3*N+M*N+i] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # C = \sum e_{i,j}
    for i in range(N):
        k1 = 0
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        w1[3*N+M*N+i] = -1
        u1[3*N+M*i:3*N+M*i+M] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # G+\epsilon = \sum i
    k1 = epi
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[4*N+M*(N+1):4*N+M*(N+2)] = -1
    u1[2*N:3*N] = 1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    # lim - sum \delta^2 = e_lim
    k1 = limit_r
    u1 = np.zeros_like(a)
    v1 = np.zeros_like(b)
    w1 = np.zeros_like(c)
    u1[4*N+M*N:4*N+M*(N+1)] = 1
    w1[4*N+M*(N+2):6*N+M*(N+2)] = 1
    K_1.append(k1)
    u_1.append(u1)
    v_1.append(v1)
    w_1.append(w1)

    # e_{lim,j} - (e_{lim,j}-2^{j-1}) = 2^{j-1}
    for j in range (M):
        k1 = array2[j]
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[4*N+M*N+j] = 1
        v1[4*N+M*N+j] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)
        
    # e_{G,j} - (e_{G,j}-2^{j-1}) = 2^{j-1}
    for j in range (M):
        k1 = array2[j]
        u1 = np.zeros_like(a)
        v1 = np.zeros_like(b)
        w1 = np.zeros_like(c)
        u1[4*N+M*(N+1)+j] = 1
        v1[4*N+M*(N+1)+j] = 1
        K_1.append(k1)
        u_1.append(u1)
        v_1.append(v1)
        w_1.append(w1)

    # a = np.array([4, 9])
    # b = np.array([9, 4])
    # c = np.array([36, 36])
    # u_1 = np.array([[0, 0], 
    #             [1, 0], 
    #             [0, 1], 
    #             [0, 0], 
    #             [0, 0]])
    # v_1 = np.array([[0, 0], 
    #             [0, 0], 
    #             [0, 0], 
    #             [1, 0], 
    #             [0, 1]])
    # w_1 = np.array([[1, -1], 
    #             [0, 0], 
    #             [0, 0], 
    #             [0, 0], 
    #             [0, 0]])
    # K_1 = np.array([0, 4, 9, 9, 4])
    for i in range(len(K_1)):
        assert(K_1[i]==u_1[i]@a+v_1[i]@b+w_1[i]@c)

    print(f"length: {np.concatenate([a_middle, a_lower]).shape}")
    print(f"constraints: {np.array(K_1).shape}")  
    
    np.savetxt("../../input/aL.txt", a.astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/aO.txt", c.astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/aR.txt", b.astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/cs.txt", np.array(K_1).astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/wL.txt", np.array(u_1).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/wO.txt", np.array(w_1).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/wR.txt", np.array(v_1).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")

    return (a, 
            b, 
            c, 
            np.array(K_1), np.array(u_1), np.array(v_1), np.array(w_1))

order = bn128_curve.curve_order
# GF_curve = galois.GF(order)
# GF_field = galois.GF(bn128_curve.field_modulus)
field = PrimeField(order)
size = 4
srsX = 12
srsAlpha = 10

def rPoly(aL, aR, aO, n):
    list_of_coeff = np.concatenate([np.flip(aO), np.flip(aR), aL])
    list_of_power = np.concatenate([np.arange(-2*n, 0), np.arange(1,n+1)])
    list_of_bi_coeff = []
    for i in range(len(list_of_coeff)):
        dummy_dict = {}
        dummy_dict[list_of_power[i]] = list_of_coeff[i]
        list_of_bi_coeff.append(field.sparse(dummy_dict))
    return field.sparse_bivar(dict(zip(list_of_power, list_of_bi_coeff)))


def sPoly(u,v,w, n, q):
    uiYs = []
    viYs = []
    wiYs = []
    for i in range(n):
        uiYs.insert(0,field.sparse(dict(zip(np.arange(n+1, n+q+1), u[:,i]))))
        viYs.append(field.sparse(dict(zip(np.arange(n+1, n+q+1), v[:,i]))))
        wiPart1 = field.sparse(dict(zip(np.arange(n+1, n+q+1), w[:,i])))
        wiPart2 = field.sparse(dict(zip([-i-1, i+1], [-1, -1])))
        wiYs.append(field.add_polys(wiPart1[0], wiPart2[0], wiPart1[1], wiPart2[1]))
#     return np.concatenate([uiYs, viYs, wiYs], dtype=object)
    return field.sparse_bivar(dict(zip(np.concatenate([np.arange(-n, 0), np.arange(1,2*n+1)]), np.concatenate([uiYs, viYs, wiYs], dtype=object))))

def kPoly(k, n, q):
    return [[field.mul_by_const(field.sparse(dict(zip(np.arange(n+1, n+q+1), k))), -1)], 0]


class KZGCommitment():
    def __init__(self, n, srsX, srsAlpha, field):
        self.G1 = bn128_curve.G1
        self.G2 = bn128_curve.G2
        self.srsD = n * 6
        self.gNegativeX = [bn128_curve.multiply(bn128_curve.G1, field.exp(srsX, -i)) for i in range(1,self.srsD)]
        self.gPositiveX = [bn128_curve.multiply(bn128_curve.G1, field.exp(srsX, i)) for i in range(0,self.srsD)]
        # hNegativeX = [bn128_curve.multiply(bn128_curve.G2, field.exp(srsX, -i)) for i in range(1,srsD)]
        self.hPositiveX = [bn128_curve.multiply(bn128_curve.G2, field.exp(srsX, i)) for i in range(0,2)]
        self.field = field
# gNegativeAlphaX = [bn128_curve.multiply(bn128_curve.G1, field.mul(srsAlpha, field.exp(srsX, -i))) for i in range(1,srsD)]
# gPositiveAlphaX = [bn128_curve.multiply(bn128_curve.G1, field.mul(srsAlpha, field.exp(srsX, i))) for i in range(1,srsD)]
# hNegativeAlphaX = [bn128_curve.multiply(bn128_curve.G2, field.mul(srsAlpha, field.exp(srsX, -i))) for i in range(1,srsD)]
# hPositiveAlphaX = [bn128_curve.multiply(bn128_curve.G2, field.mul(srsAlpha, field.exp(srsX, i))) for i in range(0,srsD)]

    def commit(self, p, init_order):
        c = None
        for i in range(len(p)):
            if init_order + i < 0:
                if c is None:
                    c = bn128_curve.multiply(self.gNegativeX[abs(init_order + i)-1], p[i])
                else:
                    c = bn128_curve.add(bn128_curve.multiply(self.gNegativeX[abs(init_order + i)-1], p[i]), c)
            else:
                if c is None:
                    c = bn128_curve.multiply(self.gPositiveX[init_order + i], p[i])
                else:
                    c = bn128_curve.add(bn128_curve.multiply(self.gPositiveX[init_order + i], p[i]), c)
        return c
    
    
    def openC(self, c, z, p, init_order):
        fz = self.field.eval_poly_at(p, z, init_order)
        dummy_dict = {}
        dummy_dict[0] = fz
        dummy_poly = self.field.mul_by_const(self.field.sparse(dummy_dict), -1)
        numerator = self.field.add_polys(p, dummy_poly[0], init_order, dummy_poly[1])
        dummy_dict = {}
        dummy_dict[0] = -z
        dummy_dict[1] = 1
        denominator = self.field.sparse(dummy_dict)
        qx = self.field.div_polys(numerator[0], denominator[0], numerator[1], denominator[1])
        return self.commit(qx[0], qx[1]), fz
    
    def verify(self, c, z, fz, w):
        leftleft = bn128_curve.add(c, bn128_curve.multiply(self.G1, self.field.neg(fz)))
        leftright = self.hPositiveX[0]
        rightleft = w
        rightright = bn128_curve.add(self.hPositiveX[1], bn128_curve.multiply(self.G2, self.field.neg(z)))
        
        e_left = bn128_pairing.pairing(leftright, leftleft)
        e_right = bn128_pairing.pairing(rightright, rightleft)
        return e_left == e_right
    

# cmScheme = KZGCommitment(n, srsX, srsAlpha, field)

# r1X = field.dimension_change(rX1[0], rX1[1])
# r1X
# c = cmScheme.commit(r1X[0], r1X[1])
# c
# # bn128_curve.is_on_curve(c, bn128_curve.b)
# proof = cmScheme.openC(c, 1, r1X[0], r1X[1])
# proof

# # c = cmScheme.commit(tXy[0], tXy[1])
# # proof = cmScheme.openC(c, y, tXy[0], tXy[1])
# verify = cmScheme.verify(c, 1, proof[1], proof[0])

# verify

def generate_srs(n, srsX, srsAlpha):
    srsD = n * 7
    gNegativeX = np.array([bn128_curve.multiply(bn128_curve.G1, field.exp(srsX, -i)) for i in range(1,srsD)])
    np.save("gNegativeX", gNegativeX, allow_pickle=True)
    gPositiveX = np.array([bn128_curve.multiply(bn128_curve.G1, field.exp(srsX, i)) for i in range(0,srsD)])
    np.save("gPositiveX", gPositiveX, allow_pickle=True)
    hNegativeX = np.array([bn128_curve.multiply(bn128_curve.G2, field.exp(srsX, -i)) for i in range(1,srsD)]).astype(str)
    np.save("hNegativeX", hNegativeX, allow_pickle=True)
    hPositiveX = np.array([bn128_curve.multiply(bn128_curve.G2, field.exp(srsX, i)) for i in range(0,srsD)]).astype(str)
    np.save("hPositiveX", hPositiveX, allow_pickle=True)
    gNegativeAlphaX = np.array([bn128_curve.multiply(bn128_curve.G1, field.mul(srsAlpha, field.exp(srsX, -i))) for i in range(1,srsD)])
    np.save("gNegativeAlphaX", gNegativeAlphaX, allow_pickle=True)
    gPositiveAlphaX = np.array([bn128_curve.multiply(bn128_curve.G1, field.mul(srsAlpha, field.exp(srsX, i))) for i in range(1,srsD)])
    np.save("gPositiveAlphaX", gPositiveAlphaX, allow_pickle=True)
    hNegativeAlphaX = np.array([bn128_curve.multiply(bn128_curve.G2, field.mul(srsAlpha, field.exp(srsX, -i))) for i in range(1,srsD)]).astype(str)
    np.save("hNegativeAlphaX", hNegativeAlphaX, allow_pickle=True)
    hPositiveAlphaX = np.array([bn128_curve.multiply(bn128_curve.G2, field.mul(srsAlpha, field.exp(srsX, i))) for i in range(0,srsD)]).astype(str)
    np.save("hPositiveAlphaX", hPositiveAlphaX, allow_pickle=True)
    
def load_srs(length, srsX):
    gNegativeXf = np.load('gNegativeX.npy', allow_pickle=True)
    gNegativeX = gNegativeXf[0:length, :]
    gPositiveXf = np.load('gPositiveX.npy', allow_pickle=True)
    gPositiveX = gPositiveXf[0:length, :]
    hNegativeXf = np.load('hNegativeX.npy', allow_pickle=True)
    hNegativeX = np.char.split(hNegativeXf[0:length, :].astype(str), ",") 
    hPositiveXf = np.load('hPositiveX.npy', allow_pickle=True)
    hPositiveX = np.char.split(hPositiveXf[0:length, :].astype(str), ",")
    gNegativeAlphaXf = np.load('gNegativeAlphaX.npy', allow_pickle=True)
    gNegativeAlphaX = gNegativeAlphaXf[0:length, :]
    gPositiveAlphaXf = np.load('gPositiveAlphaX.npy', allow_pickle=True)
    gPositiveAlphaX = gPositiveAlphaXf[0:length, :]
    hNegativeAlphaXf = np.load('hNegativeAlphaX.npy', allow_pickle=True)
    hNegativeAlphaX = np.char.split(hNegativeAlphaXf[0:length, :].astype(str), ",")
    hPositiveAlphaXf = np.load('hPositiveAlphaX.npy', allow_pickle=True)
    hPositiveAlphaX = np.char.split(hPositiveAlphaXf[0:length, :].astype(str), ",")
    
    for i in range(hNegativeX.shape[0]):
        hNegativeX[i][0][0] = int(hNegativeX[i][0][0].strip("()"))
        hNegativeX[i][0][1] = int(hNegativeX[i][0][1].strip("()"))
        hNegativeX[i][0] = FQ2(hNegativeX[i][0])
        hPositiveX[i][0][0] = int(hPositiveX[i][0][0].strip("()"))
        hPositiveX[i][0][1] = int(hPositiveX[i][0][1].strip("()"))
        hPositiveX[i][0] = FQ2(hPositiveX[i][0])
        hNegativeAlphaX[i][0][0] = int(hNegativeAlphaX[i][0][0].strip("()"))
        hNegativeAlphaX[i][0][1] = int(hNegativeAlphaX[i][0][1].strip("()"))
        hNegativeAlphaX[i][0] = FQ2(hNegativeAlphaX[i][0])
        hPositiveAlphaX[i][0][0] = int(hPositiveAlphaX[i][0][0].strip("()"))
        hPositiveAlphaX[i][0][1] = int(hPositiveAlphaX[i][0][1].strip("()"))
        hPositiveAlphaX[i][0] = FQ2(hPositiveAlphaX[i][0])
        
        hNegativeX[i][1][0] = int(hNegativeX[i][1][0].strip("()"))
        hNegativeX[i][1][1] = int(hNegativeX[i][1][1].strip("()"))
        hNegativeX[i][1] = FQ2(hNegativeX[i][1])
        hPositiveX[i][1][0] = int(hPositiveX[i][1][0].strip("()"))
        hPositiveX[i][1][1] = int(hPositiveX[i][1][1].strip("()"))
        hPositiveX[i][1] = FQ2(hPositiveX[i][1])
        hNegativeAlphaX[i][1][0] = int(hNegativeAlphaX[i][1][0].strip("()"))
        hNegativeAlphaX[i][1][1] = int(hNegativeAlphaX[i][1][1].strip("()"))
        hNegativeAlphaX[i][1] = FQ2(hNegativeAlphaX[i][1])
        hPositiveAlphaX[i][1][0] = int(hPositiveAlphaX[i][1][0].strip("()"))
        hPositiveAlphaX[i][1][1] = int(hPositiveAlphaX[i][1][1].strip("()"))
        hPositiveAlphaX[i][1] = FQ2(hPositiveAlphaX[i][1])
        
        hNegativeX[i] = tuple(hNegativeX[i])
        hPositiveX[i] = tuple(hPositiveX[i])
        hNegativeAlphaX[i] = tuple(hNegativeAlphaX[i])
        hPositiveAlphaX[i] = tuple(hPositiveAlphaX[i])
        
    
    return gNegativeX, gPositiveX, hPositiveX, hNegativeX, gNegativeAlphaX, gPositiveAlphaX, hPositiveAlphaX, hNegativeAlphaX
    
# generate_srs(5000, srsX, srsAlpha)
# load_srs(10, srsX)

class KZGBatchCommitment():
    def __init__(self, n, srsX, srsAlpha, field):
        self.G1 = bn128_curve.G1
        self.G2 = bn128_curve.G2
        self.srsD = n * 8
        srss = load_srs(self.srsD, srsX)
        self.gNegativeX = srss[0]
        self.gPositiveX = srss[1]
        self.hPositiveX = srss[2]
        self.hNegativeX = srss[3]
        self.gNegativeAlphaX = srss[4]
        self.gPositiveAlphaX = srss[5]
        self.hPositiveAlphaX = srss[6]
        self.hNegativeAlphaX = srss[7]

        self.field = field
        self.srsX = srsX

    def commit(self, list_of_p, list_of_init_order, g_max):
        list_of_c = []
        for j, p in enumerate(list_of_p):
            c = None
            init_order = list_of_init_order[j]
            for i in range(len(p)):
                # index = init_order + i + self.srsD - g_max
                index = init_order + i + self.srsD - g_max
                if index < 0:
                    if c is None:
                        c = bn128_curve.multiply(self.gNegativeX[abs(index)-1], p[i])
                    else:
                        c = bn128_curve.add(bn128_curve.multiply(self.gNegativeX[abs(index)-1], p[i]), c)
                else:
                    if c is None:
                        c = bn128_curve.multiply(self.gPositiveX[index], p[i])
                    else:
                        c = bn128_curve.add(bn128_curve.multiply(self.gPositiveX[index], p[i]), c)
            list_of_c.append(c)
            
        return list_of_c
    
    def commita(self, list_of_p, list_of_init_order, g_max):
        list_of_c = []
        for j, p in enumerate(list_of_p):
            c = None
            init_order = list_of_init_order[j]
            for i in range(len(p)):
                index = init_order + i + self.srsD - g_max
                # index = init_order + i
                if index < 0:
                    if c is None:
                        c = bn128_curve.multiply(self.gNegativeAlphaX[abs(index)-1], p[i])
                    else:
                        c = bn128_curve.add(bn128_curve.multiply(self.gNegativeAlphaX[abs(index)-1], p[i]), c)
                elif index == 0:
                    print(f"zero term with coefficient {p[i]}")
                else:
                    if c is None:
                        c = bn128_curve.multiply(self.gPositiveAlphaX[index-1], p[i])
                    else:
                        c = bn128_curve.add(bn128_curve.multiply(self.gPositiveAlphaX[index-1], p[i]), c)
            list_of_c.append(c)
            
        return list_of_c
    
    
    def openC(self, list_of_c, list_of_z_for_p, list_of_p, list_of_init_order, g_max):
        
        # evaluate all polynomials at all points and then interpolations
        list_of_fz = []
        list_of_all_z = list(set().union(*list_of_z_for_p))
        list_of_gamma = []
        zTx = list(np.array(np.poly1d(list_of_all_z, True).c)[::-1].astype(int))
#         print(zTx)
        list_of_zSix = []
        fx = [[0],0]
        fxz = [[0],0]
        beta = 1
        rand_z = 2
        for i, p in enumerate(list_of_p):
            list_of_fz_p = []
            # compute all p(z) for z \in S_i
            for z in list_of_z_for_p[i]:
                list_of_fz_p.append(self.field.eval_poly_at(p, z, list_of_init_order[i]))
                
            list_of_fz.append(list_of_fz_p)
            
#             print(list_of_fz_p)
            
            # compute r_i
            gamma_p = self.field.lagrange(list_of_z_for_p[i], list_of_fz_p)
#             print(f"gamma_p: {gamma_p}")
            # compute z_{T\Si}
            SExSi = list(set(list_of_all_z) ^ set(list_of_z_for_p[i]))
            zSix = list(np.array(np.poly1d(SExSi, True).c)[::-1].astype(int))
#             print(zSix)

            # f_i(x)-r_i(x)
            sub_poly = self.field.sub_polys(p, gamma_p[0], list_of_init_order[i], gamma_p[1])
            # z_{T\Si}*[f_i(x)-r_i(x)]
            part_poly = self.field.mul_polys(zSix, sub_poly[0], 0, sub_poly[1])
            # beta^i * z_{T\Si}*[f_i(x)-r_i(x)]
            part_poly = self.field.mul_by_const(part_poly, self.field.exp(beta, i))
            # summation
            fx = self.field.add_polys(fx[0], part_poly[0], fx[1], part_poly[1])
#             test = 1
#             print(self.field.eval_poly_at(p, test, list_of_init_order[i]))
#             print(self.field.eval_poly_at(gamma_p[0], test, gamma_p[1]))
#             print(self.field.eval_poly_at(part_poly[0], test, part_poly[1]))
            
            # r_i(z)
            eval_gamma = [[self.field.eval_poly_at(gamma_p[0], rand_z, gamma_p[1])],0]
            list_of_gamma.append(eval_gamma[0][0])
#             print(eval_gamma)
            # compute z_{T\Si}(z)
            eval_zSix = [[self.field.eval_poly_at(zSix, rand_z, 0)],0]
            list_of_zSix.append(eval_zSix[0][0])
#             print(eval_zSix)
            # f_i(x)-r_i(z)
            sub_poly = self.field.sub_polys(p, eval_gamma[0], list_of_init_order[i], eval_gamma[1])
            # z_{T\Si}(z)*[f_i(x)-r_i(z)]
            part_poly = self.field.mul_polys(eval_zSix[0], sub_poly[0], 0, sub_poly[1])
            # beta^i * z_{T\Si}(z)*[f_i(x)-r_i(z)]
            part_poly = self.field.mul_by_const(part_poly, self.field.exp(beta, i))
#             print(part_poly)
            fxz = self.field.add_polys(fxz[0], part_poly[0], fxz[1], part_poly[1])
        
        # f/Z_T
        px = self.field.div_polys(fx[0], zTx, fx[1], 0)
#         print(px)
#         print(fx)
#         print(zTx)
        test = 6
#         print(self.field.div(self.field.eval_poly_at(fx[0], test, fx[1]),self.field.eval_poly_at(zTx, test, 0)))
#         print(self.field.eval_poly_at(px[0], test, px[1]))
#         print(self.field.eval_poly_at(fx[0], test, fx[1]))
#         print(self.field.eval_poly_at(zTx, test, 0))
        
        
        # Z_T(z)
        eval_zt = self.field.eval_poly_at(zTx, rand_z, 0)
        # f/Z_T * Z_T(z)
        l_second_half = self.field.mul_by_const(px, eval_zt)
        # L = beta^i * z_{T\Si}(z)*[f_i(x)-r_i(z)] - f/Z_T * Z_T(z)
        lx = self.field.sub_polys(fxz[0], l_second_half[0], fxz[1], l_second_half[1])
#         print(self.field.eval_poly_at(lx[0], 13295, lx[1])/(13295-rand_z))
        # L / (x-z)
        lx = self.field.div_polys(lx[0], [-rand_z, 1], lx[1], 0)
#         print(self.field.eval_poly_at(lx[0], 13295, lx[1]))
#         print(lx)
        
        w = self.commit([px[0]], [px[1]], g_max)
        wDash = self.commit([lx[0]], [lx[1]], g_max)  
        
        return w[0], wDash[0], rand_z, beta, list_of_fz, list_of_gamma, list_of_zSix, eval_zt, g_max
        
    def verify(self, list_of_c, w, wDash, rand_z, beta, list_of_fz, list_of_gamma, list_of_zSix, eval_zt, g_max):
        
        cm_poly = None
        ga_poly = 0
        for i, c in enumerate(list_of_c):
            gamma_p = list_of_gamma[i]
            zSix = list_of_zSix[i]
#             print(gamma_p)
#             print(zSix)
            if cm_poly == None:
                cm_poly = bn128_curve.multiply(c, self.field.mul(zSix, self.field.exp(beta, i)))
            else:
                cm_poly = bn128_curve.add(bn128_curve.multiply(c, self.field.mul(zSix, self.field.exp(beta, i))), cm_poly)
#             print(cm_poly)
            ga_poly = self.field.add(self.field.mul(gamma_p, self.field.mul(zSix, self.field.exp(beta, i))), ga_poly)
        
        w_poly = bn128_curve.multiply(w, eval_zt)
        # f = bn128_curve.add(cm_poly, w_poly)
        
        non_a_part = bn128_curve.neg(bn128_curve.add(w_poly, bn128_curve.multiply(bn128_curve.G1, ga_poly)))
        
        rr = bn128_curve.add(non_a_part, bn128_curve.multiply(wDash, rand_z))
        non_a_part_w_a = bn128_curve.multiply(rr, srsAlpha*srsX**(self.srsD - g_max))
        left1right = self.hPositiveX[0].astype(FQ2)
        # alpha_part = bn128_curve.multiply(bn128_curve.G1, 1)
        left2left = bn128_curve.add(cm_poly, non_a_part_w_a)
        # left2right = self.hPositiveAlphaX[self.srsD - g_max]
        rightleft = wDash
        rightright = self.hPositiveAlphaX[1 + self.srsD - g_max].astype(FQ2)

        print(f""
              f"H: {cm_poly}\n"
              f"RR: {rr}\n"
              f"gwx: {wDash}\n"
              f"h: {left1right}\n"
              f"h_alpha_d-max: {self.hPositiveAlphaX[self.srsD - g_max].astype(FQ2)}\n"
              f"h_alpha_d-max+1: {rightright}\n"
              )
        
        e_left = bn128_pairing.pairing(left1right, left2left)
        e_right = bn128_pairing.pairing(rightright, rightleft)
        return e_left == e_right
#         return True
    

def parse_number(str_in):
    i = 0
    while str_in[i].isdigit() or str_in[i] == "-":
        i += 1

    return str_in[:i], i

def parse_term(str_in):
    str_in = str_in.strip()
    # print(str_in)
    # print(str_in.split(")"))
    order = int(str_in.split(",")[0][1:])
    coeff = int(str_in.split(")")[0].split("P")[1])
    if len(str_in.split(")")[1].strip()) == 0:
        pow = 0
    else:
        if len(str_in.split(")")[1].strip().split("^")) == 1:
            pow = 1
        else:
            pow = int(str_in.split(")")[1].split("^")[1])

    return order, coeff, pow

def make_poly(coeffs):
    init_order = coeffs[-1][-1]
    coeff_dict = {}
    for coef in coeffs:
        coeff_dict[coef[0]+init_order] = coef[1]

    return field.sparse(coeff_dict)

def make_poly_bivar(coeffs):
    init_order = coeffs[-1][-1]
    coeff_dict = {}
    for coef in coeffs:
        coeff_dict[coef[0]+init_order] = coef[1]

    return field.sparse_bivar(coeff_dict)


def read_poly(string_in):
    new_poly = [[], 0]
    tokens = string_in.split("+")
    list_of_tokens = []
    main_poly = []
    for token in tokens:
        if token.count("(") == 2 and token.count(")") == 1:
            list_of_tokens = []
            tos = token.split("(")
            order = int(tos[1][:-1])
            coeff = parse_term("("+tos[2])
            list_of_tokens.append(order)
            list_of_tokens.append(coeff)
        
        elif token.count("(") == 1 and token.count(")") == 1:
            list_of_tokens.append(parse_term(token))

        elif token.count("(") == 1 and token.count(")") == 2:
            tos = token.split(")")
            coeff = parse_term(tos[0]+")"+tos[1])
            list_of_tokens.append(coeff)
            if len(tos) == 2:
                pow = 0
            elif len(tos[2].split("^")) == 1:
                pow = 1
            else:
                pow = int(tos[2].split("^")[1])

            list_of_tokens.append(pow)

            # print(make_poly(list_of_tokens[1:-1]))
            main_poly.append((list_of_tokens[0], make_poly(list_of_tokens[1:-1]), list_of_tokens[-1]))

        elif token.count("(") == 2 and token.count(")") == 2:
            tos = token.split("(")
            order = int(tos[1][:-1])
            toss = tos[2].split(")")
            # print(toss)
            coeff = parse_term("("+toss[0]+")"+toss[1])

            if len(toss) == 2:
                pow = 0
            elif len(toss[2].split("^")) == 1:
                pow = 1
            else:
                pow = int(toss[2].split("^")[1])

            # print(make_poly([coeff]))
            main_poly.append((order, make_poly([coeff]), pow))

    return make_poly_bivar(main_poly)
            

def setup_data(size):

    request_data(size)
    a0 = np.load("../../input/inputData/a0.npy").astype("int")
    b0 = np.load("../../input/inputData/b0.npy").astype("int")
    a1 = np.load("../../input/inputData/a1.npy").astype("int")
    b1 = np.load("../../input/inputData/b1.npy").astype("int")

    aL, aR, aO, k, u, v, w = bushfire(a0, b0, 
                                    a1, 
                                    b1, 
                                    1, 1, 100, size)

    assignment = [aL, aR, aO]
    circuit = [u,v,w,k]

    assert (aL @ u.T + aR @ v.T + aO @ w.T == k).all()

    n = aL.shape[0]
    q = k.shape[0]

    return aL, aR, aO, k, u, v, w, n, q

def sonic_experiment(size, aL, aR, aO, k, u, v, w, n, q, save=False, load=False):
    
    cmScheme = KZGBatchCommitment(n, srsX, srsAlpha, field)

    # with open(f"../../output/polys-rsk-{size}.txt", "r") as f:
    #     polys = f.readline()
    #     polys = polys.split("=")[1:]
    #     last = polys[-1]
    #     polys = [x.rsplit(',', 1)[0].strip() for x in polys[:-1]]
    #     polys.append(last[:-1])

    # sXY = read_poly(polys[0])
    # rXY = read_poly(polys[1])
    # tXY = read_poly(polys[-2])

    # sXY = sPoly(u,v,w,n,q)
    
    if save:
        with open(f"../../output/polys{size}.txt", "r") as f:
            polys = f.readline()
            polys = polys.split("=")[1:]
            last = polys[-1]
            polys = [x.rsplit(',', 1)[0].strip() for x in polys[:-1]]
            polys.append(last[:-1])

        sXY = read_poly(polys[0])
        rXY = read_poly(polys[1])
        # r1Raw = read_poly(polys[2])
        # r1Local = read_poly(polys[3])
        tXY = read_poly(polys[-2])

        neg_kXY = kPoly(k, n, q)
        with open(f"sXY_{size}.txt", 'wb') as f:
            pickle.dump(sXY, f)

        with open(f"rXY_{size}.txt", 'wb') as f:
            pickle.dump(rXY, f)

        with open(f"tXY_{size}.txt", 'wb') as f:
            pickle.dump(tXY, f)
                
        with open(f"neg_kXY_{size}.txt", 'wb') as f:
            pickle.dump(neg_kXY, f)

        

    if load:
        with open(f"sXY_{size}.txt", 'rb') as f:
            sXY = pickle.load(f)

        with open(f"rXY_{size}.txt", 'rb') as f:
            rXY = pickle.dump(f)

        with open(f"tXY_{size}.txt", 'rb') as f:
            tXY = pickle.dump(f)

        with open(f"neg_kXY_{size}.txt", 'rb') as f:
            neg_kXY = pickle.dump(f)

    st = time.process_time()

    # rXY = rPoly(aL,aR,aO,n)
    # print(rXY)
    # r_dash_XY = field.add_polys_bivar(rXY[0], sXY[0], rXY[1], sXY[1])

    
    # cX = field.mul_polys_bivar(rX1[0], r_dash_XY[0], rX1[1], r_dash_XY[1])

    # tXY = field.add_polys_bivar(cX[0], neg_kXY[0], cX[1], neg_kXY[1])
    # print(tXY)
    
    
    if save:
        rX1 = field.eval_poly_Y(rXY[0], rXY[1], 1)
        rX1 = field.dimension_change(rX1[0], rX1[1])

        r1Raw = [rX1[0][:len(rX1)-n//2], rX1[1]]
        r1Local = field.sub_polys(rX1[0], r1Raw[0], rX1[1], r1Raw[1])
        print(f"N: {n}")
        r1Local = field.mul_polys(r1Local[0], [1], r1Local[1], n)

        # with open(f"r_dash_XY_{size}.txt", 'wb') as f:
        #     pickle.dump(r_dash_XY, f)
        with open(f"rX1_{size}.txt", 'wb') as f:
            pickle.dump(rX1, f)

        with open(f"r1Raw_{size}.txt", 'wb') as f:
            pickle.dump(r1Raw, f)

        with open(f"r1Local_{size}.txt", 'wb') as f:
            pickle.dump(r1Local, f)
        # with open(f"cX_{size}.txt", 'wb') as f:
        #     pickle.dump(cX, f)

    if load:
        # with open(f"r_dash_XY_{size}.txt", 'wb') as f:
        #     pickle.dump(r_dash_XY, f)
        with open(f"rX1_{size}.txt", 'rb') as f:
            rX1 = pickle.dump(f)

        with open(f"r1Raw_{size}.txt", 'b') as f:
            r1Raw = pickle.dump(f)

        with open(f"r1Local_{size}.txt", 'rb') as f:
            r1Local = pickle.dump(f)
        # with open(f"cX_{size}.txt", 'wb') as f:
        #     pickle.dump(cX, f)
        
    et = time.process_time()
    res = et - st
    print('CPU Execution time-poly construction:', res, 'seconds')


    st = time.process_time()

    g_max = 8 * n

    s1Y = field.eval_poly_X(sXY[0], sXY[1], 1)
    kY = field.mul_by_const(neg_kXY[0][0], -1)
    kY
    # setup: commit Sy, k
    commitSetup = cmScheme.commita([s1Y[0], kY[0]], [s1Y[1], kY[1]], g_max)

    # commit R
    # rX1_changed = field.dimension_change(rX1[0], rX1[1])
    commitR = cmScheme.commita([rX1[0]], [rX1[1]], g_max)[0]
    commitR

    commitRRaw = cmScheme.commita([r1Raw[0]], [r1Raw[1]], g_max)[0]
    commitRLocal = cmScheme.commita([r1Local[0]], [r1Local[1]], g_max)[0]


    # y
    y = 3
    #commit T
    tXy = field.eval_poly_Y(tXY[0], tXY[1], y)
    tXy = field.dimension_change(tXy[0], tXy[1])
    commitT = cmScheme.commita([tXy[0]], [tXy[1]], g_max)[0]
    commitT

    #commit Sx
    sXy = field.eval_poly_Y(sXY[0], sXY[1], y)
    sXy = field.dimension_change(sXy[0], sXy[1])
    commitSx = cmScheme.commita([sXy[0]], [sXy[1]], g_max)[0]
    commitSx

    # z
    z = 2

    # opens

    # list_of_c = [commitR, commitRRaw, commitRLocal, commitT] 
    # list_of_z_for_p = [[z, z*y], [z], [z], [z]]
    # list_of_p = [rX1[0], r1Raw[0], r1Local[0], tXy[0]]
    # list_of_init_order = [rX1[1], r1Raw[1], r1Local[1], tXy[1]]

    # opens = cmScheme.openC(list_of_c, list_of_z_for_p, list_of_p, list_of_init_order, g_max)

    # fz = opens[4]

    # # opens outsourced

    # list_of_c_o = [commitSetup[0], commitSx, commitSetup[1]] 
    # list_of_z_for_p_o = [[y], [z, 1], [y]]
    # list_of_p_o = [s1Y[0], sXy[0], kY[0]]
    # list_of_init_order_o = [s1Y[1], sXy[1], kY[1]]

    # openOutsource = cmScheme.openC(list_of_c_o, list_of_z_for_p_o, list_of_p_o, list_of_init_order_o, g_max)

    # fz_o = openOutsource[4]

    #open all together

    list_of_c = [commitR, commitRRaw, commitRLocal, commitT, commitSetup[0], commitSx, commitSetup[1]] 
    list_of_z_for_p = [[z, z*y], [z], [z], [z], [y], [z, 1], [y]]
    list_of_p = [rX1[0], r1Raw[0], r1Local[0], tXy[0], s1Y[0], sXy[0], kY[0]]
    list_of_init_order = [rX1[1], r1Raw[1], r1Local[1], tXy[1], s1Y[1], sXy[1], kY[1]]

    opens = cmScheme.openC(list_of_c, list_of_z_for_p, list_of_p, list_of_init_order, g_max)

    fz = opens[4]

    et = time.process_time()
    
    res = et - st
    print('CPU Execution time - proof generation:', res, 'seconds')

    # verify
    t = fz[3][0]
    dj = fz[2][0]
    r_tilde = fz[1][0]
    r1 = fz[0][0]
    r2 = fz[0][1]
    s = fz[5][0]
    k_fz = fz[6][0]
    s1 = fz[5][1]
    s2 = fz[4][0]
    # s = fz_o[1][0]
    # k_fz = fz_o[2][0]
    # s1 = fz_o[1][1]
    # s2 = fz_o[0][0]

    print(f"\
            dj: {dj} \n \
            z: {z} \n \
            y: {y} \n \
            pi_1: {opens[0]} \n \
            pi_2: {opens[1]} \n \
            r1: {r1} \n \
            r1_tilde: {r_tilde} \n \
            t: {t} \n \
            k: {k_fz} \n \
            s_tilde: {s} \n \
            r2: {r2} \n \
            s1_tilde: {s1} \n \
            s2_tilde: {s2} \n \
            D: {commitRLocal} \n \
            R_tilde: {commitRRaw} \n \
            R: {commitR} \n \
            T: {commitT} \n \
            K: {list_of_c[6]} \n \
            S_x: {list_of_c[5]} \n \
            S_y: {list_of_c[4]} \n \
        ")
    
    # K: {list_of_c_o[2]} \n \
    # S_x: {list_of_c_o[1]} \n \
    # S_y: {list_of_c_o[0]} \n \

    return cmScheme.verify(list_of_c, *opens) and t == field.sub(field.mul(r1, field.add(r2, s)), k_fz) and s1 == s2
#cmScheme.verify(list_of_c_o, *openOutsource) and
size = 4
aL, aR, aO, k, u, v, w, n, q = setup_data(size)
print(sonic_experiment(size, aL, aR, aO, k, u, v, w, n, q, True))

