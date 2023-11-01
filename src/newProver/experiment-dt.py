# scipy, py_ecc, galois, sklearn


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

    
def decision_tree(x_p, s_j, t_j, l_i, n, k, k_prime):
    h = 1
        
    v_j_p = np.zeros_like(x_p)
    v_j_p[s_j[0] - 1] = 1 # starts from 0
    
    x_s_j_p = np.array(x_p) * np.array(v_j_p)
    x_s_j = np.array([0])
    x_s_j[0] = np.sum(x_s_j_p)
    
    v_j_p_g = np.empty(0, dtype=int)
    v_j_p_g_prime = np.empty(0, dtype=int)
    
    omega_j_p = np.zeros_like(x_p)
    for p in range(n):
        omega_j_p[p] = s_j[0] - p - 1 #starts from 0
        if omega_j_p[p] > 0: #bit decomposition of v_j_p_g, assign 0s to v_j_p_g_b_prime
            decomposition = np.array(fixed_length_decomposition((s_j[0] - p), k_prime))
            decomposition = np.squeeze(decomposition)  # Convert to 1D array if needed
            v_j_p_g = np.concatenate((v_j_p_g, decomposition)).astype(int)
            v_j_p_g_prime = np.concatenate((v_j_p_g_prime, np.zeros(k_prime))).astype(int)
        elif omega_j_p[p] < 0:
            decomposition_prime = np.array(fixed_length_decomposition((s_j[0] - p - 2), k_prime))
            decomposition_prime = np.squeeze(decomposition_prime)  # Convert to 1D array if needed
            v_j_p_g_prime = np.concatenate((v_j_p_g_prime, decomposition_prime)).astype(int)
            v_j_p_g = np.concatenate((v_j_p_g, np.zeros(k_prime))).astype(int)
        else:
            v_j_p_g_prime = np.concatenate((v_j_p_g_prime, np.zeros(k_prime))).astype(int)
            v_j_p_g = np.concatenate((v_j_p_g, np.zeros(k_prime))).astype(int)
    
    v_j_p_g_b = np.zeros_like(v_j_p_g)
    v_j_p_g_b_prime = np.zeros_like(v_j_p_g_prime)    
    sigma_v_j_p_g = np.zeros_like(v_j_p)
    sigma_v_j_p_g_prime = np.zeros_like(v_j_p)
    for p in range(n):
        for g in range(k_prime):
            v_j_p_g_b[p*k_prime + g] = v_j_p_g[p*k_prime + g] - 2**(g)
            v_j_p_g_b_prime[p*k_prime + g] = v_j_p_g_prime[p*k_prime + g] + 2**(g)
            sigma_v_j_p_g[p] = sigma_v_j_p_g[p] + v_j_p_g[p*k_prime + g]
            sigma_v_j_p_g_prime[p] = sigma_v_j_p_g_prime[p] + v_j_p_g_prime[p*k_prime + g]
    
    theta_j_p = np.array(sigma_v_j_p_g - omega_j_p - 1) * np.array(sigma_v_j_p_g_prime - omega_j_p + 1)
    
    w_j = np.zeros_like(x_s_j)
    w_j = x_s_j - t_j
    c_j = np.array([0])
    if w_j[0] > 0:
        b_j_m = np.zeros(k).astype(int)
        decomposition = fixed_length_decomposition(w_j[0] - 1,k)
        decomposition = np.squeeze(decomposition)
        b_j_m_prime = decomposition
        c_j[0] = 1
    else: 
        b_j_m_prime = np.zeros(k).astype(int)
        decomposition2 = fixed_length_decomposition(w_j[0],k)
        decomposition2 = np.squeeze(decomposition2)
        b_j_m = decomposition2

    
    b_j_m_b = np.zeros_like(b_j_m)
    b_j_m_prime_b = np.zeros_like(b_j_m_prime)
    for m in range(k):
        b_j_m_b[m] = b_j_m[m] + 2**m
        b_j_m_prime_b[m] = b_j_m_prime[m] - 2**m
        
    sigma_b_j_m = np.sum(b_j_m)
    sigma_b_j_m_prime = np.sum(b_j_m_prime)
    
    beta_i_a = np.empty(0, dtype=int)
    for i in range(2**h,2**(h+1)): #stops before 2**(h+1) - 1
        beta_i_a = np.concatenate((beta_i_a, fixed_length_binary_decomposition(i,h+1))).astype(int)

    
    two_power_a_minus_1 = np.zeros_like(beta_i_a)
    for i in range(2**h,2**(h+1)):
        for alpha in range(1,h+2): #1 to h+1
            two_power_a_minus_1[(i - 2**h)*(h + 1) + alpha - 1] = 2**(alpha - 1) 
    
    u_i_a = beta_i_a * two_power_a_minus_1
    
    beta_i_j = np.zeros(2**h).astype(int)
    for i in range(2**h,2**(h+1)):
        beta_i_j[i-2**h] = beta_i_a[(i - 2**h)*(h + 1)]
    
    c_i_j = np.concatenate((c_j,c_j)).astype(int)
    
    epsilon_i = c_i_j * beta_i_j
    epsilon_i_prime = (1 - c_i_j) * (1 - beta_i_j)
    
    z_i = epsilon_i + epsilon_i_prime
    
    e_i = l_i * z_i
    #print(f"beta_i_a: {np.array(beta_i_a)}")

    a_upper = np.concatenate( [x_p,
                                v_j_p,
                                v_j_p_g,
                                v_j_p_g_prime,
                                omega_j_p,
                                (sigma_v_j_p_g - omega_j_p - 1),
                                theta_j_p,
                                b_j_m,
                                c_j,
                                (sigma_b_j_m - w_j),
                                b_j_m_prime,
                                (sigma_b_j_m_prime - w_j + 1),
                                beta_i_a,
                                beta_i_a,
                                c_i_j,
                                (1 - c_i_j),
                                l_i
                               ]
                            ).astype(int)
    b_upper = np.concatenate( [v_j_p,
                               (1 - v_j_p),
                                v_j_p_g_b,
                                v_j_p_g_b_prime,
                                v_j_p,
                                (sigma_v_j_p_g_prime - omega_j_p + 1),
                                (1 - v_j_p),
                                b_j_m_b,
                                (1 - c_j),
                                (1 - c_j),
                                b_j_m_prime_b,
                                c_j,
                                (1 - beta_i_a),
                                two_power_a_minus_1,
                                beta_i_j,
                                (1 - beta_i_j),
                                z_i
                               ]
                            ).astype(int)
    c_upper = np.concatenate( [x_s_j_p,
                                np.zeros_like(v_j_p),
                                np.zeros_like(v_j_p_g),
                                np.zeros_like(v_j_p_g_prime),
                                np.zeros_like(omega_j_p),
                                theta_j_p,
                                np.zeros_like(theta_j_p),
                                np.zeros_like(b_j_m),
                                np.zeros_like(c_j),
                                np.zeros_like(c_j),
                                np.zeros_like(b_j_m_prime),
                                np.zeros_like(c_j),
                                np.zeros_like(beta_i_a),
                                u_i_a,
                                epsilon_i,
                                epsilon_i_prime,
                                e_i
                                ]
                            ).astype(int)
    
    # check shape of a_upper
    shape_of_a_upper = 5*n + 2*n*k_prime + 2*k + 17
    assert(a_upper.shape[0] == shape_of_a_upper)
    assert(b_upper.shape[0] == shape_of_a_upper)
    assert(c_upper.shape[0] == shape_of_a_upper)
    assert(((a_upper * b_upper).astype(int)  == c_upper).all())
    
    
    index = np.zeros(2**h).astype(int)
    for i_prime in range(2**h,2**(h+1)):
        for a in range(1,h+2):
            index[i_prime - 2**h] = index[i_prime - 2**h] + u_i_a[(i_prime - 2**h)*(h + 1) + a - 1]
    
    y = np.sum(e_i)
    # Ineternal input
    a_middle = np.concatenate([x_s_j, 
                                index, 
                                z_i, 
                                [y],
                                omega_j_p,
                                w_j
                                ]
                                ).astype(int)
    b_middle = np.zeros_like(a_middle).astype(int)
    c_middle = np.zeros_like(a_middle).astype(int)
    assert(a_middle.shape[0] == n+7)
    assert(((a_middle * b_middle).astype(int)  == c_middle).all())

    # Ext input
    a_lower = np.concatenate([x_p, s_j, t_j, l_i]).astype(int)
    b_lower = np.zeros_like(a_lower).astype(int)
    c_lower = np.zeros_like(a_lower).astype(int)
    assert(a_lower.shape[0] == n+4)
     # check multiplication satisfied
    assert(((a_lower * b_lower).astype(int)  == c_lower).all())
        

    a = np.concatenate([a_upper, a_middle, a_lower])
    b = np.concatenate([b_upper, b_middle, b_lower])
    c = np.concatenate([c_upper, c_middle, c_lower])

    assert(((a * b).astype(int)  == c).all())
    assert(a.shape[0] == 7*n + 2*n*k_prime + 2*k + 28)
    
    # linear constraints. k_q is a list of scalar, u_q, v_q, w_q are list of numpy arrays
    k_q = []
    u_q = []
    v_q = []
    w_q = []
    
    #\sum_{p = 1}^{n} (x_{[s_{[j,p]}]}))_{j = 1}^{2^h - 1} = (x_{[s_{[j]}]})_{j = 1}^{2^h - 1}
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 17] = 1 #18 - 1
    w[0:n] = -1
    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    
    #\sum_{a = 1}^{h + 1} (u_{[i,a]}))_{i = 2^h}^{2^{h + 1} - 1} = (i)_{i = 2^h}^{2^{h + 1} - 1
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 18] = 1 
    w[5*n + 2*n*k_prime + 2*k + 7:5*n + 2*n*k_prime + 2*k + 9] = -1
    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 19] = 1 
    w[5*n + 2*n*k_prime + 2*k + 9:5*n + 2*n*k_prime + 2*k + 11] = -1
    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #\zeta{[i]} + \zeta'_{[i]})_{i = 2^h}^{2^{h + 1} - 1
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 20] = 1 
    w[5*n + 2*n*k_prime + 2*k + 11] = -1
    w[5*n + 2*n*k_prime + 2*k + 13] = -1
    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 21] = 1 
    w[5*n + 2*n*k_prime + 2*k + 12] = -1
    w[5*n + 2*n*k_prime + 2*k + 14] = -1
    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #\sum_{i = 2^h}^{2^{h + 1} - 1} \epsilon_{[i]} = y
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 22] = 1 
    w[5*n + 2*n*k_prime + 2*k + 15:5*n + 2*n*k_prime + 2*k + 17] = -1

    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #(s_{[j]} - p = \omega_{[j,p]})_{j = 1, p = 1}^{2^h - 1, n}
    for p in range(n):
        k_2 = p + 1 #p starts from 1 to n
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
    
        u[5*n + 2*n*k_prime + 2*k + 23 + p] = -1 
        u[7*n + 2*n*k_prime + 2*k + 24] = 1

    
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #x_{[s_{[j]}]} - t_{[j]} = w_{[j]})_{j = 1}^{2^h - 1}
    k_2 = 0 
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)

    u[5*n + 2*n*k_prime + 2*k + 17] = 1 
    u[7*n + 2*n*k_prime + 2*k + 25] = -1
    u[6*n + 2*n*k_prime + 2*k + 23] = -1

    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #(x_{[p]})_{p = 1}^{n} \) match \( (x_{[p]})_{j = 1, p = 1}^{2^h - 1, n}
    for p in range(n): 
        k_2 = 0 
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)

        u[p] = 1 
        u[6*n + 2*n*k_prime + 2*k + 24 + p] = -1

        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
        
    #(l_{[i]})_{i = 2^h}^{2^{h +1} - 1} \) match \( (l_{[i]})_{i = 2^h}^{2^{h +1} - 1}
    for i in range(2**h,2**(h+1)):
        k_2 = 0 
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)

        u[5*n + 2*n*k_prime + 2*k + 15 + i - 2**h] = 1
        u[7*n + 2*n*k_prime + 2*k + 26 + i - 2**h] = -1

        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #(v_{[j,p]})_{j = 1, p = 1}^{2^h -1 , n}
    for p in range(n):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)

        u[n + p] = 1
        v[p] = -1

        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
        
    #(v_{[j,p]})_{j = 1, p = 1}^{2^h -1 , n}
    for p in range(n):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)

        u[n + p] = 1
        v[2*n + 2*n*k_prime + p] = -1

        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #(1 - v_{[j,p]})_{j = 1, p = 1}^{2^h -1 , n}
    for p in range(n):
        k_2 = 1
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)

        u[n + p] = 1
        v[n + p] = 1

        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #(1 - v_{[j,p]})_{j = 1, p = 1}^{2^h -1 , n}
    for p in range(n):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)

        v[n + p] = 1
        v[4*n + 2*n*k_prime + p] = -1

        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #(v_{[j,p,g]} - 2^{g-1})_{j = 2^h - 1, p = 1, g = 1}^{1, n, k'}
    for p in range(n):
        for g in range(k_prime):
            k_2 = 2**g
            u = np.zeros_like(a)
            v = np.zeros_like(b)
            w = np.zeros_like(c)

            u[2*n + p*k_prime + g] = 1
            v[2*n + p*k_prime + g] = -1

            k_q.append(k_2)
            u_q.append(u)
            v_q.append(v)
            w_q.append(w)
    
    #\sum_{g=1}^{k'} v_{[j,p,g]} - \omega_{[j,p]} - 1)_{j = 1, p = 1}^{2^h - 1, n}
    for p in range(n):
        for g in range(k_prime):
            k_2 = -2**g
            u = np.zeros_like(a)
            v = np.zeros_like(b)
            w = np.zeros_like(c)

            u[2*n + n*k_prime + p*k_prime + g] = 1
            v[2*n + n*k_prime + p*k_prime + g] = -1

            k_q.append(k_2)
            u_q.append(u)
            v_q.append(v)
            w_q.append(w)
    
    #\sum_{g=1}^{k'} v_{[j,p,g]} - \omega_{[j,p]} - 1)_{j = 1, p = 1}^{2^h - 1, n}
    for p in range(n):
        k_2 = -1
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[2*n + 2*n*k_prime + p] = 1
        u[3*n + 2*n*k_prime + p] = 1
        
        for g in range(k_prime):
            u[2*n + p*k_prime + g] = -1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #\sum_{g=1}^{k'} v'_{[j,p,g]} - \omega_{[j,p]} + 1)_{j = 1, p = 1}^{2^h - 1, n}
    for p in range(n):
        k_2 = 1
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[2*n + 2*n*k_prime + p] = 1
        v[3*n + 2*n*k_prime + p] = 1
        
        for g in range(k_prime):
            u[2*n +n*k_prime + p*k_prime + g] = -1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
        
    #\theta_{[j,p]})_{j = 1, p = 1}^{2^h - 1, n}
    for p in range(n):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[4*n + 2*n*k_prime + p] = 1
        w[3*n + 2*n*k_prime + p] = -1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)

    #b_{[j,m]} + 2^{m - 1})_{j = 1, m = 1}^{2^h - 1, k}
    for m in range(k):
        k_2 = 2**m
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[5*n + 2*n*k_prime + m] = -1
        v[5*n + 2*n*k_prime + m] = 1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #c_{[j]})_{j = 1}^{2^h - 1}
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + k] = 1
    v[5*n + 2*n*k_prime + 2*k + 2] = -1
        
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #c_{[j]})_{j = 1}^{2^h - 1}\) match \((c_{[j]})_{i=2^h , j = 1}^{2^{h+1} - 1, 2^h - 1}
    for i in range(2**h,2**(h+1)):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[5*n + 2*n*k_prime + k] = 1
        u[5*n + 2*n*k_prime + 2*k + 11 + i - 2**h] = -1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #(1 - c_{[j]})_{j = 1}^{2^h - 1}
    k_2 = 1
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + k] = 1
    v[5*n + 2*n*k_prime + k] = 1
        
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #1 - c_{[j]})_{j = 1}^{2^h - 1}
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    v[5*n + 2*n*k_prime + k] = 1
    v[5*n + 2*n*k_prime + k + 1] = -1
        
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #(1 - c_{[j]})_{j = 1}^{2^h - 1}\) match \((1 - c_{[j]})_{i = 2^h, j = 1}^{2^{h+1} - 1, 2^h - 1}
    for i in range(2**h,2**(h+1)):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[5*n + 2*n*k_prime + 2*k + 13 + i - 2**h] = 1
        v[5*n + 2*n*k_prime + k] = -1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    #\sum_{m=1}^{k} b_{[j,m]} - w_{[j]})_{j = 1}^{2^h - 1}
    k_2 = 0
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + k + 1] = 1
    u[6*n + 2*n*k_prime + 2*k + 22] = 1
    for m in range(k):
        u[5*n + 2*n*k_prime + m] = -1
        
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #(b'_{[j,m]} - 2^{m - 1})_{j = 1, m = 1}^{2^h - 1, k}
    for m in range(k):
        k_2 = 2**m
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[5*n + 2*n*k_prime + k + 2 + m] = 1
        v[5*n + 2*n*k_prime + k + 2 + m] = -1
            
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
   
    
    #\sum_{m = 1}^{k}b'_{[j,m]} - w_{[j]} + 1)_{j = 1}^{2^h - 1}
    k_2 = 1
    u = np.zeros_like(a)
    v = np.zeros_like(b)
    w = np.zeros_like(c)
    
    u[5*n + 2*n*k_prime + 2*k + 2] = 1
    u[6*n + 2*n*k_prime + 2*k + 23] = 1
    
    for m in range(k):
        u[5*n + 2*n*k_prime + k + 2 + m] = -1
    
    k_q.append(k_2)
    u_q.append(u)
    v_q.append(v)
    w_q.append(w)
    
    #\beta_{[i,a]})_{i = 2^h, a = 1}^{2^{h + 1} - 1, h + 1}
    for i in range(2**h,2**(h+1)):
        for alpha in range(1,h+2):
            k_2 = 0
            u = np.zeros_like(a)
            v = np.zeros_like(b)
            w = np.zeros_like(c)
            
            u[5*n + 2*n*k_prime + 2*k + 3 + (i - 2**h)*(h + 1) + alpha - 1] = 1
            u[5*n + 2*n*k_prime + 2*k + 7 + (i - 2**h)*(h + 1) + alpha - 1] = -1
            
            k_q.append(k_2)
            u_q.append(u)
            v_q.append(v)
            w_q.append(w)
            
            
    #\beta_{[i,j]})_{i = 2^h, j = 1}^{2^{h + 1} - 1, 2^h - 1}\) match \((\beta_{[i,a]})_{i = 2^h, a = 1}^{2^{h + 1} - 1, h + 1}
    for i in range(2**h,2**(h+1)):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        u[5*n + 2*n*k_prime + 2*k + 3 + (i - 2**h)] = 1
        v[5*n + 2*n*k_prime + 2*k + 11 + (i - 2**h)] = -1
        
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)
    
    
    #(1 - \beta_{[i,a]})_{i = 2^h, a = 1}^{2^{h + 1} - 1, h + 1}
    for i in range(2**h,2**(h+1)):
        for alpha in range(1,h+2):
            k_2 = 1
            u = np.zeros_like(a)
            v = np.zeros_like(b)
            w = np.zeros_like(c)
            
            u[5*n + 2*n*k_prime + 2*k + 3 + (i - 2**h)*(h + 1) + alpha - 1] = 1
            v[5*n + 2*n*k_prime + 2*k + 3 + (i - 2**h)*(h + 1) + alpha - 1]  = 1
            
            k_q.append(k_2)
            u_q.append(u)
            v_q.append(v)
            w_q.append(w)
    
    #1 - \beta_{[i,j]})_{i = 2^h, j = 1}^{2^{h + 1} - 1, 2^h - 1}\) match \((1 - \beta_{[i,a]})_{i = 2^h, a = 1}^{2^{h + 1} - 1, h + 1}
    for i in range(2**h,2**(h+1)):
        k_2 = 0
        u = np.zeros_like(a)
        v = np.zeros_like(b)
        w = np.zeros_like(c)
        
        v[5*n + 2*n*k_prime + 2*k + 3 + (i - 2**h)] = 1
        v[5*n + 2*n*k_prime + 2*k + 13 + (i - 2**h)] = -1
        
        k_q.append(k_2)
        u_q.append(u)
        v_q.append(v)
        w_q.append(w)

    
    ###
    for i in range(len(k_q)):
        assert(k_q[i]==u_q[i]@a+v_q[i]@b+w_q[i]@c)
    
    
    # print(f"a: {np.concatenate([a_upper, a_middle, a_lower])}")
    # print(f"b: {np.concatenate([b_upper, a_middle, b_lower])}")
    # print(f"c: {np.concatenate([c_upper, a_middle, c_lower])}")
    
    print(f"abc length: {np.concatenate([a_upper, a_middle, a_lower]).shape}")
    # print(f"k: {np.array(k_q)}")  
    # print(f"u: {np.array(u_q)}")
    # print(f"v: {np.array(v_q)}")
    # print(f"w: {np.array(w_q)}")
    print(f"k length: {np.array(k_q).shape}")  
    
    np.savetxt("../../input/aL.txt", a.astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/aO.txt", c.astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/aR.txt", b.astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/cs.txt", np.array(k_q).astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/wL.txt", np.array(u_q).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/wO.txt", np.array(w_q).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")
    np.savetxt("../../input/wR.txt", np.array(v_q).reshape((-1)).astype(int), delimiter=' ', newline=" ", fmt="%0d")

    return (a, 
            b, 
            c, 
            np.array(k_q), np.array(u_q), np.array(v_q), np.array(w_q))
            
def fixed_length_decomposition(n, k):
    if abs(n) >= 2 ** k:
        raise ValueError("Absolute value of n is too large to be represented with k bits.")

    result = [0] * k  
    abs_n = abs(n)
    for i in range(k - 1, -1, -1):
        coefficient = abs_n // (2 ** i)
        abs_n -= coefficient * (2 ** i)
        result[i] = coefficient * (2 ** i) if n >= 0 else -coefficient * (2 ** i)
    return result
    
def fixed_length_binary_decomposition(n, k):
    if abs(n) >= 2 ** k:
        raise ValueError("Absolute value of n is too large to be represented with k bits.")

    result = [0] * k  
    abs_n = abs(n)
    for i in range(k - 1, -1, -1):
        coefficient = abs_n // (2 ** i)
        abs_n -= coefficient * (2 ** i)
        result[i] = coefficient if n >= 0 else -coefficient
    return result


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
    

class KZGBatchCommitment():
    def __init__(self, n, srsX, srsAlpha, field):
        self.G1 = bn128_curve.G1
        self.G2 = bn128_curve.G2
        self.srsD = n * 6
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
                # index = init_order + i + self.srsD - g_max
                index = init_order + i
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
        rand_z = 1
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
            
        non_a_part_w_a = bn128_curve.multiply(bn128_curve.add(non_a_part, bn128_curve.multiply(wDash, rand_z)), srsAlpha)
        left1right = self.hPositiveX[self.srsD - g_max].astype(FQ2)
        # alpha_part = bn128_curve.multiply(bn128_curve.G1, 1)
        left2left = bn128_curve.add(cm_poly, non_a_part_w_a)
        # left2right = self.hPositiveAlphaX[self.srsD - g_max]
        rightleft = wDash
        rightright = self.hPositiveAlphaX[1 + self.srsD - g_max].astype(FQ2)
        
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


def sonic_experiment(size, aL, aR, aO, k, u, v, w, n, q, save=False, load=True):
    
    cmScheme = KZGBatchCommitment(n, srsX, srsAlpha, field)
    
    with open(f"../../output/polys_dt.txt", "r") as f:
        polys = f.readline()
        polys = polys.split("=")[1:]
        last = polys[-1]
        polys = [x.rsplit(',', 1)[0].strip() for x in polys[:-1]]
        polys.append(last[:-1])

    sXY = read_poly(polys[0])
    rXY = read_poly(polys[1])
    tXY = read_poly(polys[-2])

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
    neg_kXY = kPoly(k, n, q)
    if save:
        with open(f"sXY_{size}.txt", 'wb') as f:
            pickle.dump(sXY, f)
        # with open(f"sXY_{size}.txt", 'rb') as f:
        #     sXY = pickle.load(f)
                
        with open(f"neg_kXY_{size}.txt", 'wb') as f:
            pickle.dump(neg_kXY, f)

    st = time.process_time()

    # rXY = rPoly(aL,aR,aO,n)
    # print(rXY)
    # r_dash_XY = field.add_polys_bivar(rXY[0], sXY[0], rXY[1], sXY[1])

    rX1 = field.eval_poly_Y(rXY[0], rXY[1], 1)
    # cX = field.mul_polys_bivar(rX1[0], r_dash_XY[0], rX1[1], r_dash_XY[1])

    # tXY = field.add_polys_bivar(cX[0], neg_kXY[0], cX[1], neg_kXY[1])
    # print(tXY)
    et = time.process_time()
    res = et - st
    print('CPU Execution time-poly construction:', res, 'seconds')
    
    if save:
        with open(f"rXY_{size}.txt", 'wb') as f:
            pickle.dump(rXY, f)
        # with open(f"r_dash_XY_{size}.txt", 'wb') as f:
        #     pickle.dump(r_dash_XY, f)
        with open(f"rX1_{size}.txt", 'wb') as f:
            pickle.dump(rX1, f)
        # with open(f"cX_{size}.txt", 'wb') as f:
        #     pickle.dump(cX, f)
        with open(f"tXY_{size}.txt", 'wb') as f:
            pickle.dump(tXY, f)
        
    st = time.process_time()

    g_max = 6 * n

    s1Y = field.eval_poly_X(sXY[0], sXY[1], 1)
    kY = field.mul_by_const(neg_kXY[0][0], -1)
    kY
    # setup: commit Sy, k
    commitSetup = cmScheme.commita([s1Y[0], kY[0]], [s1Y[1], kY[1]], g_max)

    # commit R
    rX1_changed = field.dimension_change(rX1[0], rX1[1])
    commitR = cmScheme.commita([rX1_changed[0]], [rX1_changed[1]], g_max)[0]
    commitR

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

    list_of_c = [commitR, commitT] 
    list_of_z_for_p = [[z, z*y], [z]]
    list_of_p = [rX1_changed[0], tXy[0]]
    list_of_init_order = [rX1_changed[1], tXy[1]]

    opens = cmScheme.openC(list_of_c, list_of_z_for_p, list_of_p, list_of_init_order, g_max)

    fz = opens[4]

    # opens outsourced

    list_of_c_o = [commitSetup[0], commitSx, commitSetup[1]] 
    list_of_z_for_p_o = [[y], [z, 1], [y]]
    list_of_p_o = [s1Y[0], sXy[0], kY[0]]
    list_of_init_order_o = [s1Y[1], sXy[1], kY[1]]

    openOutsource = cmScheme.openC(list_of_c_o, list_of_z_for_p_o, list_of_p_o, list_of_init_order_o, g_max)

    fz_o = openOutsource[4]

    et = time.process_time()
    
    res = et - st
    print('CPU Execution time - proof generation:', res, 'seconds')

    # verify
    t = fz[1][0]
    r1 = fz[0][0]
    r2 = fz[0][1]
    s = fz_o[1][0]
    k = fz_o[2][0]
    s1 = fz_o[1][1]
    s2 = fz_o[0][0]

    return cmScheme.verify(list_of_c, *opens) and cmScheme.verify(list_of_c_o, *openOutsource) and t == field.sub(field.mul(r1, field.add(r2, s)), k) and s1 == s2


# Change this function:
def setup_data():

    aL, aR, aO, k, u, v, w = decision_tree(np.array([1, 1, 0, 1, 1, 2000, 0, 150, 360, 1, 5]), np.array([10]), np.array([2]), np.array([0, 1]), 11, 4, 4)

    assignment = [aL, aR, aO]
    circuit = [u,v,w,k]

    assert (aL @ u.T + aR @ v.T + aO @ w.T == k).all()

    n = aL.shape[0]
    q = k.shape[0]
    
    print("constraints generation completed")

    return aL, aR, aO, k, u, v, w, n, q


aL, aR, aO, k, u, v, w, n, q = setup_data()

sonic_experiment(4, aL, aR, aO, k, u, v, w, n, q, True)