# ECDSA signature following tutorial from https://levelup.gitconnected.com/ecdsa-how-to-programmatically-sign-a-transaction-95eec854bca7
from Crypto.Hash import keccak
from fastecdsa.curve import Curve
from fastecdsa.point import Point
from fastecdsa.curve import secp256k1


N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Ac = 0
Bc = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
def egcd(a, b):
     if a == 0:
          return (b, 0, 1)
     else:
          g, y, x = egcd(b % a, a)
          return (g, x - (b // a) * y, y)
          
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exemption('modular inverse does not exist')
    else:
        return x % m



pair = None
with open("output/proof.txt") as f:
    proof = f.read()
    substr = proof.split("prRRaw = A (P ")[1].split("), prT = A (P ")[0]
    pair = substr.split(") (P ")
    print(pair)
    # pair = proof[193: 269] + proof[274: 351]
# print(pair)
# k = '0x' + keccak.new(data=(pair[0]+pair[1]).encode(), digest_bits=256).hexdigest()
# print(k)
# GenPoint = Point(Gx, Gy, curve=secp256k1)
# print(GenPoint)
# SK=999888777666555444333222111
# VK = SK * GenPoint
# print(VK)
# randomNum = 936462993105
# XY1 = randomNum * GenPoint
# r = XY1.x % N
# s = ((int(k, 16) + r * SK) * modinv(randomNum, N)) % N
# w = modinv(s, N)

# print(f"signature on commitment of rRaw:= \n    r: {r} \n    s^-1: {w}")

# u1 = int(k, 16) * w % N
# u2 = r * w % N
# XY2 = u1 * GenPoint + u2 * VK
# print(u1)
# print(u2)
# print(u1 * GenPoint)

# print(f"signature verification success: {r == XY2.x % N}")
