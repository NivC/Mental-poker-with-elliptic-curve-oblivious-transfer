import random
import sys
import hashlib

hash_func = lambda msg: hashlib.sha256(msg).digest()

class ECPoint:
    a = -3
    b = 41058363725152142129326129780047268409114441015993725554835256314039467401291
    p = 2**256-2**224+2**192+2**96-1
    Gx = 48439561293906451759052585252797914202762949526041747995844080717082404635286
    Gy = 36134250956749795798585127919587881956611106672985015071877198253568414405109
    N = 115792089210356248762697446949407573529996955224135760342422259061068512044369

    def from_bytes(bytes):
        x = int.from_bytes(bytes[32:], byteorder="big", signed=False)
        y = int.from_bytes(bytes[:32], byteorder="big", signed=False)
        assert((y**2)%ECPoint.p == (x**3 + ECPoint.a*x + ECPoint.b)%ECPoint.p or (x == 0 and y == 0))
        return ECPoint(x,y)

    def to_bytes(self):
        return (self.x + (self.y<<256)).to_bytes(64, byteorder="big", signed=False)

    def __init__(self, x = Gx, y = Gy):
        self.x, self.y = x, y

    def __add__(self, other):
        if self.x == 0: return other
        if other.x == 0: return self
        if self.y == -other.y%ECPoint.p: return ECPoint(0,0)

        if self.y == other.y:
            l = (3*self.x**2 + ECPoint.a)*pow(2*self.y,ECPoint.p-2,ECPoint.p)
        else:
            l = (other.y - self.y)*pow(other.x - self.x, ECPoint.p-2, ECPoint.p)
        x = l**2 - self.x - other.x
        y = l*(self.x - x) - self.y
        return ECPoint(x%ECPoint.p,y%ECPoint.p)

    def __neg__(self):
        return ECPoint(self.x, -self.y%ECPoint.p)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, n):
        if n == 0: return ECPoint(0,0)

        half_prod = self*(n//2)
        if n%2 == 0:
            return half_prod + half_prod
        else:
            return half_prod + half_prod + self         

ecp1 = ECPoint()
ecp2 = ecp1*(ECPoint.N)
print(ecp2.x,"\n",ECPoint.Gx,"\n")

rand = random.SystemRandom()
y = rand.randint(0,ECPoint.p-1)
B = ECPoint()
S = B*y
T = S*y

c = 3
x = rand.randint(0,ECPoint.p-1)
R = S*c + B*x

Ry = R*y
Tj = ECPoint(0,0)
for j in range(50):
    print(hash_func((Ry-Tj).to_bytes())[:32])
    Tj = Tj + T
print()
print(hash_func((S*x).to_bytes())[:32])
sys.exit(1)
a = -3 # elliptic parameter
b = 41058363725152142129326129780047268409114441015993725554835256314039467401291 # elliptic parameter
p = 2**256-2**224+2**192+2**96-1 # prime
#co-factor is h=1
Gx = 48439561293906451759052585252797914202762949526041747995844080717082404635286 #initial point x
Gy = 36134250956749795798585127919587881956611106672985015071877198253568414405109 #initial point y
N = 115792089210356248762697446949407573529996955224135760342422259061068512044369 #group order

def check_if_in_curve(p,a,b,x,y):
    #(0,0) is infinity point
    if (x==0 and y==0):
        return True
    if ((y**2 % p) == ((x**3+a*x+b) % p)):
        return True
    else:
        return False

def curve_inversion(x,y): #inversion of a point in curve
    return x,-y

def field_inversion(p,x): #inversion of a point in prime field
    if x==0:
        return 0
    else:
        return pow(x, p-2, p)
    
# https://pdfs.semanticscholar.org/ac3c/28ebf9a40319202b3c4f64cc81cdaf193da5.pdf Page 11
def curve_multiplication(p,x1,y1,x2,y2):
    if (x1==0 and y1==0):
        return x2,y2
    if (x2==0 and y2==0):
        return x1,y1
    if (x1==x2):
        if (y1==y2):
            lamda=(3*x1**2-3)*field_inversion(p,2*y1 % p) % p
        else:
            lamda=0
    else:
        lamda=(y2-y1)*field_inversion(p,x2-x1) % p
        
    if lamda==0:
        x3=0
        y3=0
    else:
        x3=(lamda**2-x1-x2) % p
        y3=(lamda*(x1-x3)-y1) % p
    return x3,y3

def curve_division(p,x1,y1,x2,y2): #Division of x1,y1 by x2,y2
    [x2,y2]=curve_inversion(x2,y2)
    [x3,y3]=curve_multiplication(p,x1,y1,x2,y2)
    return x3 % p, y3 % p
        
def slow_curve_power(p,x1,y1,n):  #For testing only. Do Not Use!
    if (n==0):
        return 0,0
    xn=x1 % p
    yn=y1 % p
    for x in range(0, n-1):
        [xn,yn]=curve_multiplication(p,x1,y1,xn,yn)
    return xn,yn    

def curve_power(p,x1,y1,n): #Power of a point
    if (n==0):
            return 0,0
    xn=x1
    yn=y1
    n = n - 1
    while n > 0:

        # If power is even
        if (n % 2 == 0):
            # Divide the power by 2
            n = n / 2
            # Multiply base to itself
            [xn,yn]=curve_multiplication(p,xn,yn,xn,yn)
        else:
            # Decrement the power by 1 and make it even
            n = n - 1
            # Take care of the extra value that we took out
            [xn,yn]=curve_multiplication(p,x1,y1,xn,yn)
    return xn,yn   

def curve_random_power(N): #chooses a random power
    rand = random.SystemRandom()
    return rand.randint(0,N)

def curve_random_point(p,Gx,Gy,N): #chooses a random point
    n=curve_random_power(N)
    [xr,yr]=curve_power(p,Gx,Gy,n)
    return xr,yr


k1 = curve_random_power(N)
k2 = curve_random_power(N)

ecp = ECPoint()
ecp1 = ecp*k1
ecp2 = ecp*k2
print(curve_multiplication(p,ecp1.x,ecp1.y,ecp2.x,ecp2.y))
print((ecp1+ecp2).x, (ecp1+ecp2).y)
ecp3 = ECPoint.from_bytes(ecp1.to_bytes())
print(ecp1.x,ecp1.y)
print(ecp3.x,ecp3.y)
"""
print(curve_inversion(1,1))
print(check_if_in_curve(p,a,b,Gx,-Gy))
print(check_if_in_curve(p,a,b,0,0))
print(field_inversion(p,3))
print(field_inversion(p,3)*3 % p)
[x3,y3]=curve_multiplication(p,Gx,Gy,Gx,Gy)
print(check_if_in_curve(p,a,b,x3,y3))
[x3,y3]=curve_multiplication(p,Gx,Gy,0,0)
print(check_if_in_curve(p,a,b,x3,y3))
[x3,y3]=curve_power(p,Gx,Gy,0)
print(check_if_in_curve(p,a,b,x3,y3))
[x3,y3]=curve_power(p,Gx,Gy,1)
print(check_if_in_curve(p,a,b,x3,y3))
[x3,y3]=curve_power(p,Gx,Gy,3)
print(check_if_in_curve(p,a,b,x3,y3))
[x3,y3]=curve_division(p,Gx,Gy,Gx,Gy)
print(x3,y3)
[x3,y3]=curve_power(p,Gx,Gy,2)
print(x3,y3)
[x3,y3]=curve_power(p,Gx,Gy,3)
[x3,y3]=curve_division(p,x3,y3,Gx,Gy)
print(x3,y3)
print(curve_random_power(N))
print(curve_random_power(N))
[xr,yr]=curve_random_point(p,Gx,Gy,N)
print(xr,yr)
[x3,y3]=curve_power(p,Gx,Gy,0)
print(x3,y3)
[x3,y3]=slow_curve_power(p,Gx,Gy,0)
print(x3,y3)
[x3,y3]=curve_power(p,Gx,Gy,17)
print(x3,y3)
[x3,y3]=slow_curve_power(p,Gx,Gy,17)
print(x3,y3)
"""
