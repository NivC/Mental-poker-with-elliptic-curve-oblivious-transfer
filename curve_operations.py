import random

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
            x3=0
            y3=0
            return x3,y3
    else:
        lamda=(y2-y1)*field_inversion(p,x2-x1) % p
        

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
    print("slow")

    for x in range(0, n-1):
        print(n-x-1)
        print(xn)
        print("")
        [xn,yn]=curve_multiplication(p,x1,y1,xn,yn)
    return xn,yn    

def curve_power(p,x1,y1,n): #Power of a point

    xn=0
    yn=0

    while n > 0:

        # If power is even
        if (n % 2 == 0):
            # Divide the power by 2
            n = n // 2
            # Multiply base to itself
            [x1,y1]=curve_multiplication(p,x1,y1,x1,y1)
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


# https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf Page 91 (100)
a=-3 # elliptic parameter
b=41058363725152142129326129780047268409114441015993725554835256314039467401291 # elliptic parameter
p=2**256-2**224+2**192+2**96-1 # prime
#co-factor is h=1
Gx=48439561293906451759052585252797914202762949526041747995844080717082404635286 #initial point x
Gy=36134250956749795798585127919587881956611106672985015071877198253568414405109 #initial point y
N=115792089210356248762697446949407573529996955224135760342422259061068512044369  # group order



print("           HERE         ")
[x3,y3]=curve_power(p,Gx,Gy,N)
print(x3,y3)
