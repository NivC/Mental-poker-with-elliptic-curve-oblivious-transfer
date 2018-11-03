import cryptography.hazmat.backends
import cryptography.hazmat.primitives.asymmetric.rsa
import itertools
import hashlib
import math
import random
import socket
import sys
import time
import urllib.parse

#key = cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key(public_exponent=65537, key_size=1024, backend=cryptography.hazmat.backends.default_backend())
#print(dir(key))
#print(key.private_numbers())
#print(dir(key.private_numbers()))
#sys.exit(1)

############################## CONSTANTS ##############################
SALT_LEN = 32
HASH_LEN = 32
INT_LEN = 64
HAND_LEN = 2
CARDS_LEN = 52#TODO: len(cards_dict)

########################## GLOBAL  VARIABLES ##########################
own_port = int(sys.argv[1])
others_addresses = []
for i in range(2, len(sys.argv)):
    parsed = urllib.parse.urlparse("//{}".format(sys.argv[i]))
    others_addresses.append((parsed.hostname, int(parsed.port)))

n_players = len(others_addresses) + 1

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#TODO: SSL
sock.settimeout(0.1)
rand = random.SystemRandom()

#Fully initialized later:
in_conn_dict = dict((addr,None) for addr in others_addresses)
out_conn_dict = dict((addr,socket.socket(socket.AF_INET, socket.SOCK_STREAM)) for addr in others_addresses)
salts_order_dict = None
joint_salt = None

###################### BASIC PROTOCOL OPERATIONS ######################
def get_n_indices(pnum):
    return CARDS_LEN - pnum*HAND_LEN

def get_permutation(arr, perm_n):
    if len(arr) == 0: return arr
    i = perm_n%len(arr)
    return [arr[i]] + get_permutation(arr[:i]+arr[i+1:], perm_n//len(arr))

def generate_message(k, k_lim):
    k += k_lim*rand.randint(0, 2**(INT_LEN*8)/k_lim - 1) #pad with random

    msg = k.to_bytes(INT_LEN, byteorder="big", signed=False)
    commitment = hashlib.sha256(msg + joint_salt).digest()
    return msg, commitment, k

def check_message(msg, commitment):
    assert(commitment == hashlib.sha256(msg + joint_salt).digest())

def broadcast_message(msg, max_msg_len=-1):
    if max_msg_len == -1:
        max_msg_len = len(msg)

    msg = msg.ljust(max_msg_len, b'\x00')

    all_msgs = [None]*n_players
    all_msgs[salts_order_dict[own_salt]] = msg
    for s in out_conn_dict.values():
        s.send(msg)

    for k,c in in_conn_dict.items():
        ###c.recv(msg)
        pass

def comm_random_int(n):
    own_msg, own_commitment, k = generate_message(0,1)

    for s in out_conn_dict.values():
        s.send(own_commitment)

    commitments = []
    for c in in_conn_dict.values():
        commitments.append(c.recv(HASH_LEN))

    for s in out_conn_dict.values():
        s.send(own_msg)

    for i,c in enumerate(in_conn_dict.values()):
        msg = c.recv(INT_LEN)
        check_message(msg, commitments[i])
        k ^= int.from_bytes(msg, byteorder="big", signed=False)

    #TODO: compare result with everybody else
    return k%n

try:
######################## ESTABLISH CONNECTIONS ########################
    sock.bind(("", own_port))
    sock.listen(len(others_addresses))

    while None in in_conn_dict.values() or len(others_addresses) > 0:
        try:
            conn, addr = sock.accept()

            port = int.from_bytes(conn.recv(INT_LEN), byteorder="big", signed=False)
            key = (addr[0], port)

            if not key in in_conn_dict:
                conn.close()
            else:
                in_conn_dict[key] = conn

        except socket.timeout:
            pass

        if len(others_addresses) == 0:
            continue

        try:
            key = others_addresses[-1]
            out_conn_dict[key].connect(key)
            out_conn_dict[key].send(own_port.to_bytes(INT_LEN, byteorder="big", signed=False))
            others_addresses.pop()
        except (ConnectionRefusedError,ConnectionAbortedError):
            pass

########################### EXCHANGE SALTS ###########################
    own_salt = rand.randint(0, 2**(8*SALT_LEN)-1).to_bytes(SALT_LEN, byteorder="big", signed=False)

    for s in out_conn_dict.values():
        s.send(own_salt)

    salts_dict = {}
    for k,c in in_conn_dict.items():
        salts_dict[k] = c.recv(SALT_LEN)

    ordered_salts = sorted([own_salt] + list(salts_dict.values()))
    assert(len(set(ordered_salts)) == n_players) #salts are assumed to be unique
    joint_salt = b''.join(ordered_salts)
    salts_order_dict = dict((ordered_salts.index(salt), k) for k,salt in salts_dict.items())

######################## CHOOSE PLAYERS ORDER ########################
    salts_perm_n = comm_random_int(math.factorial(n_players))

    salts_perm = get_permutation(ordered_salts, salts_perm_n)
    own_pnum = salts_perm.index(own_salt)
    pnum_dict = dict((salts_perm.index(v),k) for k,v in salts_dict.items())

############### CHOOSE AND COMMIT TO INDEX PERMUTATIONS ##############
    own_n_indices = get_n_indices(own_pnum)
    index_perm_n = rand.randint(0, math.factorial(own_n_indices)-1)
    index_perm = get_permutation(list(range(own_n_indices)), index_perm_n)

    own_index_messages, own_index_commitments, own_index_ks = zip(*[generate_message(k, CARDS_LEN) for k in index_perm])
    index_commitments = {}

    for s in out_conn_dict.values():
        s.send(b''.join(own_index_messages))
    
    for pnum,k in pnum_dict.items():
        temp = in_conn_dict[k].recv(get_n_indices(pnum)*HASH_LEN)
        index_commitments[k] = [temp[j:j+HASH_LEN] for j in range(0,len(temp),HASH_LEN)]

finally:
    sock.close()
