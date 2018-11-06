import itertools
import hashlib
import math
import random
import socket
import sys
import urllib.parse

import time#TODO: remove and check what else should be removed
#################### CONSTANTS AND HELPER FUNCTIONS ###################
hash_func = lambda msg: hashlib.sha256(msg).digest()
SUITS = [chr(0x2665), chr(0x2660), chr(0x2666), chr(0x2663)]
RANKS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", " J", " Q", " K", " A"]
CARDS = [r+s for s in SUITS for r in RANKS]
HASH_LEN = len(hash_func(b""))
INT_LEN = 32
HAND_LEN = 2
CARDS_LEN = len(CARDS)
GAME_RESET_PROB = 0.5

def to_b(number, length = INT_LEN):
    return number.to_bytes(length, byteorder="big", signed=False)

from_b = lambda b: int.from_bytes(b, byteorder="big", signed=False)

def get_n_pointers(pnum):
    return CARDS_LEN - pnum*HAND_LEN

def get_permutation(arr, perm_n):
    if len(arr) == 0: return arr
    i = perm_n%len(arr)
    return [arr[i]] + get_permutation(arr[:i]+arr[i+1:], perm_n//len(arr))

########################### GLOBAL VARIABLES ##########################
own_port = int(sys.argv[1])
others_addresses = []
for i in range(2, len(sys.argv)):
    parsed = urllib.parse.urlparse("//{}".format(sys.argv[i]))
    others_addresses.append((parsed.hostname, int(parsed.port)))

N_PLAYERS = len(others_addresses) + 1
ADVANCE_MESSAGE_LEN = HAND_LEN*INT_LEN*N_PLAYERS
READY_ADVANCE_MESSAGE = b'\x00'*ADVANCE_MESSAGE_LEN

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#TODO: SSL
sock.settimeout(0.1)
rand = random.SystemRandom()

#Not yet fully initialized:
in_conn_dict = {}
out_conn_dict = {}
temp_conn_dict = dict((addr, socket.socket(socket.AF_INET, socket.SOCK_STREAM)) for addr in others_addresses)
joint_salt = None
ordered_ids = None

###################### BASIC PROTOCOL OPERATIONS ######################
def split_bytes(bytes, stride, n_parts):
    return [bytes[i*stride:(i+1)*stride] for i in range(n_parts)]

def generate_message(k, k_lim):
    k += k_lim*rand.randint(0, 2**(INT_LEN*8)/k_lim - 1) #pad with random

    msg = to_b(k)
    commitment = hash_func(msg + joint_salt)
    return msg, commitment, k

def check_message(msg, commitment):
    assert(commitment == hash_func(msg + joint_salt))

def broadcast_message(msg, max_msg_len=-1):
    if max_msg_len == -1:
        max_msg_len = len(msg)

    msg = msg.ljust(max_msg_len, b'\x00')

    for s in out_conn_dict.values():
        s.send(msg)

    ordered_msgs = [None]*N_PLAYERS
    ordered_msgs[ordered_ids.index(own_id)] = msg
    msgs_dict = {own_id:msg}
    for id_,c in in_conn_dict.items():
        msgs_dict[id_] = c.recv(max_msg_len)
        ordered_msgs[ordered_ids.index(id_)] = msgs_dict[id_]

    checksum = hash_func(b"".join(ordered_msgs))
    for s in out_conn_dict.values():
        s.send(checksum)

    for c in in_conn_dict.values():
        assert(checksum == c.recv(HASH_LEN))

    return msgs_dict

def choose_random_int(n):
    own_msg, own_commitment, k = generate_message(0,1)

    commitments_dict = broadcast_message(own_commitment)

    for s in out_conn_dict.values():
        s.send(own_msg)

    for id_,c in in_conn_dict.items():
        msg = c.recv(INT_LEN)
        check_message(msg, commitments_dict[id_])
        k ^= from_b(msg)

    return k%n

def validate_pointer_chain(pnum, pointer_chain, first_pointer, pointer_commitments_dict, pnum_to_id):
    assert(len(pointer_chain) == pnum+1)

    pointer = first_pointer
    for i,masked_pointer in enumerate(pointer_chain):
        assert(pointer < get_n_pointers(pnum-i))
        commitment = pointer_commitments_dict[pnum_to_id[pnum-i]][pointer*HASH_LEN:(pointer+1)*HASH_LEN]
        check_message(to_b(masked_pointer), commitment)
        pointer = masked_pointer%CARDS_LEN + HAND_LEN

################ OBLIVIOUS TRANSFER AND ELLIPTIC CURVES ###############
class ECPoint:#TODO: name of curve?
    a = -3
    b = 41058363725152142129326129780047268409114441015993725554835256314039467401291
    p = 2**256-2**224+2**192+2**96-1
    Gx = 48439561293906451759052585252797914202762949526041747995844080717082404635286
    Gy = 36134250956749795798585127919587881956611106672985015071877198253568414405109
    N = 115792089210356248762697446949407573529996955224135760342422259061068512044369
    LEN = 64

    def from_bytes(bytes):
        x = int.from_bytes(bytes[ECPoint.LEN//2:], byteorder="big", signed=False)%ECPoint.p
        y = int.from_bytes(bytes[:ECPoint.LEN//2], byteorder="big", signed=False)%ECPoint.p
        assert((y**2)%ECPoint.p == (x**3 + ECPoint.a*x + ECPoint.b)%ECPoint.p or (x == 0 and y == 0))
        return ECPoint(x,y)

    def to_bytes(self):
        return (self.x + (self.y<<(ECPoint.LEN*4))).to_bytes(ECPoint.LEN, byteorder="big", signed=False)

    def __init__(self, x = Gx, y = Gy):
        self.x, self.y = x%ECPoint.p, y%ECPoint.p

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
        return ECPoint(x%ECPoint.p, y%ECPoint.p)

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

def oblivious_transfer_offer(options, id_):
    y = rand.randint(0,ECPoint.p-1)
    B = ECPoint()
    S = B*y
    T = S*y

    out_conn_dict[id_].send(S.to_bytes())
    R = ECPoint.from_bytes(in_conn_dict[id_].recv(ECPoint.LEN))

    Ry = R*y
    Tj = ECPoint(0,0)
    encrypted_options = []
    for opt in options:
        key_j = from_b(hash_func((Ry-Tj).to_bytes())[:INT_LEN])
        encrypted_options.append(to_b(key_j^opt))
        Tj = Tj + T

    out_conn_dict[id_].send(b"".join(encrypted_options))

def oblivious_transfer_choose(choice, n_options, id_):#TODO: multiple choices
    B = ECPoint()
    S = ECPoint.from_bytes(in_conn_dict[id_].recv(ECPoint.LEN))

    x = rand.randint(0,ECPoint.p-1)
    R = S*choice + B*x

    out_conn_dict[id_].send(R.to_bytes())
    msg = in_conn_dict[id_].recv(INT_LEN*n_options)
    key = from_b(hash_func((S*x).to_bytes())[:INT_LEN])
    return from_b(msg[choice*INT_LEN:(choice+1)*INT_LEN])^key

############################## GAME FLOW ##############################
def reveal_face_up_chain(first_pointer, pointer_commitments_dict, pnum_to_id):
    pointer = first_pointer + HAND_LEN
    face_up_chain = []
    for pnum in range(N_PLAYERS-1,-1,-1):
        if pnum == own_pnum:
            face_up_chain.append(own_masked_pointers[pointer])
            for s in out_conn_dict.values():
                s.send(to_b(own_masked_pointers[pointer]))
        else:
            face_up_chain.append(from_b(in_conn_dict[pnum_to_id[pnum]].recv(INT_LEN)))
        pointer = face_up_chain[-1]%CARDS_LEN + HAND_LEN

    validate_pointer_chain(N_PLAYERS-1, face_up_chain, first_pointer + HAND_LEN, pointer_commitments_dict, pnum_to_id)
    return face_up_chain

def process_players_input(expose_advance_message, wait_for_decision, message):
    if wait_for_decision:
        player_input = input(message+"\n").strip().lower()
    else:
        player_input = ""

    if player_input == "s":
        print("You exposed your hand.")
        advance_messages_dict = broadcast_message(expose_advance_message, ADVANCE_MESSAGE_LEN)
    else:
        if player_input == "f":
            print("You folded your hand.")
        advance_messages_dict = broadcast_message(READY_ADVANCE_MESSAGE, ADVANCE_MESSAGE_LEN)

    for pnum,id_ in pnum_to_id.items():
        if id_ == own_id or advance_messages_dict[id_] == READY_ADVANCE_MESSAGE:
            continue

        merged_chains = list(map(from_b, split_bytes(advance_messages_dict[id_], INT_LEN, HAND_LEN*(pnum+1))))
        chains = [merged_chains[i*(pnum+1):(i+1)*(pnum+1)] for i in range(HAND_LEN)]
        for i in range(HAND_LEN):
            validate_pointer_chain(pnum, chains[i], i, pointer_commitments_dict, pnum_to_id)
        print("Player %d exposed their hand:"%(pnum), " ".join(map(card_from_chain, chains)))

    return not player_input in ("s", "f")

######################## ESTABLISH CONNECTIONS ########################
own_id = to_b(rand.randint(0, 2**(8*INT_LEN)-1))
own_id_msg = to_b(own_port) + own_id

sock.bind(("", own_port))
sock.listen(len(others_addresses))


while len(in_conn_dict) < N_PLAYERS-1 or len(others_addresses) > 0:
    try:
        conn, addr = sock.accept()

        id_msg = conn.recv(INT_LEN + INT_LEN)
        port = from_b(id_msg[:INT_LEN])
        key = (addr[0], port)

        if not key in temp_conn_dict:#TODO: switch order TODO:??
            conn.close()
        else:
            id_ = id_msg[INT_LEN:]
            assert(not id_ in in_conn_dict and id_ != own_id)
            out_conn_dict[id_] = key#temporary
            in_conn_dict[id_] = conn

    except socket.timeout:
        pass

    if len(others_addresses) == 0:
        continue

    try:
        key = others_addresses[-1]
        temp_conn_dict[key].connect(key)
        temp_conn_dict[key].send(own_id_msg)
        others_addresses.pop()
    except (ConnectionRefusedError,ConnectionAbortedError):
        pass

for id_ in out_conn_dict:
    out_conn_dict[id_] = temp_conn_dict[out_conn_dict[id_]]

######################## CHOOSE PLAYERS ORDER ########################
ordered_ids = sorted([own_id] + list(in_conn_dict.keys()))
joint_salt = b"".join(ordered_ids)

id_perm_n = choose_random_int(math.factorial(N_PLAYERS))
id_perm = get_permutation(ordered_ids, id_perm_n)
own_pnum = id_perm.index(own_id)
pnum_to_id = dict((id_perm.index(id_), id_) for id_ in ordered_ids)

game_n = 0
while game_n < 5:#TODO: remove limit
    ############### CHOOSE AND COMMIT TO INDEX PERMUTATIONS ##############
    own_n_pointers = get_n_pointers(own_pnum)
    own_pointers_perm = rand.randint(0, math.factorial(own_n_pointers)-1)
    own_pointers = get_permutation(list(range(own_n_pointers)), own_pointers_perm)

    own_pointer_messages, own_pointer_commitments, own_masked_pointers = zip(*[generate_message(k, CARDS_LEN) for k in own_pointers])

    pointer_commitments_dict = broadcast_message(b"".join(own_pointer_commitments), HASH_LEN*CARDS_LEN)

    ############# WITH SOME PROBABILITY VALIDATE COMMITMENTS #############
    continue_game = int(rand.random() > GAME_RESET_PROB)
    msgs_dict = broadcast_message(to_b(continue_game))
    if not all(map(from_b, msgs_dict.values())):
        pointer_messages_dict = broadcast_message(b"".join(own_pointer_messages), INT_LEN*CARDS_LEN)
        for pnum, id_ in pnum_to_id.items():
            pointer_messages = split_bytes(pointer_messages_dict[id_], INT_LEN, get_n_pointers(pnum))
            commitments = split_bytes(pointer_commitments_dict[id_], HASH_LEN, get_n_pointers(pnum))

            list(map(check_message, pointer_messages, commitments))

            pointers = [from_b(pm)%CARDS_LEN for pm in pointer_messages]
            assert(set(pointers) == set(range(get_n_pointers(pnum))))
        continue

    game_n += 1
    print("\n" + "-"*40 + "\nGame %d (You are player %d)"%(game_n, own_pnum))

    ############### COMPUTE POINTER CHAINS FOR ALL PLAYERS ###############
    players_communication_order = [(t+1) + (own_pnum-t>0)*(own_pnum-2*t-2) for t in range(N_PLAYERS-1)]#TODO: this is the slowest order
    own_pointer_chains = [[p] for p in own_masked_pointers[:HAND_LEN]]

    for pnum in players_communication_order:
        if pnum < own_pnum:
            for chain in own_pointer_chains:
                pointer = chain[-1]%CARDS_LEN
                assert(pointer < get_n_pointers(pnum)-HAND_LEN)
                chain.append(oblivious_transfer_choose(pointer, get_n_pointers(pnum)-HAND_LEN, pnum_to_id[pnum]))
        else:
            for _ in range(HAND_LEN):
                oblivious_transfer_offer(own_masked_pointers[HAND_LEN:], pnum_to_id[pnum])

    for i in range(HAND_LEN):
        validate_pointer_chain(own_pnum, own_pointer_chains[i], i, pointer_commitments_dict, pnum_to_id)
    expose_advance_message = b"".join(map(to_b, [p for c in own_pointer_chains for p in c]))

    ################ CHOOSE POINTERS TO CARDS PERMUTATION ################
    cards_perm_n = choose_random_int(math.factorial(CARDS_LEN))
    cards_perm = get_permutation(list(range(CARDS_LEN)), cards_perm_n)
    card_from_chain = lambda c: CARDS[cards_perm[c[-1]%CARDS_LEN]]

    ################# CHOOSE FACE-UP CARDS REVEAL ORDER ##################
    reveal_perm_n = choose_random_int(math.factorial(CARDS_LEN - HAND_LEN*N_PLAYERS))
    reveal_perm = get_permutation(list(range(CARDS_LEN - HAND_LEN*N_PLAYERS)), reveal_perm_n)

    wait_for_decision = True
    ######################### FIRST BETTING ROUND ########################
    print("Your cards are:", " ".join(map(card_from_chain, own_pointer_chains)))
    wait_for_decision &= process_players_input(expose_advance_message, wait_for_decision, "Enter S to expose your hand or F to fold it. Hit Enter to see the flop.")

    ######################## SECOND BETTING ROUND ########################
    flop_chains = [reveal_face_up_chain(reveal_perm[i], pointer_commitments_dict, pnum_to_id) for i in range(3)]
    print("The flop is", "".join(map(card_from_chain, flop_chains)))
    wait_for_decision &= process_players_input(expose_advance_message, wait_for_decision, "Enter S to expose your hand or F to fold it. Hit Enter to see the turn.")

    ######################## THIRD BETTING ROUND #########################
    turn_chain = reveal_face_up_chain(reveal_perm[3], pointer_commitments_dict, pnum_to_id)
    print("The turn is", card_from_chain(turn_chain))
    wait_for_decision &= process_players_input(expose_advance_message, wait_for_decision, "Enter S to expose your hand or F to fold it. Hit Enter to see the river.")

    ######################### LAST BETTING ROUND #########################
    river_chain = reveal_face_up_chain(reveal_perm[4], pointer_commitments_dict, pnum_to_id)
    print("The river is", card_from_chain(river_chain))
    wait_for_decision &= process_players_input(expose_advance_message, wait_for_decision, "Enter S to expose your hand or F to fold it. Hit Enter to see other players' hands (you will get one more chance to expose your hand).")

    ########################### SHOW/MUCK ROUND ##########################
    process_players_input(expose_advance_message, wait_for_decision, "Enter S to expose your hand or hit Enter to muck your hand.")
