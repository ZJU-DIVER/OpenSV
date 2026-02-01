from typing import Optional, Any

import numpy as np
import math
from tqdm import tqdm, trange
from functools import partial
from multiprocessing import Pool, Manager, shared_memory
import math
import struct

from opensv.utils.utils import get_utility, split_permutation_num

Array = np.ndarray


class shm_dict:
    def _make_memory(self, sz: int) -> None:
        a = -np.ones((sz), dtype=float)
        self.shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=self.shm.buf)
        b[:] = a[:]

    def init(self, n: int, limit: int = None, rev: int = 0) -> None:
        self.n = n
        self.comb_pres = []
        self.comb_list = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(i + 1):
                self.comb_list[i][j] = math.comb(i, j)
        for i in range(n + 1):
            self.comb_pres.append(self.comb_list[n][i])
        for i in range(1, n + 1):
            self.comb_pres[i] = self.comb_pres[i] + self.comb_pres[i - 1]

        if limit is None:
            limit = n
        alloc = 0
        for i in range(0, limit + 1):
            alloc += self.comb_list[n][i]
        self._make_memory(alloc)
        self.rev = rev

    def _pool_mapping(self, s: int) -> int:
        if s <= 0:
            return 0

        pcnt = s.bit_count()
        num = self.comb_pres[pcnt - 1]
        cur = 1
        for i in range(0, self.n):
            if (s >> i) & 1:
                num += self.comb_list[i][cur]
                cur += 1

        return num

    def _set(self, x: int, v: float) -> None:
        self.shm.buf[x << 3 : (x + 1) << 3] = struct.pack("<d", v)

    def _get(self, x: int) -> float:
        return struct.unpack("<d", self.shm.buf[x << 3 : (x + 1) << 3])[0]

    def set(self, index: Array, v: float) -> None:
        id = 0
        for i in index:
            id |= 1 << i
        if self.rev:
            id = id ^ ((1 << self.n) - 1)
        id = self._pool_mapping(id)
        return self._set(id, v)

    def get(self, index: Array) -> float:
        id = 0
        for i in index:
            id |= 1 << i
        if self.rev:
            id = id ^ ((1 << self.n) - 1)
        id = self._pool_mapping(id)
        return self._get(id)

    # def __del__(self):
    #     self.shm.close()
    #     self.shm.unlink()


class native_dict:
    def init(self) -> None:
        self.mem = Manager().dict()

    def set(self, index: Array, v: float) -> None:
        id = 0
        for i in index:
            id |= 1 << i
        self.mem[id] = v

    def get(self, index: Array) -> float:
        id = 0
        for i in index:
            id |= 1 << i
        if id in self.mem:
            return self.mem[id]
        return -1


class hybrid_dict:
    def init(self, n: int, limit_shm: int = None, limit_dict: int = None) -> None:
        self.n = n
        self.limit_shm = limit_shm or -1
        self.limit_dict = limit_dict or n
        self.mid = native_dict()
        if self.limit_shm >= 0:
            self.lef = shm_dict()
            self.rig = shm_dict()
            self.lef.init(n, limit_shm)
            self.rig.init(n, limit_shm, 1)
        self.mid.init()

    def set(self, index: Array, v: float) -> None:
        cnt = index.shape[0]
        if cnt <= self.limit_shm:
            return self.lef.set(index, v)
        if self.n - cnt <= self.limit_shm:
            return self.rig.set(index, v)
        if cnt <= self.limit_dict or self.n - cnt <= self.limit_dict:
            return self.mid.set(index, v)
        return None

    def get(self, index: Array) -> float:
        cnt = index.shape[0]
        if cnt <= self.limit_shm:
            return self.lef.get(index)
        if self.n - cnt <= self.limit_shm:
            return self.rig.get(index)
        if cnt <= self.limit_dict or self.n - cnt <= self.limit_dict:
            return self.mid.get(index)
        return -1

    def _set(self, index: int, v: float) -> None:
        self.mid.mem[index] = v

    def _get(self, index: int) -> float:
        return self.mid.mem[index]


cost_after_find = True


def get_utility_memorized(
    mem: hybrid_dict,
    x: Array,
    y: Array,
    index: Array,
    x_valid: Array,
    y_valid: Array,
    clf,
):
    global cost_after_find
    res = mem.get(index)
    if res >= 0:
        # mem._set(-1, mem._get(-1) + 1)
        return res, cost_after_find
    x_temp, y_temp = x[index, :], y[index]
    u = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
    mem.set(index, u)
    return u, True


def subtask1(
    mem: hybrid_dict,
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    mexp: int = 0,
    sub_players: range = None,
) -> tuple[Array, Array]:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = np.asarray(list(range(N)))

    sh = np.zeros(N * N).reshape((N, N))
    sum_cuad = np.zeros(N)
    for player in tqdm(sub_players):
        for loc in range(N):
            for _ in range(mexp):
                rng.shuffle(idxes)
                xloc = np.argwhere(idxes == player)
                idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]

                acc, _ = get_utility_memorized(
                    mem, x_train, y_train, idxes[:loc], x_valid, y_valid, clf
                )
                new_acc, _ = get_utility_memorized(
                    mem, x_train, y_train, idxes[: loc + 1], x_valid, y_valid, clf
                )
                xoi = new_acc - acc

                sh[player][loc] += xoi
                sum_cuad[loc] += xoi * xoi

    return (sh, sum_cuad)


def subtask2(
    mem: hybrid_dict,
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    mexp: int = 0,
    mst: Array = None,
    old_sh: Array = None,
    sub_players: range = None,
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = np.asarray(list(range(N)))

    sh = np.zeros(N * N).reshape((N, N))
    for player in tqdm(sub_players):
        for loc in range(N):
            m = int(max(0, mst[player][loc]))
            utility = m * 2
            mcnt = 0
            max_limit = utility + math.comb(N - 1, loc) * N * 2
            while utility > 0:
                rng.shuffle(idxes)
                xloc = np.argwhere(idxes == player)[0][0]
                idxes[xloc], idxes[loc] = idxes[loc], idxes[xloc]

                acc, used = get_utility_memorized(
                    mem, x_train, y_train, idxes[:loc], x_valid, y_valid, clf
                )
                if used:
                    utility -= 1
                new_acc, used = get_utility_memorized(
                    mem, x_train, y_train, idxes[: loc + 1], x_valid, y_valid, clf
                )
                if used:
                    utility -= 1

                xoi = new_acc - acc
                sh[player][loc] += xoi
                mcnt += 1
                max_limit -= 1
                if max_limit < 0:
                    break
            sh[player][loc] = (old_sh[player][loc] + sh[player][loc]) / (mcnt + mexp)

    return sh


def stratified_mem_ulimit_mp(
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    num_proc: int = 1,
    num_utility: int = 500,
    mem_param: list = [None, None, True],
) -> Array:
    print(
        get_utility(x_train, y_train, x_valid, y_valid, clf),
        get_utility([], [], x_valid, y_valid, clf),
    )
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)

    N = len(y_train)

    global cost_after_find
    cost_after_find = mem_param[2]
    mem = hybrid_dict()
    mem.init(N, mem_param[0], mem_param[1])
    mem._set(-1, 0)

    M = num_utility // 2
    sh = np.zeros(N * N).reshape((N, N))
    sum_cuad = np.zeros(N)
    mexp = np.floor(np.max([M / (2 * N * N), 2])).astype(int)

    sub_length = split_permutation_num(N, num_proc)
    cur = 0
    sub_players = []
    for i in sub_length:
        sub_players.append(range(cur, cur + i))
        cur += i

    pool = Pool()
    func = partial(subtask1, mem, x_train, y_train, x_valid, y_valid, clf, mexp)
    ret = pool.map(func, sub_players)
    pool.close()
    pool.join()

    sh = np.sum([r[0] for r in ret], axis=0)
    sum_cuad = np.sum([r[1] for r in ret], axis=0)

    # M = (num_utility - (mexp * N * N * 2) + mem._get(-1)) // 2
    M = (num_utility - (mexp * N * N * 2)) // 2
    s2 = np.zeros(N * N).reshape((N, N))
    for player in trange(N):
        s2[player] = sum_cuad
    s2 = (s2 - sh**2 / mexp) / (mexp - 1)
    sum_s2 = np.sum(s2)
    mst = M * s2 / sum_s2 - mexp
    mst = np.clip(mst, 0, None)
    mst = np.floor(mst * M / np.sum(mst))

    # mem = hybrid_dict()
    # mem.init(N, mem_param[0], mem_param[1])
    # mem._set(-1, 0)

    pool = Pool()
    func = partial(
        subtask2, mem, x_train, y_train, x_valid, y_valid, clf, mexp, mst, sh
    )
    ret = pool.map(func, sub_players)
    pool.close()
    pool.join()
    ret_val = np.asarray(ret)

    sh = ret_val.sum(axis=0)
    val = np.sum(sh, axis=1) / N
    # val = val * (final_acc - init_acc) / np.sum(val)

    print(f"Duplicated count: {mem._get(-1)} / {num_utility}")
    with open("log.txt", "a") as f:
        f.write(f"stratified_mem_ulimit_mp,{N},{num_utility},{mem._get(-1)}\n")

    return val
