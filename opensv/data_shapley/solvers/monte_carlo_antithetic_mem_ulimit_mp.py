from typing import Optional, Any

import numpy as np
from tqdm import tqdm
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

    def __del__(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except:
            pass


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


def subtask(
    mem,
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    init_acc: float = 0,
    final_acc: float = 1,
    epsilon: float = 0,
    num_utility: int = 0,
) -> Array:
    rng = np.random.default_rng()
    N = len(y_train)
    idxes = np.asarray(list(range(N)))
    val = np.zeros(N)
    perm = 0
    with tqdm(total=num_utility) as pbar:
        while num_utility > 0:
            rng.shuffle(idxes)
            acc = init_acc
            rev_acc = final_acc
            perm += 1
            for i in range(1, N + 1):
                if (abs(final_acc - acc) <= epsilon) and (
                    abs(rev_acc - init_acc) <= epsilon
                ):
                    break
                new_acc, used = get_utility_memorized(
                    mem, x_train, y_train, idxes[:i], x_valid, y_valid, clf
                )
                if used:
                    num_utility -= 1
                    pbar.update(1)
                new_rev_acc, used = get_utility_memorized(
                    mem, x_train, y_train, idxes[i:], x_valid, y_valid, clf
                )
                if used:
                    num_utility -= 1
                    pbar.update(1)
                val[idxes[i - 1]] += new_acc - acc + rev_acc - new_rev_acc
                acc = new_acc
                rev_acc = new_rev_acc
            if num_utility <= 0:
                break
    return (val, perm)


def monte_carlo_antithetic_mem_ulimit_mp(
    x_train: Array,
    y_train: Array,
    x_valid: Optional[Array] = None,
    y_valid: Optional[Array] = None,
    clf: Optional[Any] = None,
    num_proc: int = 1,
    num_utility: int = 500,
    mem_param: list = [None, None, True],
    truncated_threshold: float = 0,
) -> Array:
    print(
        get_utility(x_train, y_train, x_valid, y_valid, clf),
        get_utility([], [], x_valid, y_valid, clf),
    )
    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)

    T = num_utility - 2
    N = len(y_train)
    epsilon = truncated_threshold
    global cost_after_find
    cost_after_find = mem_param[2]
    mem = hybrid_dict()
    mem.init(N, mem_param[0], mem_param[1])
    mem._set(-1, 0)

    init_acc = get_utility([], [], x_valid, y_valid, clf)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    sub_length = split_permutation_num(T, num_proc)
    pool = Pool()
    func = partial(
        subtask,
        mem,
        x_train,
        y_train,
        x_valid,
        y_valid,
        clf,
        init_acc,
        final_acc,
        epsilon,
    )
    ret = pool.map(func, sub_length)
    pool.close()
    pool.join()
    ret_val = np.sum([r[0] for r in ret], axis=0)
    # ret_perm = np.sum([r[1] for r in ret], axis=-1)
    # val = ret_val / ret_perm / 2
    # val = ret_val
    val = ret_val * (final_acc - init_acc) / np.sum(ret_val)

    print(f"Duplicated count: {mem._get(-1)} / {num_utility}")
    with open("log.txt", "a") as f:
        f.write(
            f"monte_carlo_antithetic_mem_ulimit_mp,{N},{num_utility},{mem._get(-1)}\n"
        )

    return val
