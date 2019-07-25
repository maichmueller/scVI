from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool, Lock, Value
from tqdm import tqdm
_LOCK = None
_LINE_COUNT = None


class TEST:

    def __init__(self):
        return

    def other_func(self, k):
        return k * 2

    def inc_val(self, k):
        with open("someshit.txt", "w") as f:
            # with _LINE_COUNT.get_lock():
            _LINE_COUNT.value += k
            f.write(str(_LINE_COUNT.value))

    def costly_func(self, some_param):
        with ProcessPoolExecutor(2) as pool:
            # queue = tqdm(pool.starmap(self.other_func, range(-1, -10)))
            queue = list(pool.submit(self.other_func, i)
                        for i in range(1, 6))
            for q in queue:
                q = q.result()
                _LOCK.acquire()
                self.inc_val(q)
                print(_LINE_COUNT.value)
                _LOCK.release()

    def run_mp(self):

        def init_gloabls(lock, val):
            global _LOCK
            global _LINE_COUNT
            _LOCK = lock
            _LINE_COUNT = val

        with ThreadPoolExecutor(cpu_count() // 2,
                                initializer=init_gloabls,
                                initargs=(Lock(), Value("i", 2))
                                ) as pool:
            ds_queue = list(pool.submit(self.costly_func,
                                        i)

                            for i in
                            range(2)
                            )
            for future in as_completed(ds_queue):
                future.result()

if __name__ == '__main__':
    test = TEST()
    test.run_mp()
