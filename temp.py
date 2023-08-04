from multiprocessing import Process


def add_to_list(num, l):
    l.append(num)


if __name__ == '__main__':
    test = range(10000)
    res = []
    process_list = []
    for i in test:
        p = Process(target=add_to_list, args=(i, res))
        p.start()
        process_list.append(p)
    if len(process_list) >= 2:
        for p in process_list:
            p.join()
        process_list = []
    print(res)
