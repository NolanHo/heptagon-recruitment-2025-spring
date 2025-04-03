import re
with open("sgemm.txt", "r") as f:
    lines = f.readlines()

result = []
for idx in range(len(lines)):
    if "M(num_tiles)" in lines[idx]:
        size = lines[idx].strip()
        # M(num_tiles) x N(oc) x K(ic): 576 x 512 x 512
        size = re.findall(r'\d+', size)
        t = lines[idx + 1].strip()
        t = re.findall(r'\d+ms', t)
        size = [int(s) for s in size]
        t = int(t[0][:-2])
        result.append({"size": size, "t": t})
        # print(result[-1])

for idx in range(0, len(result), 3):
    t1, t2, t3 = result[idx]["t"], result[idx + 1]["t"], result[idx + 2]["t"]
    avg_t = (t1 + t2 + t3) / 3
    print(f"{result[idx]['size']} -> {avg_t}ms")
