viable = []
for i in range(2, 1001):
    for j in range(2, 1001):
        for k in range(2, 1001):
            if i*j*k*k <= 1000:
                viable.append((i, j, k))

for v in viable:
    print(v)

print(len(viable))
