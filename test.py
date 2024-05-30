import numpy as np

a = []
for c in range(2): 
    for t in range(3): 
        for i in range(4): 
            for j in range(0, 8, 2): 
                a.append(f'chirp{c}-tx{t}-rx{i}-i{j}')
                a.append(f'chirp{c}-tx{t}-rx{i}-i{j + 1}')
                a.append(f'chirp{c}-tx{t}-rx{i}-q{j}')
                a.append(f'chirp{c}-tx{t}-rx{i}-q{j + 1}')
a = np.array(a)

a1 = a[1::4].copy()
a2 = a[2::4].copy()
a[2::4] = a1
a[1::4] = a2

LVDS0 = a[0::2]
LVDS1 = a[1::2]

LVDS0 = LVDS0.reshape(2, 12, 8)
LVDS1 = LVDS1.reshape(2, 12, 8)
print(LVDS0); exit()

print(LVDS0[:, 0])