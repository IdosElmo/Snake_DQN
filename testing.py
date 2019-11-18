from collections import deque

x = [0, 0]
b = deque()
c = deque('abc')

x = [1, 0]
b.append(x)
b.append(5)
b.append(10)

# for q in b:
#     print(q)
print(b)

for i in range(len(b) - 1, 0, -1):
    b[i] = b[i-1]

print(b)