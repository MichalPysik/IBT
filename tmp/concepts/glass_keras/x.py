f = open('glass.csv', 'r')
w = open('glass_modified.csv', 'w')

w.write(f.readline())

for line in f:
    w.write(line[:-2])
    w.write(str(int(line[-2])-1) + '\n')

f.close()
w.close()