import os 
import numpy as np
import options

TrainingFile = 'ModelNet10'
path = os.walk('./Data/' + TrainingFile)

def distance(x, y):
    return sum((x-y) ** 2) ** 0.5

def cal_area(x, y, z):
    a = distance(x, y)
    b = distance(x, z)
    c = distance(y, z)
    s = (a + b + c) * 0.5
    temp = (s * (s - a) * (s - b) * (s - c))
    if temp < 0:
        temp = 0
    return temp ** 0.5

num = 0
cnt = 0
opt = options.Option()
for root, dirs, files in path:
    print(root)
    dir = root.split('/')[-1]
    if dir == TrainingFile:
        pass
    elif dir == 'train' or dir == 'test':
        for item in files:
            if item[-1] != 'f':
                continue
            with open(root + '/' + item) as f:
                head = 2
                buf = f.readlines()
                if buf[0][-2] != 'F':
                    temp = buf[0][3:]
                    head = 1
                else:
                    temp = buf[1].split()
                num = int(temp[0])
                mesh_num = int(temp[1])
                data = np.zeros((num, 3), dtype='float')
                mesh = np.zeros((mesh_num, 3, 3), dtype='float')
                area = np.zeros(mesh_num, dtype='float')
                sum_area = 0
                point = np.zeros((opt.pointnum, 3), dtype='float')

                for i in range(num):
                    temp = buf[head+i].split()
                    for j in range(3):
                        data[i][j] = float(temp[j])
                
                for i in range(mesh_num):
                    temp = buf[i+head+num].split()
                    if int(temp[0]) != 3:
                        print('Error!')
                        exit(0)
                    for j in range(3):
                        mesh[i][j] = data[int(temp[j+1])]
                    area[i] = cal_area(mesh[i][0], mesh[i][1], mesh[i][2])
                    if area[i] != area[i]:
                        print(i)
                        print(mesh[i])
                    sum_area = sum_area + area[i]
                
                prob = area / sum_area
                idx = np.random.choice(range(mesh_num), opt.pointnum, p = prob)
                
                center = np.zeros(3, dtype = 'float')
                for i in range(opt.pointnum):
                    x = mesh[idx[i]][1] - mesh[idx[i]][0]
                    y = mesh[idx[i]][2] - mesh[idx[i]][0]
                    c1 = np.random.random()
                    c2 = np.random.random() * (1-c1)
                    point[i] = mesh[idx[i]][0] + c1*x + c2*y
                    center = center + point[i]

                center = center / opt.pointnum
                max_axis = 0
                for i in range(opt.pointnum):
                    point[i] = point[i] - center
                    for j in range(3):
                        max_axis = max(max_axis, abs(point[i][j]))
                
                point = point / max_axis

                filename = os.path.splitext(item)[0]
                np.save(root + '/' + filename, point)

    else:
        current_class = dir
