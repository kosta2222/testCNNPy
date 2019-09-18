size_axis=(3,3) # количество осей,матрица
center_w_l=(1,1) # положение центра ядра


def create_axis_indexes(size_axis:int,center_w_l:int)->list:
    coordinates=[]
    for i in range(-center_w_l,size_axis-center_w_l): # range(-1,3-1)=range(-1,2)
        coordinates.append(i)

    return coordinates

def create_indexes(size_axis:tuple,center_w_l:tuple)->tuple:
    coords_a=create_axis_indexes(size_axis[0],center_w_l[0])
    coords_b=create_axis_indexes(size_axis[1],center_w_l[1])

    return (coords_a,coords_b)


print(create_indexes(size_axis,center_w_l))
print(create_indexes((3,3),(0,0)))
