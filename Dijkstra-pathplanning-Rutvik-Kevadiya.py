#!/usr/bin/env python3

import numpy as np
import cv2
import heapq as hq
from time import time
import moviepy.video.io.ImageSequenceClip as ISC

# The function below returns True if (x,y) is in circle
def in_circle(x,y):
    xc = 300
    yc = 185
    r = 45                       
    if (x - xc)*(x - xc) + (y - yc)*(y - yc) <= r*r:
        return True
    else:
        return False

# The function below returns True if (x,y) is in hexagon
def in_hexagon(x,y):
    if x <= 235 + 5 and x >= 165 - 5:
        if y <= 0.577*x + 30.712 and y >= 0.577*x - 61.66:
            if y <= -0.577*x + 261.662 and y >= -0.577*x + 169.277:
                return True
            else:
                return False
        else:
            return False
    else:
        return False

# The function below returns True if (x,y) is in quadrilateral
def in_quadrilateral(x,y):
    if y <= 0.316*x + 178.90:
        if y >= -1.231*x + 221.325:
            if y >= 0.857*x + 104.855 or (y <= 0.857*x + 104.855 and y <= -3.2*x + 452.76):
                return True
            else:
                return False
        else:
            return False
    else:
        return False

# The function below returns True if (x,y) is in obstacle space
def in_obstacle(x,y):
    if in_circle(x,y) or in_quadrilateral(x,y) or in_hexagon(x,y):
        return True
    else:
        return False

# The function returns new node state if moving left is possible
def ActionMoveLeft(Node_State_i):
    if Node_State_i[0] != 0:
        if not in_obstacle(Node_State_i[0] - 1,Node_State_i[1]):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[0] = New_Node_State_i[0] - 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving up-left is possible
def ActionMoveUpLeft(Node_State_i):
    if Node_State_i[0] != 0 and Node_State_i[1] != 249:
        if not in_obstacle(Node_State_i[0] - 1,Node_State_i[1] + 1):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[0] = New_Node_State_i[0] - 1
            New_Node_State_i[1] = New_Node_State_i[1] + 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving up is possible
def ActionMoveUp(Node_State_i):
    if Node_State_i[1] != 249:
        if not in_obstacle(Node_State_i[0],Node_State_i[1] + 1):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[1] = New_Node_State_i[1] + 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving up-right is possible
def ActionMoveUpRight(Node_State_i):
    if Node_State_i[0] != 399 and Node_State_i[1] != 249:
        if not in_obstacle(Node_State_i[0] + 1,Node_State_i[1] + 1):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[0] = New_Node_State_i[0] + 1
            New_Node_State_i[1] = New_Node_State_i[1] + 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving right is possible
def ActionMoveRight(Node_State_i):
    if Node_State_i[0] != 399:
        if not in_obstacle(Node_State_i[0] + 1,Node_State_i[1]):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[0] = New_Node_State_i[0] + 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving down-right is possible
def ActionMoveDownRight(Node_State_i):
    if Node_State_i[0] != 399 and Node_State_i[1] != 0:
        if not in_obstacle(Node_State_i[0] + 1,Node_State_i[1] - 1):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[0] = New_Node_State_i[0] + 1
            New_Node_State_i[1] = New_Node_State_i[1] - 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving down is possible
def ActionMoveDown(Node_State_i):
    if Node_State_i[1] != 0:
        if not in_obstacle(Node_State_i[0],Node_State_i[1] - 1):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[1] = New_Node_State_i[1] - 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# The function returns new node state if moving down-left is possible
def ActionMoveDownLeft(Node_State_i):
    if Node_State_i[0] != 0 and Node_State_i[1] != 0:
        if not in_obstacle(Node_State_i[0] - 1,Node_State_i[1] - 1):
            New_Node_State_i = np.copy(Node_State_i)
            New_Node_State_i[0] = New_Node_State_i[0] - 1
            New_Node_State_i[1] = New_Node_State_i[1] - 1
            return True,(New_Node_State_i[0],New_Node_State_i[1])
        else:
            return False,False
    else:
        return False,False

# This function returns the index of a given node in open list
def find_index(Node_State_i,open_list):
    for i in range(len(open_list)):
        if open_list[i][1] == Node_State_i:
            return i
    return False

# This function generates the path from initial position to goal position using back tracking
def generate_path(closed_list,path_dict):
    i = closed_list[-1][3]
    node_path = [closed_list[-1][1]]
    while i > 0:
        node_path.append(path_dict[i][0])
        i = path_dict[i][1]
    node_path.reverse()
    return node_path


while True:
    initial_x = int(input("Enter x coordinate of initial node in the range of 0-399: "))
    initial_y = int(input("Enter y coordinate of initial node in the range of 0-249: ")) 
    if in_obstacle(initial_x,initial_y):
        print("The initial coordinates entered are in obstacle space. Please enter again.")
        continue
    else:
        while True:
            goal_x = int(input("Enter x coordinate of goal node in the range of 0-399: "))
            goal_y = int(input("Enter y coordinate of goal node in the range of 0-249: "))
            if in_obstacle(goal_x,goal_y):
                print("The goal coordinates entered are in obstacle space. Please enter again.")
                continue
            else:
                break
        break    

initial_node = (initial_x,initial_y)
goal_node = (goal_x,goal_y)

open_list = [[0,initial_node,1,0]]
open_list_set = {initial_node}
hq.heapify(open_list)

closed_list = []
closed_list_set = set()
node_num = 1            # Variable used for Node index
path_dict = {}
viz_matrix = np.zeros((400,250,3),np.uint8)  # Visualization Matrix
nodes = [[initial_node]]    # List used to store the nodes explored in each iteration


for i in range(400):
    for j in range(250):
        if in_obstacle(i,j):
            viz_matrix[i][j] = (255,0,0)

while True:
    Curr_Node = hq.heappop(open_list)
    Node_State_i = Curr_Node[1]
    open_list_set.remove(Node_State_i)
    Node_Index_i = Curr_Node[2]
    Node_c2c_i = Curr_Node[0]
    Node_Parent_Index_i = Curr_Node[3]
    Curr_Nodes = []             # List to keep track of nodes explored in the current iteration

    if Node_State_i == goal_node:
        closed_list.append(Curr_Node)
        closed_list_set.add(Node_State_i)
        path_dict[Node_Index_i] = [Node_State_i,Node_Parent_Index_i]
        break

    possible,New_Node_State_i = ActionMoveLeft(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1,1),New_Node_State_i,node_num,Node_Index_i])
            hq.heapify(open_list)
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    possible,New_Node_State_i = ActionMoveUpLeft(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1.4 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1.4,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1.4,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)

    possible,New_Node_State_i = ActionMoveUp(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    possible,New_Node_State_i = ActionMoveUpRight(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1.4 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1.4,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1.4,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    possible,New_Node_State_i = ActionMoveRight(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    possible,New_Node_State_i = ActionMoveDownRight(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1.4 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1.4,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1.4,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    possible,New_Node_State_i = ActionMoveDown(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    possible,New_Node_State_i = ActionMoveDownLeft(Node_State_i)
    if possible and New_Node_State_i not in closed_list_set:
        if New_Node_State_i in open_list_set:
            idx = find_index(New_Node_State_i,open_list)
            if Node_c2c_i + 1.4 < open_list[idx][0]:
                open_list[idx][3] = Node_Index_i
                open_list[idx][0] = round(Node_c2c_i + 1.4,1)
                hq.heapify(open_list)
        else:
            node_num = node_num + 1
            hq.heappush(open_list,[round(Node_c2c_i + 1.4,1),New_Node_State_i,node_num,Node_Index_i])
            open_list_set.add(New_Node_State_i)
            Curr_Nodes.append(New_Node_State_i)
    
    nodes.append(Curr_Nodes)
    
    if len(open_list) == 0:
        print("No Solution Found")
        break
    else:
        closed_list.append(Curr_Node)
        closed_list_set.add(Node_State_i)
        path_dict[Node_Index_i] = [Node_State_i,Node_Parent_Index_i]

node_path = generate_path(closed_list,path_dict)

cv2.imshow('visualization',cv2.flip(cv2.transpose(viz_matrix),0))
cv2.waitKey(0)

for i in nodes:
    for j in i:
       viz_matrix[j[0]][j[1]] = (0,0,255)

    cv2.imshow('visualization',cv2.flip(cv2.transpose(viz_matrix),0))  # Exploration
    cv2.waitKey(1)

for i in node_path:
    viz_matrix[i[0]][i[1]] = (0,255,0)

cv2.imshow('visualization',cv2.flip(cv2.transpose(viz_matrix),0))       # Final Path
cv2.waitKey(0)