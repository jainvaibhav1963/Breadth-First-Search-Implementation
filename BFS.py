# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 07:07:26 2021

@author: jain
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

# initial and goal state
start = [0,0]
goal = [250,360]

#######################################################################################3
# defining obstacle space
maze = np.zeros((300,400))

for i in range(300):
    for j in range(400):
        # three rectangles
        if (i >= 20) and (i <=70) and (j >= 200) and (j <= 210):
            maze[i][j] = 1
        if (i >= 60) and (i <=70) and (j >= 210) and (j <= 230):
            maze[i][j] = 1
        if (i >= 20) and (i <=30) and (j >= 210) and (j <= 230):
            maze[i][j] = 1 
            # circle
        if ((j-90)**2+(i-(-70+300))**2 <= 35**2):
            maze[i][j] = 1
            # Ellipse
        if ((j-246)/60)**2 + ((i-155)/30)**2 <= 1 :
             maze[i][j] = 1
            #  inclined rectangle
        if (-0.7*j+1*(300-i))>=73.4 and ((300-i)+1.428*j)>=172.55 and (-0.7*j+1*(300-i))<=99.81 and ((300-i)+1.428*j)<=429:
            maze[i][j] = 1
            # polygon
        if ((300-i)+j>=391 and j-(300-i)<=265 and (300-i)+0.8*j<=425.7 and  (300-i)+0.17*j<=200 and 0.9*j -(300-i) >=148.7) or   (13.5*j+(300-i)<=5256.7 and 1.43*j-(300-i)>=369 and (300-i)+0.81*j >=425):    
            maze[i][j] = 1
            
######################################################################################333           

maze1 = maze.copy() 

img = maze.copy()
[row,col] = np.where(img == 1)

# making all obstacles black 
for b in range(len(row)):
    img[row[b]][col[b]] = 255
    
# movement functions (8)
def N(maze1):
    global img
    # s is source
    #check N, make eveything that is 0 - k, if it is 1 o anything else dont do anything
    if (maze1[p-1][q] == 0) and (p != 0):
        maze1[p-1][q] = k+1
        
        img[p-1][q] = 100
        
        return maze1
    
def S(maze1):
    global img
    
    if (p != 299) and (maze1[p+1][q] == 0):
        maze1[p+1][q] = k+1
        
        img[p+1][q] = 100
        
        return maze1    
    
def E(maze1):
    global img
    
    if (q != 399) and (maze1[p][q+1] == 0):
        maze1[p][q+1] = k+1
        
        img[p][q+1] = 100
        
        return maze1
    
def W(maze1):
    global img
    
    if (maze1[p][q-1] == 0) and (q != 0):
        maze1[p][q-1] = k+1 
        
        img[p][q-1] = 100
        
        return maze1

def NW(maze1):
    global img
    
    if (maze1[p-1][q-1] == 0) and (p != 0) and (q != 0):
        maze1[p-1][q-1] = k+1
        
        img[p-1][q-1] = 100
        
        return maze1

def NE(maze1):
    global img
    
    if (p != 0) and (q != 399) and (maze1[p-1][q+1] == 0):
        maze1[p-1][q+1] = k+1
        
        img[p-1][q+1] = 100
        
        return maze1

def SE(maze1):
    global img
    
    if (p != 299) and (q != 399) and (maze1[p+1][q+1] == 0):
        maze1[p+1][q+1] = k+1
        
        img[p+1][q+1] = 100
        
        return maze1

def SW(maze1):
    global img
    
    if (p != 299) and (maze1[p+1][q-1] == 0) and (q != 0):
        maze1[p+1][q-1] = k+1 
        
        img[p+1][q-1] = 100
        
        return maze1

######################################################################################

# make the start node to be 2
maze1[start[0]][start[1]] = 2

g = False
k = 1

#img_counter = 0
video = []
while (g == False):
    
    [m,n] = np.where(maze1 == k+1)
    k = k+1
    
    img1 = img.copy()
    video.append(img1)
    cv2.imwrite(str(k) + '.png',img1)
    for u in range(len(m)):
        p = m[u]
        q = n[u]
        
        # if goal reached
        if ([p,q] == goal):
            g = True
            break
        
        N(maze1)
        S(maze1)
        E(maze1)
        W(maze1)
        NW(maze1)
        NE(maze1)
        SE(maze1)
        SW(maze1)
        
###########################################################################################
# labeling the shortest path

maze2 = maze1.copy()
# shortest_path

# k = maze2[goal[0]][goal[1]]

# [p,q] = goal

# Same as movement functions but instead of k+1 it is k-1
i, j = goal
k = maze2[i][j]
the_path = [[i,j]]
while k > 2:
    if i > 0 and maze2[i - 1][j] == k-1:
        i, j = i-1, j
        the_path.append([i, j])
        k-=1
    elif j > 0 and maze2[i][j - 1] == k-1:
        i, j = i, j-1
        the_path.append([i, j])
        k-=1
    elif i < len(maze2) - 1 and maze2[i + 1][j] == k-1:
        i, j = i+1, j
        the_path.append([i, j])
        k-=1
    elif j < len(maze2[i]) - 1 and maze2[i][j + 1] == k-1:
        i, j = i, j+1
        the_path.append([i, j])
        k -= 1
    elif i > 0 and j > 0 and maze2[i-1][j-1] == k-1:
        i,j = i-1, j-1
        the_path.append([i, j])
        k -= 1
    elif i > 0 and j < len(maze2[i]) - 1 and maze2[i-1][j+1] == k-1:
        i,j = i-1,j+1
        the_path.append([i, j])
        k -= 1
    elif i < len(maze2) - 1 and j > 0 and maze2[i+1][j - 1] == k-1:
        i,j = i+1,j-1
        the_path.append([i, j])
        k -= 1      
        
    elif i < len(maze2) - 1 and j < len(maze2[i]) - 1 and maze2[i+1][j + 1] == k-1:
        i,j = i+1,j+1
        the_path.append([i, j])
        k -= 1      

# the_path is the coordinates for the shortest path
m = maze.copy()

for j in range(len(the_path)):
    for i in range(0,1):
        m[the_path[j][i]][the_path[j][i+1]] = 150

img1 = m.copy()
[row1,col1] = np.where(img1 == 1)

for b in range(len(row1)):
    img1[row1[b]][col1[b]] = 255
    
cv2.imwrite('Shortest_path.png',img1) 

plt.imshow(img1),plt.show()

###########################################################################################
# video maker

img=[]
for i in range(2,len(the_path)+2):
    img.append(cv2.imread(str(i)+'.png'))

height,width,layers=img[1].shape

img1 = cv2.imread('Shortest_path.png')
img.append(img1)
video=cv2.VideoWriter('video_BFS.mp4',-1,10,(width,height))

for j in range(len(the_path)+1):

    video.write(img[j])

cv2.destroyAllWindows()
video.release()

