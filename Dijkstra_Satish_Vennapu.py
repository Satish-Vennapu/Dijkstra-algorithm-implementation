#Importing libraries

import numpy as np
import cv2 as cv
import time
import heapq

#######################################################################################################################
# GitHub link:  https://github.com/Satish-Vennapu/Dijkstra-algorithm-implementation.git
################################################################################################################################################
#Defining the obstacles in the arena
# The robot avoids the start and goal nodes in this space

def obstacle_space(arena):
    height, width, _ = arena.shape
    for x in range(width):
        for y in range(height):
            # Clearance of 5 around the boundaries
            if (x-5<=0) or (x-595>=0) or (y-5<=0) or (y-245>=0):
                arena[y][x] = [0,0,255]
            # Rectangle of size 50 by 100 with clearance of 5
            if ((x>=95) and (x<=155)) and ((y>=0) and (y<=105)):
                arena[y][x] = [0, 255, 255] 
            # Rectangle of size 50 by 100 with clearance of 5
            if ((x>=95) and (x<=155)) and ((y>=145) and (y<=255)):
                arena[y][x] = [255, 255, 0]
            # Hexagon with clearance of 5
            if (y+(0.57*x)-218.53)>=0 and (-y+(0.57*x)+32.60)>=0 and (x-230)>=0 and (y+(0.57*x)-378.62)<=0 and (-y+(0.57*x)-129)<=0 and (x-370)<=0:
                arena[y][x] = [255,0,255]
            # Triangle with clearance of 5
            if ((y+(2*x)-1143)<=0 and (-y+(2*x)-895)<=0 and (x-455)>=0):
                arena[y][x] = [255,150,0]
    return arena

###############################################################################################################################################
# Function for taking the start node and goal node inputs from user

def user_inputs(arena):
    start_node = []
    goal_node = []
    while True:
        while True:
            node_x = int(input("Enter the X Co-ordinate of the start node: "))
            # checking for validity of the entered start node if it is out of the arena
            if (node_x<0 or node_x>arena.shape[1]-1):
                print("Invalid x co-ordinate of the start point!!")
                continue
            else:
                start_node.append(node_x)
                break
        while True:
            node_y = int(input("Enter the Y Co-ordinate of the start node: "))
            # checking for validity of the entered start node if it is out of the arena
            if (node_y<0 or node_y>arena.shape[0]-1):
                print("Invalid y co-ordinate of the start point!!")
                continue
            else:
                start_node.append(node_y)
                break
        # checking if the start node is in the obstacle space
        if (arena[arena.shape[0]-1 - start_node[1]][start_node[0]][0]==255):
            print("The given start node is in the obstacle space")
            start_node.clear()
        else:
            break
    
    while True:
        while True:
            node_x = int(input("Enter the X Co-ordinate of the goal node: "))
            # checking for validity of the entered goal node if it is out of the arena
            if (node_x<0 or node_x>arena.shape[1]-1):
                print("Invalid x co-ordinate of the goal node!!")
                continue
            else:
                goal_node.append(node_x)
                break
        while True:
            node_y = int(input("Enter the Y Co-ordinate of the goal node: "))
            # checking for validity of the entered goal node if it is out of the arena
            if (node_y<0 or node_y>arena.shape[0]-1):
                print("Invalid y co-ordinate of the goal node!!")
                continue
            else:
                goal_node.append(node_y)
                break
        # checking if the goal node is in the obstacle space
        if (arena[arena.shape[0]-1 - goal_node[1]][goal_node[0]][0]==255):
            print("The given goal node is in the obstacle space")
            goal_node.clear()
        else:
            break
    # if the start and goal node are valid, return start node and goal node
    return start_node, goal_node

################################################################################################################################################
# Defining the action moves for the algorithm

# Function for moving left(west) if it is a possible move
def west(node, arena):
    k,l = np.copy(node)
    if (k-1>0) and (arena[l][k-1][0]<255) :
        k = k-1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving right(east) if it is a possible move  
def east(node, arena):
    k,l = np.copy(node)
    if (k+1<arena.shape[1]) and (arena[l][k+1][0]<255) :
        k = k+1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving up(north) if it is a possible move  
def north(node, arena):
    k,l = np.copy(node)
    if (l-1>0) and (arena[l-1][k][0]<255) :
        l = l-1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving down(south) if it is a possible move  
def south(node, arena):
    k,l = np.copy(node)
    if (l+1<arena.shape[0]) and (arena[l+1][k][0]<255) :
        l = l+1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving top_left(north_west) if it is a possible move  
def north_west(node, arena):
    k,l = np.copy(node)
    if (l-1>0) and (k-1>0) and (arena[l-1][k-1][0]<255) :
        l = l-1
        k = k-1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving bottom_left(south_west) if it is a possible move  
def south_west(node, arena):
    k,l = np.copy(node)
    if (l+1<arena.shape[0]) and (k-1>0) and (arena[l+1][k-1][0]<255) :
        l = l+1
        k = k-1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving top_right(north_east) if it is a possible move  
def north_east(node, arena):
    k,l = np.copy(node)
    if (l-1>0) and (k+1<arena.shape[1]) and (arena[l-1][k+1][0]<255) :
        l = l-1
        k = k+1
        return True, (k,l)
    else:
        return False, (k,l)
# Function for moving bottom_right(south_east) if it is a possible move  
def south_east(node, arena):
    k,l = np.copy(node)
    if (l+1<arena.shape[0]) and (k+1<arena.shape[1]) and (arena[l+1][k+1][0]<255) :
        l = l+1
        k = k+1
        return True, (k,l)
    else:
        return False, (k,l)

##############################################################################################################################################
#Defining the working of Dijkstra algorithm

def algorithm(start_node, goal_node, arena):
    # open list contains cost to come, parent node (for back tracking), child node (updating current node)
    open_lst = []
    # closed dictionary contains 
    closed_dict = {}
    # boolean set to false and triggers when back tracking starts
    start_back_tracking = False
    # Converting the open_lst to heap
    heapq.heapify(open_lst)
    heapq.heappush(open_lst, [0, start_node, start_node])
    while(len(open_lst)>0):
        node = heapq.heappop(open_lst)
        closed_dict[(node[2][0], node[2][1])] =node[1]
        cost_to_come = node[0] 
        # Checking if the current node is the goal node
        if list(node[2]) == goal_node: 
            # Setting the back tracking boolean to true
            start_back_tracking = True
            print("Reached the goal node and Starting Back Tracking")
            break 

        temp,current_node = north(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1)<open_lst[k][0]): 
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1
                            heapq.heapify(open_lst)
                        break
                if(not catch): 
                    heapq.heappush(open_lst,[cost_to_come+1, node[2], list(current_node)])
                    heapq.heapify(open_lst)

        temp,current_node = south(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                    
        
        temp,current_node = east(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                    
        
        temp,current_node = west(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                   


        temp,current_node = north_east(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1.4)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1.4
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1.4, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                   
                
    
        temp,current_node = south_east(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1.4)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1.4
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1.4, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                    
        
        
        temp,current_node = south_west(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1.4)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1.4
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1.4, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                    
        
        
        temp,current_node = north_west(node[2], arena)
        if(temp):
            if current_node not in closed_dict:
                catch = False
                for k in range(len(open_lst)):
                    if(open_lst[k][2] == list(current_node)):
                        catch = True
                        # if cost to come is less than the present, cost and node is updated
                        if((cost_to_come+1.4)<open_lst[k][0]):
                            open_lst[k][1] = node[2]
                            open_lst[k][0] = cost_to_come+1.4
                            heapq.heapify(open_lst)
                        break
                if(not catch):
                    heapq.heappush(open_lst,[cost_to_come+1.4, node[2], list(current_node)])
                    heapq.heapify(open_lst)
                    
        
        heapq.heapify(open_lst)

    if(start_back_tracking):
        #Triggering the back tracking function
        back_tracking(start_node, goal_node, closed_dict, arena)
    else:
        print("Path can not be calculated!!")

####################################################################################################################
# Defining back tracking after the exploring phase

def back_tracking(start_node, goal_node, closed_dict, arena):
    cc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter('Dijkstra_Satish_Vennapu.avi',cc,1000,(arena.shape[1],arena.shape[0]))
    # All the nodes during exploration
    explored_nodes = closed_dict.keys() 
    # container to record the path
    optimal_path = [] 
    # For visualizing the explored nodes
    for explored_node in explored_nodes:
        # Explored nodes are painted green
        arena[explored_node[1]][explored_node[0]] = [0,255,0] 
        cv.imshow("Visualization of Exploration of Nodes", arena)
        cv.waitKey(1)
        output.write(arena)
    parent_node = closed_dict[tuple(goal_node)]
    # Adding the goal node to optimal path 
    optimal_path.append(goal_node) 
    while(parent_node!= start_node):
        # Adding the parent_node to the optimal path
        optimal_path.append(parent_node)
        parent_node = closed_dict[tuple(parent_node)]
    # Highlighting the start node and goal node
    cv.circle(arena,(start_node),radius=5,color=(255,0,0),thickness=-1)
    cv.circle(arena,(goal_node),radius=5,color=(0,0,255),thickness=-1)
    # Adding the start node to the optimal path
    optimal_path.append(start_node) 
    while(len(optimal_path)>0):
        # Highlighting the optimal path 
        path_node = optimal_path.pop()
        arena[path_node[1]][path_node[0]] = [0,0,255]
        output.write(arena)
    # Visualizing the nodes explored on arena
    cv.imshow("Visualization of Exploration of Nodes", arena)
    output.release()

###################################################################################################################################################
# starting of timer for program execution
start_timer = time.time() 
#Creating a black empty arena
arena = np.zeros((250,600,3),dtype="uint8") 
# putting obstacles in the empty arena
arena = obstacle_space(arena)
# User defined Start and goal node 
start_node, goal_node = user_inputs(arena) 
#Shifting the origin from top left to bottom left
start_node[1] = arena.shape[0]-1 - start_node[1]
goal_node[1] = arena.shape[0]-1 - goal_node[1]
# running the Dijkstra 
algorithm(start_node, goal_node, arena) 
# Calculating the time taken for find the optimal path
end_timer = time.time() 
cv.waitKey(0) 
# Closes all the opencv windows
cv.destroyAllWindows() 
print("Total time for Code Execution: ", end_timer-start_timer)
############################################################################################################################################