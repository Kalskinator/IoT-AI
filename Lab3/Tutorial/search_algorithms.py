# Section1: Search Algorithms

# 2.1.Dreath-First Search (DFS)

# The algorithm starts at the root node (selecting some arbitrary node as
# the root node in the case of a graph) and explores as far as possible along each branch before backtracking.

# In this example, you will find a graph solved with DFS. Link of Description: Graph Example - DFSLinks to an external site.


# https://favtutor.com/blogs/depth-first-search-python
# Using a Python dictionary to act as an adjacency list
def dreath_first_search():
    graph = {"5": ["3", "7"], "3": ["2", "4"], "7": ["8"], "2": [], "4": ["8"], "8": []}

    visited = set()  # Set to keep track of visited nodes of graph.

    def dfs(visited, graph, node):  # function for dfs
        if node not in visited:
            print(node)
            visited.add(node)
            for neighbour in graph[node]:
                dfs(visited, graph, neighbour)

    # Driver Code
    print("Following is the Depth-First Search")
    dfs(visited, graph, "5")


# 2.1.Breath-First Search (BFS)

# BFS algorithm starts at the tree root and explores all nodes at
# the present depth prior to moving on to the nodes at the next depth level.
# Extra memory, usually a queue, is needed to keep track of the child nodes that were encountered but not yet explored.

# In this example, you will find a graph solved with BFS. Link of Description: Graph Example - BFSLinks to an external site.


# https://favtutor.com/blogs/breadth-first-search-python
def breath_first_search():
    graph = {"5": ["3", "7"], "3": ["2", "4"], "7": ["8"], "2": [], "4": ["8"], "8": []}

    visited = []  # List for visited nodes.
    queue = []  # Initialize a queue

    def bfs(visited, graph, node):  # function for BFS
        visited.append(node)
        queue.append(node)

        while queue:  # Creating loop to visit each node
            m = queue.pop(0)
            print(m, end=" ")

            for neighbour in graph[m]:
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)

    # Driver Code
    print("Following is the Breadth-First Search")
    bfs(visited, graph, "5")  # function calling


if __name__ == "__main__":
    # dreath_first_search()
    breath_first_search()
