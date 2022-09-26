# Dijkstra Shortest Path

## Introduction

One big and very important class of problems consists on finding paths between two given points. These types of problems are applicable on Google Maps, sat-nav systems and so forth. But for these systems, it is not good enough only finding a possible path that connect the two points. Usually we are looking for the shortest possible path between them. Dijkstra's Algorithm is one of the most common methods of finding such paths.
This script aims to find the shortest path between any two cities in a particular map. As an example, a graph representing the map of cities in Romania will be used.
The implementation is inspired by Computerphile's video about this subject, found here: youtube.com/watch?v=GazC3A4OQTE

## Map Model - Romanian Cities Graph

Given the following graph representing a map of cities from Romania, we will look for the shortest distance and the path we must travel to reach any given destination from any given starting point.

![romanian_cities_graph](https://user-images.githubusercontent.com/68711010/192386092-394f7405-e510-4add-82fb-fc64c173a7fc.png)

## Dijkstra's Algorithm

Dijkstra's Algorithm is a method of finding the shortest path between any two points, or in this case, two locations on a map. This algorithm requires a planar graph to model the map we desire to use. Each vertex represents a city or a location. The weights on the edges of the graph represents the distances between the vertices. 
This implementation uses:

#### **Dijkstra Nodes** 

Represents each vertex and stores additional information about:

- The total distance from the starting vertex until the current vertex;

- The previous vertex of the current route.

#### **Stack**

The stack data structure is used to store all nodes that were previously visited.

#### **Priority Queue**

Represents the sequence of vertices the algorithm will analyze and calculate the path distances to them. It is ordered from closest vertex to farthest.

### Steps

We choose the starting vertex which represents where our route will start from. 
The path distance for that node will be 0 and we add it to the Priority Queue.
Until the PQ is empty OR the first node is the destination we are looking for: get the first node of the queue, iterate through every edge of that node, calculate their distances and add them to the PQ.
After we reach the destination node, we return to the user the set of vertices composing the path we must take to reach our destination, and the total distance we need to travel to get there.

