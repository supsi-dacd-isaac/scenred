# scenred
Code for optimal scenario tree reduction for multivariate data. 
The code reduces a set of scenarios (obtained from sampling a multivariate distribution, or from historical data) to a scenario tree, in which each node has an associate probability, such as at each point in time, the sum of the probabilities in all the branches of the tree is equal to 1. 
The algorithm implemented is based on the one described in [1], which rely on a greedy strategy to optimally reduce the scenarios, based on the Kantorovich distance.

Example
=====
Test.m present an example, in which 791 bivariate scenarios are reduced using two different methods.
The first one is based on the d/d_0 ratio, where d is the Kantorovich distance after aggregating the scenarios, and d_0 is the distance between all the scenarios and a tree with only one scenario (as proposed in [1]).
The second one needs to directly specify the number of desired scenarios for each timestep.

Caveat
=====
The code is provided 'as is'.
This scenario reduction technique is thought to be used in (near) real-time control, in combination with chance constraints, or to reduce the computational time of stochastic control algorithms.
Do not use it to model tail risks, nor for design purposes, e.g. for choosing the nominal power of the backup diesel generators to avoid core meltdown of a nuclear reactor in case of national black-out. Use all the data in your possess for this task, and apply a 1.5 safety factor =)    

References
=====

 [[1]](https://www.mathematik.hu-berlin.de/~heitsch/ieee03ghr.pdf) *N. Gröwe-Kuska, H. Heitsch, and W. Römisch, “Scenario reduction and scenario tree construction for power management problems,” in 2003 IEEE Bologna PowerTech - Conference Proceedings, 2003, vol. 3, pp. 152–158.*
