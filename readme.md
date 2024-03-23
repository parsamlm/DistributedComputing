<h1>Distributed Computing</h1>
This repository contains the code for the Distributed Computing course at the University of Genoa. The course is part of the
Master's degree in Computer Science.
<hr>
<h3><a href="https://github.com/parsamlm/DistributedComputing/blob/main/Report%20File.pdf">Report File</a></h3>
<h3>Assignment 1 - MMN</h3>
The MMN Queue is an integral part of our distributed computing project. It is designed to manage tasks in a
multi-server system, ensuring efficient distribution and execution of tasks. First of all, MM1 code is converted to
MMN one. MMN stands for Multi-Server Multi-Queue. The MMN Queue is a system that manages tasks in a multi-server. In this
project, we have implemented a strategy called supermarket strategy. The supermarket strategy is a strategy that assigns 
tasks to the server with the shortest queue. Also another strategy is implemented called the round-robin strategy. The
round-robin strategy is a strategy that assigns tasks to the servers in a round robin fashion.
<br>
The project has been implemented in Python, and has two main modules: 'mmn_queue.py' and 'discrete_event_sim.py'.
The 'mmn_queue.py' module contains the MMNQueue class, which is responsible for managing the tasks in the system. 
The 'discrete_event_sim.py' module contains the DiscreteEventSimulator class, which is responsible for simulating the 
system and generating the results.
<hr>
<h3>Assignment 2 - Erasure Coding</h3>
The Erasure Coding project is a distributed computing project that implements the erasure coding technique to 
ensure data reliability and fault tolerance. The project has two different configuration files: 'p2p.cfg' and
'client_server.cfg'. The 'p2p.cfg' file contains the configuration for the peer-to-peer network, while the 'client_server.cfg'
file contains the configuration for the client-server network. Another strategy is added to the project called selfish node
strategy. The selfish node strategy is a strategy that only prefer to upload its own data to the network.
The project has been implemented in Python, and has two main modules: 'storage.py' and 'discrete_event_sim.py'. 
