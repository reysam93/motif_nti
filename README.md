# Motif-based Graph Learning
This repository includes the code associated with the paper "[Enhanced graph-learning schemes driven by similar distributions of motifs](https://arxiv.org/abs/2207.04747)", by Samuel Rey, T. Mitchell Roddenberry, Santiago Segarra, and Antonio G. Marques in the `master` branch.


## Abstract
This paper looks at the task of network topology inference, where the goal is to learn an unknown graph from nodal observations. One of the novelties of the approach put forth is the consideration of prior information about the density of motifs of the unknown graph to enhance the inference of classical Gaussian graphical models.
Dealing with the density of motifs directly constitutes a challenging combinatorial task.
However, we note that if two graphs have similar motif densities, one can show that the expected value of a polynomial applied to their empirical spectral distributions will be similar. Guided by this, we first assume that we have a reference graph that is related to the sought graph (in the sense of having similar motif densities) and then, we exploit this relation by incorporating a similarity constraint and a regularization term in the network topology inference optimization problem. The (non-)convexity of the optimization problem is discussed and a computational efficient alternating majorization-minimization algorithm is designed. We assess the performance of the proposed method through exhaustive numerical experiments where different constraints are considered and compared against popular baselines algorithms on both synthetic and real-world datasets.

## Organization of the repository
The organization of the paper is as follows:
* The notebooks in the root of the repository contain the different experiments presented in the paper.
* `src`: contains the code that defines the graph learning algorithm based on motif similarity, and also the implementation of the different baselines algorithms used for comparison.
* `utils`: contains different utility scripts such as the ones used for processing the real data or estimating the hyperparameters of the different methods. 
* `data`: contains the data already processed used in the experiment based on real-world data. The reference to obtain the original data can be found in the paper.  
