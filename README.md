# xNN Package

A package implementing the nearest neighbour with bandit feedback and hierarchical nearest neighbour algorithms from Pasteris et al., alongside navigating nets for faster nearest neighbour search, and KNN with UCB, KNN with KL UCB and Slivkin's contextual bandits with similarity information for benchmarking. Additionally, includes an implementation of the dynamic ternary tree rebalancing from Matsuzaki and Morihata.

Tested on python 3.8.10, 3.10.13, 3.12.1, 3.12.3, 3.12.7

Packages used:  
- numpy
- scipy (for the UCI firewall dataset)


TODOS:
- Check model saving and loading is working correctly.
- Maybe remove UCI firewall dataset and CICIDS2017 dataset (as they maybe shouldn't be in the package itself?)
- Clean up and check code documentation is all okay enough?
- Add paper references to this readme
- Set up licence and referencing stuff?
- Set up basic tutorial jupyter notebook potentially? showing how to use the bandits?
- look at combining/making compatible to use with the existing contextual bandit packages?
