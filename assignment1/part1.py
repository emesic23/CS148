import datasets
import matplotlib.pyplot as plt

# start with a totally balanced split in 2 dimensions
N = 100
M = 100
dims = 2

# Part 1 A
# TODO: run the following lines.
#       Comment on the resulting plots
dataset = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims)
dataset.save_dataset_plot()

# Part 1 B
# TODO: Increase the dimensionality to 3.
dims = 3
gaus_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims).get_dataset()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(gaus_dataset_points[:, 0], gaus_dataset_points[:, 1], gaus_dataset_points[:, 2], c=gaus_dataset_points[:, 3])
fig.savefig("figs/Gaus3d.png")

plt.show()
# Part 1 D
# TODO: go to random_control.py and set a random seed at your choosing. 