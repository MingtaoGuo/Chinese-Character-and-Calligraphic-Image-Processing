import matplotlib.pyplot as plt
import numpy as np




if __name__ == "__main__":
    loss = np.loadtxt("../data/loss_list.txt")
    acc = np.loadtxt("../data/acc_list.txt")
    plt.plot(np.arange(0, acc.shape[0]), acc)
    plt.xlabel("Iteration (x20)")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(np.arange(0, loss.shape[0]), loss)
    plt.xlabel("Iteration (x20)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()