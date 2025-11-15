import matplotlib.pyplot as plt


def plot_convergence(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history, label="PSO Objective")
    plt.title("PSO Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.legend()
    plt.show()
