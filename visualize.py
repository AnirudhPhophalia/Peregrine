import matplotlib.pyplot as plt


def plot_grid(x_grid):
    plt.figure(figsize=(12, 5))
    plt.imshow(
        x_grid,
        aspect="auto",
        cmap="hot",
        interpolation="nearest"
    )

    plt.xlabel("Time slot")
    plt.ylabel("Band index")
    plt.title(
        "RF Ground Truth Occupancy Grid"
    )
    plt.colorbar(
        label="Occupied (1) / Empty (0)"
    )
    plt.tight_layout()
    plt.show()