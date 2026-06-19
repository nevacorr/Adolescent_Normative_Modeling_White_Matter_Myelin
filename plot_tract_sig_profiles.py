import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_tract_sig_profiles(struct_var, zmean, reject, working_dir, nsplits, roi_ids, gender):


    # Convert dictionary values to a NumPy array for plotting
    # data_matrix = np.array(list(reject.values()))
    data_matrix = reject.astype(int)
    N = data_matrix.shape[0]
    data_matrix = data_matrix.reshape(N // 60, 60)

    # Create a list of tracts
    tracts = []
    tract_data = {}

    for i, r in enumerate(roi_ids):
        prefix = r.split('_')[0] # Extract the tract name
        if prefix not in tracts:
            tracts.append(prefix)
            tract_data[prefix] = []

    for i, t in enumerate(tracts):
        tract_data[t].append(data_matrix[i]) #Add row to corresponding tract

    num_tracts = len(tracts)

    vmin = np.min(data_matrix)
    vmax = np.max(data_matrix)

    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1

    # Create the figure
    fig, axes = plt.subplots(num_tracts, 1, figsize=(8, 2 * num_tracts), constrained_layout=True)

    if num_tracts == 1:
        axes = [axes]

    for ax, tract in zip(axes,tracts):
        tract_matrix = np.array(tract_data[tract])
        cax = ax.imshow(tract_matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_title(f'{tract}  {struct_var} {gender}')
        ax.set_xlabel("Node")

        # Remove y-axis tick marks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])


    fig.colorbar(cax, ax=axes, orientation="vertical", label="Value")

    # Show plot
    plt.show()

    mystop=1
