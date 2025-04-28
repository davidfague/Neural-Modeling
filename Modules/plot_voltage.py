import matplotlib.pyplot as plt

def plot_voltage(sim_data, indices, colors, xlims=None, title_suffix="", save_file = None, show=False):
    colors = [color.split('*')[0] for color in colors]
    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 6))

        # plt.subplot(1, 1, 1)
        # plt.plot(sim_data['v'][time_points, idx], colors[i])
        plt.plot(sim_data['v'][:, idx], colors[i])
        plt.ylim([-80, 0])
        # plt.xlim(100000,110000) # reduced
        # plt.xlim(50000, 60000) # complex
        if xlims:
            plt.xlim(xlims)
        plt.axhline(y=-60, color=colors[i], linestyle='--')
        plt.title(f'Voltage at index {idx} {title_suffix}')
        
        # plt.subplot(1, 2, 2)
        # plt.plot(sim_data['v'][time_points, segment_mapping[idx]], colors[i])
        # plt.ylim([-90, 10])
        # plt.title(f'Refactored Model - Voltage at index {segment_mapping[idx]} {title_suffix}')
        if save_file:
            plt.savefig(f"{save_file}_{idx}.png", format='png', bbox_inches="tight", dpi=300)
        if show:
            plt.show()