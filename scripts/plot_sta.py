import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback
import gc  # for garbage collection


def _plot_sta(
    sta,
    quantiles,
    title,
    xlabel_spike_type,
    ylabel_ed_from,
    cbar_spike_type,
    limit=True,
    time_bounds_ms=(-50, 50),
) -> plt.figure:
    """
    Plot a binned STA matrix with exactly 5 x-ticks at -50, -25, 0, 25, 50 ms.

    Parameters
    ----------
    sta : np.ndarray
        2D matrix [bins -> time] after binning to quantiles (rows).
    quantiles : array-like
        Quantile boundaries used for binning (for y tick labels).
    title, xlabel_spike_type, ylabel_ed_from, cbar_spike_type : str
        Labels.
    limit : bool
        If True, color scale is limited to mean ± 0.99*std.
    time_bounds_ms : tuple(float, float)
        Time window in ms used to *label* the x-axis (must match data window).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Image with optional clipping
    if limit:
        mean_val = float(np.mean(sta))
        std_val = float(np.std(sta))
        lower_limit = mean_val - 0.99 * std_val
        upper_limit = mean_val + 0.99 * std_val
        img = ax.imshow(
            sta,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            vmin=lower_limit,
            vmax=upper_limit,
            aspect="auto",
            interpolation="nearest",
        )
    else:
        img = ax.imshow(
            sta,
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            aspect="auto",
            interpolation="nearest",
        )

    ax.set_title(title)

    # --- Five fixed labels at -50, -25, 0, 25, 50 ms ---
    n_bins = sta.shape[1]
    tmin, tmax = time_bounds_ms
    desired_tick_labels = np.array([-50, -25, 0, 25, 50], dtype=int)

    # Map desired ms -> bin indices (positions) robustly to current width
    time_axis = np.linspace(tmin, tmax, n_bins)           # ms per bin
    bin_axis = np.arange(n_bins, dtype=float)             # 0..n_bins-1
    tick_positions = np.interp(desired_tick_labels, time_axis, bin_axis)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(x) for x in desired_tick_labels])
    ax.set_xlabel(f"Time from {xlabel_spike_type} spikes (ms)")

    # y-axis: quantiles
    ax.set_yticks(ticks=np.arange(11) - 0.5)
    ax.set_yticklabels(np.round(quantiles, 3))
    ax.set_ylabel(f"transfer impendance ratio from {ylabel_ed_from}")

    fig.colorbar(img, ax=ax, shrink=0.5, label=r"$\Delta\mu$% " + cbar_spike_type + " spike rate")
    return fig


def _compute_sta_for_each_train_in_a_list(
    list_of_trains,
    spikes,
    win_length=None,
    parameters=None,
    time_bounds_ms=(-50, 50),
):
    """
    Compute STA for each train in list_of_trains with a window derived from ms bounds.

    - If win_length is None, it is computed from time_bounds_ms and parameters.h_dt:
        win_length (bins) = 2 * round((time_bounds_ms[1]-time_bounds_ms[0])/2 / h_dt)

    Returns
    -------
    stas : np.ndarray
        Concatenated STA rows with shape [n_trains, win_length]
    """
    if parameters is None:
        raise ValueError("parameters must be provided")

    # Derive win_length from ms if not provided (symmetric window)
    if win_length is None:
        half_window_ms = (time_bounds_ms[1] - time_bounds_ms[0]) / 2.0
        half_window_samples = int(round(half_window_ms / parameters.h_dt))
        win_length = max(2 * half_window_samples, 1)

    stas = []
    total_steps = int(parameters.h_tstop / parameters.h_dt)

    for train in list_of_trains:
        if len(train) == 0:
            stas.append(np.zeros((1, win_length), dtype=np.float32))
            continue

        cont_train = np.zeros(total_steps, dtype=np.float32)
        cont_train[train] = 1.0
        # remove warmup / initial steps if desired
        cont_train[: parameters.skip] = 0.0

        sta = analysis.SummaryStatistics.spike_triggered_average(
            cont_train.reshape((1, -1)), spikes, win_length
        )
        # Normalize to percent change relative to mean rate (avoid division by zero)
        sta = (sta - np.mean(cont_train)) / (np.mean(cont_train) + 1e-15) * 100.0
        stas.append(sta.astype(np.float32))

    # Ensure 2D
    stas = [arr.reshape(1, -1) if arr.ndim == 1 else arr for arr in stas]
    stas = np.concatenate(stas, axis=0)
    return stas


def _map_stas_to_quantiles_and_plot(
    sta,
    spikes,
    section,
    elec_dist_from,
    title,
    xlabel_spike_type,
    cbar_spike_type,
    sim_directory,
    save,
    save_directory,
    indexes=None,
    time_bounds_ms=(-50, 50),
):
    if section not in ["apic", "dend"]:
        raise ValueError("section must be 'apic' or 'dend'")
    if elec_dist_from not in ["soma", "nexus"]:
        raise ValueError("elec_dist_from must be 'soma' or 'nexus'")

    elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{elec_dist_from}.csv"))
    morph = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    if indexes is not None:
        elec_dist = elec_dist.iloc[indexes, :]
        morph = morph.iloc[indexes, :]

    quantiles = analysis.SummaryStatistics.get_quantiles_based_on_elec_dist(
        morph=morph, elec_dist=elec_dist, spikes=spikes, section=section
    )
    sta_binned = analysis.SummaryStatistics.bin_matrix_to_quantiles(
        matrix=sta, quantiles=quantiles, var_to_bin=elec_dist
    )

    fig = _plot_sta(
        sta_binned,
        quantiles,
        title,
        xlabel_spike_type,
        elec_dist_from,
        cbar_spike_type,
        limit=True,
        time_bounds_ms=time_bounds_ms,
    )
    plt.show()
    if save:
        outpath = os.path.join(
            save_directory, f"{section}_{elec_dist_from}_{cbar_spike_type}_wrt_{xlabel_spike_type}.png"
        )
        fig.savefig(outpath, dpi=fig.dpi, bbox_inches="tight")
    plt.close(fig)  # Release memory


def _analyze_spike_relationships(
    sim_directory,
    spike_type,
    wrt_spike_type,
    section,
    elec_dist,
    parameters,
    save,
    save_directory,
):
    # You can adjust this in one place and it will propagate everywhere:
    time_bounds_ms = (-50, 50)

    # Load arrays (downcast to float32 to save memory)
    v = analysis.DataReader.read_data(sim_directory, "v").astype(np.float32)
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    ica = analysis.DataReader.read_data(sim_directory, "ica").astype(np.float32)
    if parameters.exc_syn_mod == "pyr2pyr":
        inmda = analysis.DataReader.read_data(sim_directory, "inmda").astype(np.float32)
    else:
        inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA").astype(np.float32)
    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    indexes = seg_data[seg_data["section"] == section].index

    try:
        # spike collection (triggered)
        if spike_type == "Na":
            spikes = []
            for i in range(len(v)):
                spike_times, _, _ = analysis.VoltageTrace.get_Na_spikes(
                    v[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0]
                )
                spikes.append(spike_times)
        elif spike_type == "Ca":
            spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                spikes.append(left_bounds)
        elif spike_type == "NMDA":
            spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                spikes.append(left_bounds)
        elif spike_type == "soma_spikes":
            spikes = soma_spikes
        else:
            raise ValueError("Invalid spike type")

        # with-respect-to spikes (reference)
        if wrt_spike_type == "soma_spikes":
            wrt_spikes = soma_spikes
        elif wrt_spike_type == "Na":
            wrt_spikes = []
            for i in range(len(v)):
                spike_times, _, _ = analysis.VoltageTrace.get_Na_spikes(
                    v[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0]
                )
                wrt_spikes.extend(spike_times)
            wrt_spikes = np.sort(np.unique(wrt_spikes))
        elif wrt_spike_type == "Ca":
            wrt_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                wrt_spikes.extend(left_bounds)
            wrt_spikes = np.sort(np.unique(wrt_spikes))
        elif wrt_spike_type == "NMDA":
            wrt_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                wrt_spikes.extend(left_bounds)
            wrt_spikes = np.sort(np.unique(wrt_spikes))
        else:
            raise ValueError("Invalid 'with respect to' spike type")

        print(f"sim_directory: {sim_directory}")

        # compute STA window from ms bounds + parameters.h_dt
        sta = _compute_sta_for_each_train_in_a_list(
            list_of_trains=spikes,
            spikes=wrt_spikes,
            parameters=parameters,
            time_bounds_ms=time_bounds_ms,
            win_length=None,  # derive from ms
        )

        _map_stas_to_quantiles_and_plot(
            sta=sta,
            spikes=spikes,
            section=section,
            elec_dist_from=elec_dist,
            title=f"{section} {spike_type} spike rate w.r.t. {wrt_spike_type} spiketimes",
            xlabel_spike_type=wrt_spike_type,
            cbar_spike_type=spike_type,
            sim_directory=sim_directory,
            save=save,
            save_directory=save_directory,
            indexes=indexes,
            time_bounds_ms=time_bounds_ms,
        )
        print(
            f"SUCCESS analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:"
        )
    except Exception as e:
        print(
            f"Error analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:"
        )
        print(traceback.format_exc())
        print(str(e))
    finally:
        # Release large arrays
        del v, ica, inmda, seg_data
        gc.collect()


def analyze_all_spike_relationships(sim_directory, parameters, save, save_directory):
    sections = ["apic", "dend"]
    spike_types = ["Na", "Ca", "NMDA", "soma_spikes"]
    elec_dists = ["soma", "nexus"]
    for spike_type in spike_types:
        for wrt_spike_type in spike_types:
            for section in sections:
                for elec_dist in elec_dists:
                    if (elec_dist == "nexus") and ((spike_type != "Ca") or (wrt_spike_type != "Ca")):
                        continue
                    if (section == "dend") and ((spike_type == "Ca") or (wrt_spike_type == "Ca")):
                        continue
                    if spike_type == "soma_spikes":
                        continue

                    _analyze_spike_relationships(
                        sim_directory,
                        spike_type,
                        wrt_spike_type,
                        section,
                        elec_dist,
                        parameters,
                        save,
                        save_directory,
                    )


if __name__ == "__main__":
    save = "-s" in sys.argv
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
        if save:
            save_directory = os.path.join(sim_directory, "STAs")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        else:
            save_directory = None

        logger = Logger()
        soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
        parameters = analysis.DataReader.load_parameters(sim_directory)
        logger.log(f"Soma firing rate: {round(soma_spikes.shape[1] * 1000 / parameters.h_tstop, 2)} Hz")
        del soma_spikes  # Release memory
        gc.collect()

        if os.path.exists(os.path.join(sim_directory, "parameters.pickle")):
            try:
                logger.log("Analyzing all spike relationships.")
                analyze_all_spike_relationships(sim_directory, parameters, save, save_directory)
            except Exception:
                print(traceback.format_exc())
        else:
            print(f"Skipping {sim_directory}. No parameters.pickle found.")

    elif "-f" in sys.argv:
        simulations_dir = sys.argv[sys.argv.index("-f") + 1]
        for sim_folder in os.listdir(simulations_dir):
            sim_directory = os.path.join(simulations_dir, sim_folder)
            if not os.path.exists(os.path.join(sim_directory, "parameters.pickle")):
                print(f"Skipping {sim_directory}. No parameters.pickle found.")
                continue

            print(f"Analyzing {sim_directory}")
            if save:
                save_directory = os.path.join(sim_directory, "STAs")
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
            else:
                save_directory = None

            logger = Logger()
            soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
            parameters = analysis.DataReader.load_parameters(sim_directory)
            logger.log(f"Soma firing rate: {round(soma_spikes.shape[1] * 1000 / parameters.h_tstop, 2)} Hz")
            del soma_spikes
            gc.collect()

            if os.path.exists(os.path.join(sim_directory, "parameters.pickle")):
                try:
                    logger.log(f"Analyzing all spike relationships for {sim_directory}.")
                    analyze_all_spike_relationships(sim_directory, parameters, save, save_directory)
                except Exception:
                    print(traceback.format_exc())
            else:
                print(f"Skipping {sim_directory}. No parameters.pickle found.")
    else:
        raise RuntimeError
