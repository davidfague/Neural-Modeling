import pandas as pd
import numpy as np
from Modules.spike_generator import PoissonTrainGenerator
import os
from functools import partial
import pickle
from Modules.logger import Logger

# for build_synapses_onto_cell_obj #TODO: move build_synapses_onto_cell_obj to cell_builder
import synapse
from Modules.synapse import Synapse
from Modules.cell_model import CellModel
from Modules.cell_builder import CellBuilder, SkeletonCell
from Modules.logger import Logger
import numpy as np
from neuron import h

def serialize_spike_train(arr):
    """Convert np.ndarray, list, int, or float to a string for saving in CSV."""
    # Handle None
    if arr is None:
        arr = []
    # Handle numpy arrays
    elif isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            arr = np.array([arr.item()])
    # Handle plain int/float
    elif isinstance(arr, (int, np.integer)):
        arr = [int(arr)]
    elif isinstance(arr, float):
        arr = [] if np.isnan(arr) else [int(arr)]
    # Now treat arr as list-like
    return "[" + " ".join(str(int(x)) for x in arr) + "]"



def deserialize_spike_train(s):
    """
    Convert a string, int, float, list, or array from CSV back to a numpy array of ints.
    Always returns a numpy array.
    """
    if isinstance(s, np.ndarray):
        return s.astype(int)
    if isinstance(s, list):
        return np.array(s, dtype=int)
    if isinstance(s, (int, np.integer)):
        return np.array([s], dtype=int)
    if isinstance(s, float):
        if np.isnan(s):
            return np.array([], dtype=int)
        else:
            return np.array([int(s)], dtype=int)
    if isinstance(s, str):
        s = s.strip("[]").strip()
        if not s:
            return np.array([], dtype=int)
        return np.fromstring(s, sep=' ', dtype=int)
    return np.array([], dtype=int)


# define a class for generating synapses abstractly
class PreSimSynapseGenerator:
    def __init__(self, sim_dir):

        self.sim_dir = sim_dir

        self.segments = pd.read_csv(os.path.join(sim_dir, "segment_data.csv"))

        with open(os.path.join(sim_dir, "parameters.pickle"), 'rb') as file:
            self.parameters = pickle.load(file)

        if os.path.exists(os.path.join(self.sim_dir, "synapses.csv")):
            self.synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))
            self.synapses['spike_train'] = self.synapses['spike_train'].apply(deserialize_spike_train)
        else:
            self.synapses = pd.DataFrame()

        self.logger = Logger(sim_dir)

        if self.parameters.segment_measurement_for_probabilities not in ['length', 'surface_area']:
            raise ValueError(f"Measurement for probabilities must be 'length' or 'surface_area'. Not {self.parameters.segment_measurement_for_probabilities}.")
        
        # load modfiles
        try:
            h.load_file('stdrun.hoc')
            # h.nrn_load_dll('./x86_64/.libs/libnrnmech.so' # IF IN SCRIPTS FOLDER
            load_modfiles = h.nrn_load_dll('../scripts/x86_64/.libs/libnrnmech.so') # IF IN SIMULATIONS FOLDER
            if load_modfiles != 1:
                raise Exception("Error loading mod files")
            else:
                print("Mod files loaded successfully")
        except:
            # Already loaded
            pass 

    def generate_synapse_locations(self):
        for syn_properties_set, use_density, syn_mod, syn_params_choices in zip([self.parameters.exc_syn_properties, self.parameters.inh_syn_properties],
                                                            [self.parameters.exc_use_density, self.parameters.inh_use_density],
                                                            [self.parameters.exc_syn_mod, self.parameters.inh_syn_mod],
                                                            [self.parameters.exc_syn_params_choices, self.parameters.inh_syn_params_choices]):
            for sec_type, synapse_properties in syn_properties_set.items():
                self.logger.log(f"Generating synapses for {sec_type.upper()} with properties: {synapse_properties}")
                segments_to_generate_on = self.get_segments_of_type(sec_type, self.segments) # get segments
                self.random_state = np.random.RandomState(self.parameters.inh_syn_properties[sec_type]['seed']['synapses'])
                np.random.seed(self.parameters.inh_syn_properties[sec_type]['seed']['synapses'])

                synapses_this_sec_type = self.build_synapses_with_specs(segments_to_generate_on = segments_to_generate_on,
                                        sec_type = sec_type,
                                        synapse_type = synapse_properties['synapse_type'],
                                        use_density = use_density,
                                        syn_number = synapse_properties['syn_number'] if not use_density else None,
                                        syn_density = synapse_properties['syn_density'] if use_density else None,
                                        initial_weight_distribution = synapse_properties['initial_weight_distribution'],
                                        release_probability_distribution = synapse_properties['release_probability_distribution'],
                                        syn_mod = syn_mod,
                                        syn_params_choices = syn_params_choices,
                                        name = f"{synapse_properties['synapse_type']}_{sec_type}"
                                        )
                self.synapses = pd.concat((self.synapses, synapses_this_sec_type), ignore_index=True)
        self.synapses = self.synapses.reset_index(drop=True) #@TODO: check if this is necessary and check self.synapses.

    def get_segments_of_type(self, sec_type, segments):
        return segments[segments['sec_type_precise'] == sec_type]
    
    def build_synapses_with_specs(self, segments_to_generate_on: pd.DataFrame, sec_type: str, synapse_type: str, use_density: bool,
                                  syn_number: int, syn_density: float, initial_weight_distribution: dict,
                                  release_probability_distribution: dict, name: str, syn_mod:str, syn_params_choices) -> pd.DataFrame:
        
        if initial_weight_distribution is None or release_probability_distribution is None:
            raise ValueError("Both gmax_dist_params and P_release_params must be provided.")
        
        if syn_number is None and syn_density is None:
            raise ValueError("Either syn_number or syn_density must be provided.")
        elif syn_number is not None and syn_density is not None:
            raise ValueError("Only one of syn_number or syn_density must be provided.")
        elif syn_number is not None and use_density:
            raise ValueError("syn_number should be none when using density.")
        elif syn_density is not None and not use_density:
            raise ValueError("syn_density should be none when not using density.")
        
        # if len(np.unique(segments_to_generate_on)) != len(segments_to_generate_on):
        #     raise ValueError(f"Segments to generate on must be unique. Not {segments_to_generate_on}.")

        #@TODO: check that this doesn't throw an error with expected inputs.
        #@TODO: move initial_weight_distribution info to a dictionary within synapse_properties, same for release probability_distribution.
        if not isinstance(syn_params_choices, dict):
            raise ValueError(f"syn_params_choices must be a dict. Not {type(syn_params_choices)}. syn_params_choices: {syn_params_choices}.")
        # if False in [True if isinstance(syn_params, dict) else False for syn_params in syn_params_choices]: # changed to dict with keys 'choices' and 'probs' where each choice is a dict.
        #     raise ValueError(f"syn_params_choices must be a dict of dictionaries. Not {type(syn_params_choices)}. syn_params_choices: {syn_params_choices}.")

        if initial_weight_distribution['function'] is not None:
            initial_weight_distribution = partial(initial_weight_distribution['function'], **initial_weight_distribution['params'], size=1)
            initial_weight_distribution_is_partial = True
        else:
            initial_weight_distribution = initial_weight_distribution['params']['mean'] # use mean if no function is provided
            initial_weight_distribution_is_partial = False
        self.logger.log(f"Initial weight distribution for {name}: {initial_weight_distribution}")

        if release_probability_distribution['function'] is not None:
            release_probability_distribution = partial(release_probability_distribution['function'], **release_probability_distribution['params'], size=1)
            release_probability_distribution_is_partial = True
        else:
            release_probability_distribution = release_probability_distribution['params']['mean']
            release_probability_distribution_is_partial = False
        self.logger.log(f"Release probability distribution for {name}: {release_probability_distribution}")

        # calculate probabilities of placing synapses on segments
        total_measurement = segments_to_generate_on[self.parameters.segment_measurement_for_probabilities].sum()
        self.logger.log(f"Total {self.parameters.segment_measurement_for_probabilities} for {name}: {total_measurement}")
        # segments_to_generate_on['probability'] = segments_to_generate_on[self.parameters.segment_measurement_for_probabilities] / total_measurement
        segments_to_generate_on = segments_to_generate_on.assign(
            probability=lambda df: df[self.parameters.segment_measurement_for_probabilities] 
                                / total_measurement
        )

        if segments_to_generate_on['probability'].sum() < 0.999 or segments_to_generate_on['probability'].sum() > 1.001:
            raise ValueError(f"Probabilities do not sum to 1 instead {segments_to_generate_on['probability'].sum()}. Check your segment measurement for probabilities.")

        if use_density:
            # calculate number of synapses per segment
            syn_number = int(total_measurement * syn_density) #@TODO make compatible with syn_density being function instead of float (not at all urgent)
        self.logger.log(f"Number of synapses being generated for {name}: {syn_number}")

        # synapses = pd.DataFrame(columns=['name', 'modfile', 'initW', 'gmax', 'release_probability', 'seg_id']) #@TODO: add columns for possible syn_params keys

        rows = []
        for _ in range(syn_number): # @TODO: do this in parallel instead of serial. Use list comprehension?
            # sample a segment
            segment_id = self.random_state.choice(a=segments_to_generate_on['seg_id'], size=1, replace=True, p=segments_to_generate_on['probability'])[0] #@TODO: check if [0] is necessary
            segment = segments_to_generate_on[segments_to_generate_on['seg_id'] == segment_id]
            # choose sub-synapse type (short term plasticity, gbar, etc. properties)
            if len(syn_params_choices['choices']) == 1:
                syn_params_made_choice = syn_params_choices['choices'][0]
            elif syn_params_choices['probs'] == 'perisomatic_distance':
                syn_params_made_choice = syn_params_choices['choices'][1] if segment['Distance'].values[0] > 100 else syn_params_choices['choices'][0]
            else:
                syn_params_made_choice = self.random_state.choice(syn_params_choices['choices'], p=syn_params_choices['probs']) # choose between CS2CP and CP2CP if it is AMPA. pyr2pyr will not be a tuple or list.

            choice_name, syn_params_this_syn = next(iter(syn_params_made_choice.items())) #TODO: check with choices in constants.__post_init__
            syn_params_this_syn['syn_params_choice'] = choice_name

            #TODO: update synapses.py so that syn_params that use AMPA_NMDA and GABA_AB modfiles have modfile indicated.

            # sample a release probability
            if release_probability_distribution_is_partial:
                syn_params_this_syn["release_probability"] = release_probability_distribution(size=1) # sample distribution
            else:
                syn_params_this_syn["release_probability"] = release_probability_distribution # distribution is a constant
            if 'int2pyr' in syn_mod or 'pyr2pyr' in syn_mod:  # these modfiles do release probability computation as spikes arrive during simulation instead of before
                syn_params_this_syn["P_0"] = syn_params_this_syn["release_probability"]
            else: # syn_mod does not have attribute for release probability so we approximate it by testing 1 release for entire simulation.
                p_test = self.random_state.uniform(low=0, high=1, size=1)
                if p_test < syn_params_this_syn["release_probability"]:
                    syn_params_this_syn["P_0"] = 1 #@TODO: when actually building synapses if syn_mod is not int2pyr or pyr2pyr then do not generate synapses with P_0 = 0. And skip assigning P_0 to the synapse object.
                else:
                    syn_params_this_syn["P_0"] = 0 #synapse is not releasing

            # sample an initial weight
            if initial_weight_distribution_is_partial:
                syn_params_this_syn["initW"] = initial_weight_distribution(size=1) # sample distribution
            else:
                syn_params_this_syn["initW"] = initial_weight_distribution # distribution is a constant

            # print(f"syn_params_this_syn: {syn_params_this_syn}")
            syn_params_this_syn["seg_id"] = segment['seg_id'].values[0]

            # pick out only the keys that contain 'gbar'
            gbar_params = {
                k: v
                for k, v in syn_params_this_syn.items()
                if 'gbar' in k
            }
            row = {
                'name': f"{name}_{_}",
                'modfile': syn_mod,
                'P_0': syn_params_this_syn["P_0"],
                'initW': syn_params_this_syn["initW"],
                'cell2cell_type': syn_params_this_syn["syn_params_choice"],
                'seg_id': syn_params_this_syn["seg_id"],
                **gbar_params,
            }
            rows.append(row)
        return pd.DataFrame(rows)
            # # add row to dataframe
            # self.synapses = pd.concat((self.synapses, pd.DataFrame({
            #     'name': f"{name}_{_}",
            #     'modfile': syn_mod,
            #     # 'initW': syn_params_this_syn["initW"],
            #     # 'gmax': syn_params_this_syn["gmax"],
            #     # 'release_probability': syn_params_this_syn["release_probability"],
            #     'P_0': syn_params_this_syn["P_0"],
            #     'initW': syn_params_this_syn["initW"],
            #     'cell2cell_type': syn_params_this_syn["syn_params_choice"],

            #     'seg_id': syn_params_this_syn["seg_id"], 
            #     **gbar_params,
            #     # **syn_params_this_syn 
            #     # #TODO: (SHOULD BE DONE) use a string to indicate which syn_params to use instead of storing all of them in the DataFrame. Also will need to pull out the syn_params that were added to syn_params in this snippet (such as initW, location, release_probability.) and give them their own column.
            # }, index=[0])), ignore_index=True)

    def assign_synapses_to_cell_assemblies(self, synapses: pd.DataFrame, coords: np.ndarray, clustering_config: dict) -> pd.DataFrame:
        """
        Assigns each synapse to a functional group (FG) and presynaptic cell (PC).
        FG and PC labels are -1 for background/not assigned.
        """
        fg_labels = -np.ones(len(synapses), dtype=int)
        pc_labels = -np.ones(len(synapses), dtype=int)
        for fg_idx, fg in enumerate(clustering_config.get('functional_groups', [])):
            fg_center = np.array(fg['center'])
            fg_radius = fg['radius']
            distances_to_fg = np.linalg.norm(coords - fg_center, axis=1)
            in_fg = distances_to_fg <= fg_radius
            fg_labels[in_fg] = fg_idx
            for pc_idx, pc in enumerate(fg.get('presynaptic_cells', [])):
                pc_center = np.array(pc['center'])
                pc_radius = pc['radius']
                distances_to_pc = np.linalg.norm(coords - pc_center, axis=1)
                in_pc = (distances_to_pc <= pc_radius) & in_fg
                pc_labels[in_pc] = pc_idx
        synapses['functional_group'] = fg_labels
        synapses['presynaptic_cell'] = pc_labels
        return synapses

    def generate_spike_trains_for_cell_assemblies(
        self,
        synapses: pd.DataFrame,
        parameters,
        h_tstop: int,
        sec_type: str,
        synapse_type: str,
        random_state: np.random.RandomState,
        fg_traces_store=None,
        pc_spike_trains_store=None,
        all_synapses_full=None
    ):
        """
        For each FG and PC, generate appropriate spike trains for synapses (including clustered, background, and delay modes).
        """

        # Ensure object columns exist and are ready for assignment
        if 'spike_train' not in synapses.columns:
            synapses['spike_train'] = [None] * len(synapses)
        if 'pc_mean_firing_rate' not in synapses.columns:
            synapses['pc_mean_firing_rate'] = np.nan

        if fg_traces_store is None:
            fg_traces_store = {}
        if pc_spike_trains_store is None:
            pc_spike_trains_store = {}

        props = getattr(parameters, f"{synapse_type}_syn_properties")[sec_type]
        spike_train_mode = props.get('spike_train_mode', 'standard')
        mean_fr_dist = partial(
            props['mean_firing_rate_distribution']['function'],
            **props['mean_firing_rate_distribution']['params'], size=1
        )

        allowed_modes = {'standard', 'pink_noise', 'rhythmic', 'delay'}
        if spike_train_mode not in allowed_modes:
            raise NotImplementedError(
                f"spike_train_mode '{spike_train_mode}' is not implemented. Allowed: {sorted(allowed_modes)}"
            )

        clustering_config = getattr(parameters, f"{synapse_type}_clustering", {}).get(sec_type, {})
        n_fg = len(clustering_config.get('functional_groups', []))
        delay_config = props.get('delay_config', {})

        def collect_reference_spike_trains(ref_synapse_type, ref_sec_type, ref_fg_id, ref_pc_id):
            if all_synapses_full is None:
                raise ValueError("all_synapses_full must be provided for delay mode.")
            mask = (all_synapses_full['spike_train'].apply(lambda x: isinstance(x, (np.ndarray, list))))
            if ref_synapse_type != 'all':
                mask &= all_synapses_full['name'].str.contains(ref_synapse_type, na=False)
            if ref_sec_type != 'all':
                mask &= all_synapses_full['name'].str.contains(ref_sec_type, na=False)
            if ref_fg_id != 'all':
                mask &= (all_synapses_full['functional_group'] == ref_fg_id)
            if ref_pc_id != 'all' and ref_pc_id is not None:
                mask &= (all_synapses_full['presynaptic_cell'] == ref_pc_id)
            trains = all_synapses_full.loc[mask, 'spike_train'].tolist()

            # out_trains = []
            # for train in trains:
            #     # Always deserialize if it's not already an array/list
            #     arr = deserialize_spike_train(train)

            #     if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
            #         out_trains.append(arr)
            # if not out_trains:
            #     raise ValueError(
            #         f"No spike trains found for delay reference with mask: "
            #         f"ref_synapse_type={ref_synapse_type}, ref_sec_type={ref_sec_type}, "
            #         f"ref_fg_id={ref_fg_id}, ref_pc_id={ref_pc_id}")
            # return out_trains

            # All trains are already arrays at this point
            out_trains = [arr for arr in trains if arr is not None and hasattr(arr, "__len__") and len(arr) > 0]
            if not out_trains:
                raise ValueError(
                    f"No spike trains found for delay reference with mask: "
                    f"ref_synapse_type={ref_synapse_type}, ref_sec_type={ref_sec_type}, "
                    f"ref_fg_id={ref_fg_id}, ref_pc_id={ref_pc_id}")
            return out_trains

        def generate_fg_trace(fg_id=None, fg=None):
            """
            Returns the modulatory trace (lambda over time) for a functional group.
            Respects FG 'modulation_mode' override if present.
            """
            mode = spike_train_mode if fg is None else fg.get('modulation_mode', spike_train_mode)
            if mode == 'standard':
                return np.ones(h_tstop)
            elif mode == 'pink_noise':
                fg_trace = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
                    num=h_tstop, random_state=random_state)
                mean_val = np.mean(fg_trace)
                if mean_val == 0:
                    raise ValueError("Mean value of pink noise trace is zero; cannot normalize.")
                return fg_trace / mean_val
            elif mode == 'rhythmic':
                base = np.ones(h_tstop)
                freq = props.get('rhythmic_frequency', None)
                depth = props.get('rhythmic_depth', None)
                delta_t = getattr(parameters, 'delta_t', 1)
                if freq is None or depth is None:
                    raise ValueError("Both 'rhythmic_frequency' and 'rhythmic_depth' must be set for rhythmic mode.")
                return PoissonTrainGenerator.rhythmic_modulation(base, freq, depth, delta_t)
            elif mode == 'delay':
                shift = delay_config.get('delay_shift', None)
                if shift is None:
                    raise ValueError("delay_shift must be specified in delay_config for 'delay' mode.")
                ref_synapse_type = delay_config.get('ref_synapse_type', 'exc')
                ref_sec_type = delay_config.get('ref_sec_type', sec_type)
                ref_fg_id = delay_config.get('ref_fg_id', fg_id)
                ref_pc_id = delay_config.get('ref_pc_id', None)
                trains = collect_reference_spike_trains(
                    ref_synapse_type, ref_sec_type, ref_fg_id, ref_pc_id)
                if not all(isinstance(train, (np.ndarray, list)) and len(train) > 0 for train in trains):
                    raise ValueError("All reference spike trains for delay must be non-empty arrays/lists.")
                delayed_lambdas = PoissonTrainGenerator.generate_lambdas_by_delaying(h_tstop, trains)
                return delayed_lambdas
            else:
                raise NotImplementedError(f"Unrecognized spike_train_mode: {mode}")

        unique_fgs = np.unique(synapses['functional_group'])
        for fg_id in unique_fgs:
            fg_mask = (synapses['functional_group'] == fg_id)
            fg = None
            if n_fg > 0 and fg_id >= 0:
                fg = clustering_config['functional_groups'][int(fg_id)]
            fg_trace = generate_fg_trace(fg_id=fg_id, fg=fg)
            if fg_trace is None or not isinstance(fg_trace, np.ndarray) or np.any(np.isnan(fg_trace)):
                raise ValueError(f"fg_trace is not valid for FG {fg_id}, got: {fg_trace}")
            fg_traces_store[(synapse_type, sec_type, fg_id)] = fg_trace

            pcs_in_fg = np.unique(synapses.loc[fg_mask, 'presynaptic_cell'])
            for pc_id in pcs_in_fg:
                pc_mask = fg_mask & (synapses['presynaptic_cell'] == pc_id)
                if pc_id == -1:
                    for idx in synapses[pc_mask].index:
                        mean_fr = mean_fr_dist(size=1)
                        if not np.isfinite(mean_fr) or mean_fr <= 0:
                            raise ValueError(f"Background mean firing rate is not positive: {mean_fr}")
                        if spike_train_mode == 'pink_noise':
                            # Generate a unique pink noise modulation for each background synapse if it is pink noise 
                            fg_trace_bg = generate_fg_trace(fg_id=None, fg=None)  # or use fg_id, fg if you want it FG-specific
                            lambdas = PoissonTrainGenerator.shift_mean_of_lambdas(fg_trace_bg, mean_fr)
                        elif spike_train_mode == 'standard':
                            lambdas = np.ones(h_tstop) * mean_fr # constant fr timecourse (still generated from poisson random sampling)
                        else:
                            lambdas = PoissonTrainGenerator.shift_mean_of_lambdas(fg_trace, mean_fr) # if it delayed or rhythmic then the modulation trace will not change between synapses of this type by definition.
                        spike_train = PoissonTrainGenerator.generate_spike_train(lambdas, random_state)
                        synapses.at[idx, 'spike_train'] = spike_train.spike_times
                        synapses.at[idx, 'pc_mean_firing_rate'] = mean_fr
                    continue
                mean_fr = mean_fr_dist(size=1)
                if not np.isfinite(mean_fr) or mean_fr <= 0:
                    raise ValueError(f"PC mean firing rate is not positive: {mean_fr}")
                # Standard: shift mean, Delay: use fg_trace directly (already population-based)
                pc_trace = (
                    fg_trace if spike_train_mode == 'delay'
                    else PoissonTrainGenerator.shift_mean_of_lambdas(fg_trace, mean_fr)
                )
                if pc_trace is None or not isinstance(pc_trace, np.ndarray) or np.any(np.isnan(pc_trace)):
                    raise ValueError(f"pc_trace is not valid for FG {fg_id} PC {pc_id}, got: {pc_trace}")
                spike_train = PoissonTrainGenerator.generate_spike_train(pc_trace, random_state)
                for idx in synapses[pc_mask].index:
                    synapses.at[idx, 'spike_train'] = spike_train.spike_times
                    synapses.at[idx, 'pc_mean_firing_rate'] = mean_fr
                pc_spike_trains_store[(synapse_type, sec_type, fg_id, pc_id)] = spike_train
        return synapses, fg_traces_store, pc_spike_trains_store

    def generate_spike_trains_for_synapses(self):
        """
        Main entrypoint: assigns FG/PC and generates spike trains for all synapses.
        Ensures all excitatory spike trains are generated before any delayed inhibition.
        """
        synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))

        segments = self.segments
        parameters = self.parameters
        h_tstop = self.parameters.h_tstop

        columns = ['pc_0', 'pc_1', 'pc_2']
        synapses_with_seg_info = synapses.merge(
            segments, on='seg_id', how='left', suffixes=('','_seg'))
        synapse_coords = synapses_with_seg_info[columns].values

        # Pass 1: Assign all FG/PCs for ALL synapses
        all_fg_labels = np.full(len(synapses), -1, dtype=int)
        all_pc_labels = np.full(len(synapses), -1, dtype=int)
        for synapse_type in ['exc', 'inh']:
            properties_set = getattr(parameters, f"{synapse_type}_syn_properties")
            for sec_type, props in properties_set.items():
                syn_mask = (
                    synapses['name'].str.contains(synapse_type, na=False)
                    & synapses['name'].str.contains(sec_type, na=False)
                )
                coords_this_type = synapse_coords[syn_mask]
                clustering_config = getattr(parameters, f"{synapse_type}_clustering", {}).get(sec_type, {})
                assigned_synapses = self.assign_synapses_to_cell_assemblies(
                    synapses.loc[syn_mask].copy(), coords_this_type, clustering_config)
                all_fg_labels[syn_mask] = assigned_synapses['functional_group'].values
                all_pc_labels[syn_mask] = assigned_synapses['presynaptic_cell'].values
        synapses['functional_group'] = all_fg_labels
        synapses['presynaptic_cell'] = all_pc_labels

        all_synapses_full = synapses.copy()

        fg_traces_store = {}
        pc_spike_trains_store = {}

        # Pass 2: Generate spike trains for ALL non-delayed first
        for synapse_type in ['exc', 'inh']:
            properties_set = getattr(parameters, f"{synapse_type}_syn_properties")
            for sec_type, props in properties_set.items():
                spike_train_mode = props.get('spike_train_mode', 'standard')
                if spike_train_mode == 'delay':
                    continue  # skip delayed for now
                syn_mask = (
                    synapses['name'].str.contains(synapse_type, na=False)
                    & synapses['name'].str.contains(sec_type, na=False)
                )
                assigned_synapses = synapses.loc[syn_mask].copy()
                seed = props['seed']['synapses']
                random_state = np.random.RandomState(seed)
                assigned_synapses, fg_traces_store, pc_spike_trains_store = self.generate_spike_trains_for_cell_assemblies(
                    assigned_synapses, parameters, h_tstop, sec_type, synapse_type, random_state,
                    fg_traces_store=fg_traces_store, pc_spike_trains_store=pc_spike_trains_store,
                    all_synapses_full=all_synapses_full)
                for col in ['spike_train', 'pc_mean_firing_rate', 'functional_group', 'presynaptic_cell']:
                    synapses.loc[assigned_synapses.index, col] = assigned_synapses[col]
                all_synapses_full.loc[assigned_synapses.index, 'spike_train'] = assigned_synapses['spike_train']

        all_synapses_full['spike_train'] = all_synapses_full['spike_train'].apply(deserialize_spike_train)

        # Pass 3: Now, generate spike trains for all delayed synapses
        for synapse_type in ['exc', 'inh']:
            properties_set = getattr(parameters, f"{synapse_type}_syn_properties")
            for sec_type, props in properties_set.items():
                spike_train_mode = props.get('spike_train_mode', 'standard')
                if spike_train_mode != 'delay':
                    continue  # only do delayed in this pass
                syn_mask = (
                    synapses['name'].str.contains(synapse_type, na=False)
                    & synapses['name'].str.contains(sec_type, na=False)
                )
                assigned_synapses = synapses.loc[syn_mask].copy()
                seed = props['seed']['synapses']
                random_state = np.random.RandomState(seed)
                assigned_synapses, fg_traces_store, pc_spike_trains_store = self.generate_spike_trains_for_cell_assemblies(
                    assigned_synapses, parameters, h_tstop, sec_type, synapse_type, random_state,
                    fg_traces_store=fg_traces_store, pc_spike_trains_store=pc_spike_trains_store,
                    all_synapses_full=all_synapses_full)
                for col in ['spike_train', 'pc_mean_firing_rate', 'functional_group', 'presynaptic_cell']:
                    synapses.loc[assigned_synapses.index, col] = assigned_synapses[col]
                all_synapses_full.loc[assigned_synapses.index, 'spike_train'] = assigned_synapses['spike_train']

        synapses['spike_train'] = synapses['spike_train'].apply(serialize_spike_train)
        synapses.to_csv(os.path.join(self.sim_dir, "synapses.csv"), index=False)



    def build_synapses_onto_cell_obj(self)->CellModel:
        ## assumes already have parameters and sim_dir defined, even logger (use run_on_all_sims)

        # build synapses from synapses csv from simulation folder
        synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))
        synapses['spike_train'] = synapses['spike_train'].apply(deserialize_spike_train)

        # # convert each string back to an array
        # synapses["spike_train"] = synapses["spike_train"].apply(
        #     lambda s: np.fromstring(s.strip("[]"), sep=" ")
        # )

        syn_param_map = {cell2cell_type: getattr(synapse, f"{cell2cell_type}_syn_params") for cell2cell_type in np.unique(synapses.cell2cell_type)}

        # set spike trains from spike_trains.csv from simulation folder

        logger = Logger(self.sim_dir) # create per‑sim logger (write info into "sims_dir/sim_dir/log.txt")

        logger.log(f"Building cell")
        cell_builder = CellBuilder(getattr(SkeletonCell, self.parameters.skeleton_cell_type), self.parameters, logger)
        cell, _ = cell_builder.build_cell()
        logger.log(f"Cell finished building.")

        neuron_r = cell.neuron_r
        param_map = syn_param_map

        all_segments, seg_data = cell.get_segments(['all'])
        # build synapses
        syn_list = [
            # unpack the namedtuple directly
            Synapse(
                segment    = all_segments[row.seg_id],
                syn_mod    = row.modfile,
                syn_params = param_map[row.cell2cell_type],
                gmax       = row.initW,
                neuron_r   = neuron_r,
                name       = row.name,
            )
            for row in synapses.itertuples(index=False)
        ]
        logger.log("Synapse object list finished building")

        # 3. Now set all the spike trains in another pass
        for syn, train, i in zip(syn_list, synapses["spike_train"], range(len(syn_list))):
            # logger.log(f"Setting spike train for synapse {i}")
            # train_to_do = np.asarray(parse_array(train))
            # print(f"tran_to_do: {train_to_do}")
            # print(f"type(tran_to_do): {type(train_to_do)}")
            syn.set_spike_train(train)#, logger=logger)
            # logger.log(f"Success setting spike train for synapse {i}")
            # logger.log(f"check syn attributes after. netcons: {syn.netcons}. vec: {syn.vec}. stim: {syn.stim}. vecstim: {syn.vecstim}")

        # 4. Finally attach them in one go
        logger.log("Storing synapses list in CellModel object")
        cell.synapses.extend(syn_list)
        logger.log("Finish synapses list in CellModel object")
        return cell

#################### ADDITIONAL CODE FOR CLUSTERING THAT WAS WORK IN PROGRESS #####################



#         # only consider synapses of this synapse type
#         synapse_ids_to_consider = synapses[synapses['name'].str.contains(synapse_type)]

# ## get cluster centers randomly
# # # get the mean and std of the coordinates
# # mean = np.mean(synapse_coords, axis=0)
# # std = np.std(synapse_coords, axis=0)
# # # get the range of the coordinates
# # range = np.max(synapse_coords, axis=0) - np.min(synapse_coords, axis=0)
# # # get 10 random cluster centers
# # cluster_centers = np.random.uniform(low=mean - 3*std, high=mean + 3*std, size=(10, 3)) # TODO: check (can be outside of mins and max, leading to error.)
# # cluster_centers = np.clip(cluster_centers, mean - 3*std, mean + 3*std)
# # get 10 random cluster centers by choosing among segments.
# cluster_centers = synapse_coords[random_state.choice(synapse_coords.shape[0], size=10, replace=False)]

# ## get synapse_ids for each cluster within bounds
# # get the coordinates of the synapses
# synapse_coords = synapses_with_seg_info[columns].values

# # use the distance of each synapse from the cluster center to determine if it belongs
# cluster_indices_by_cluster = []
# for cluster_center in cluster_centers:
#     distances = np.linalg.norm(synapse_coords - cluster_center, axis=1)#cluster_center, axis=1)
#     # get the indices of the synapses that are within 3 std of the cluster center
#     cluster_indices = np.where(distances < 100)[0]
#     # make sure the indices are unique across clusters
#     cluster_indices = np.unique(cluster_indices)

#     # right now cluster_indices tell the row of synapses_with_seg_info. Need to do the same, but
#     # only consider for cluster_indices the segments 
#     # of synapses_with_seg_info['sec_type_precise'] == sec_type
#     cluster_indices = np.where(synapses_with_seg_info['sec_type_precise'] == sec_type)[0][cluster_indices] #TODO: CHECK

#     # track for all clusters so we can deal with overlapping clusters
#     cluster_indices_by_cluster.append(cluster_indices)

# ## deal with overlapping clusters
# from collections import defaultdict
# # turn each cluster's indices into a mutable set
# cluster_sets = [set(idxs) for idxs in cluster_indices_by_cluster]
# # build a map from each synapse-index to the list of clusters it appears in
# idx_to_clusters = defaultdict(list)
# for cid, idxs in enumerate(cluster_sets):
#     for idx in idxs:
#         idx_to_clusters[idx].append(cid)
# # (optional) for reproducibility
# # np.random.seed(42)
# # for any index in >1 cluster, choose one cluster to keep it
# for idx, cids in idx_to_clusters.items():
#     if len(cids) > 1:
#         keep = np.random.choice(cids)
#         for cid in cids:
#             if cid != keep:
#                 cluster_sets[cid].remove(idx)

# # convert back to sorted numpy arrays (if you need arrays)
# cluster_indices_by_cluster = [
#     np.array(sorted(s)) for s in cluster_sets
# ]

# ## generate cluster spike trains
# # get cluster centers
# # get row indices that will be clustered for each cluster center (list of lists)
# # make sure that row indices are unique across clusters
# # generate spike trains for each cluster
# from Modules.spike_generator import PoissonTrainGenerator
# for cluster_indices in cluster_indices_by_cluster:
#     # generate FR profile for this cluster
#     firing_rates = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
#         num = parameters.h_tstop,
#         random_state = random_state)
    
#     mean_fr = 

#     # generate spike train for each synapse
#     for synapse in synapses.iloc[cluster_indices]:
#         firing_rates_shifted = PoissonTrainGenerator.shift_mean_of_lambdas(firing_rates, desired_mean=mean_fr) 
#         spike_train = PoissonTrainGenerator.generate_spike_train(
#             lambdas = firing_rates_shifted, 
#             random_state = random_state)

# ## generate background spike train
# # get row indices that are not in clusters
# # generate spike trains for these synapses



################################ MORE #############


# USE cell_builder.assign_spikes and presynaptic.py for reference code. 
# @TODO: adapt presynaptic.py to use segments.csv instead of cell object. cell object is heavilty embedded in presynaptic.py module 

# generate functional groups from params

# generate presynaptic cells from functional groups and params

# generate spike trains for presynaptic cells from params

# load synapse locations from synapses csv

# cluster synapse locations into presynaptic cells

# assign synapses to presynaptic cells

# save spike trains and synapse assignments as spike_trains.csv to simulation folder


# #### new code for clusters ####

# # generate 'background' for all. Then form clusters

# def generate_new_spike_trains(synapses: pd.DataFrame, segments:pd.DataFrame)-> pd.DataFrame: # TODO: make functions and class. started this way then realized better to not.
#     """
#     Generate spike trains for each synapse in the synapses DataFrame.
#     """
#     synapses_with_seg_info = synapses.merge( # TODO: alternative could be used to save memory.
#         segments, 
#         on='seg_id', 
#         how='left',               # carry along all synapses even if a seg_id is missing
#         suffixes=('','_seg')      # e.g. if both have a 'length' column
#     )

#     ## generate cluster spike trains
#     # get cluster center coordinates
#     centers_coords = [get_cluster_center_coords(synapses_with_seg_info)]
#     # get row indices that will be clustered for each cluster center (list of lists)
#     # make sure that row indices are unique across clusters
#     # generate spike trains for each cluster

#     ## generate background spike train
#     # get row indices that are not in clusters
#     # generate spike trains for these synapses

# def get_cluster_center_coords(synapses_with_seg_info: pd.DataFrame, num_centers:int = 10, columns:list = ['pc_0', 'pc_1', 'pc_2']) -> tuple:
#     """
#     Get the coordinates of the cluster center.
#     randomly pick coordiates within 3 std the range of the coordinates of the synapses #TODO: update to pick branches
#     """
#     # get the coordinates of the synapses
#     synapse_coords = synapses_with_seg_info[columns].values
#     # get the mean and std of the coordinates
#     mean = np.mean(synapse_coords, axis=0)
#     std = np.std(synapse_coords, axis=0)
#     # get the range of the coordinates
#     range = np.max(synapse_coords, axis=0) - np.min(synapse_coords, axis=0)

#     cluster_center = np.random.uniform(low=mean - 3*std, high=mean + 3*std, size=(num_centers,3))
#     # make sure the cluster center is within the range of the coordinates
#     cluster_center = np.clip(cluster_center, mean - 3*std, mean + 3*std)
#     # return the coordinates of the cluster center
#     return tuple(cluster_center)

# def get_cluster_row_indices(synapses_with_seg_info: pd.DataFrame, cluster_center: tuple, columns:list = ['pc_0', 'pc_1', 'pc_2'])-> list:
#     """
#     Get the row indices of the cluster provided the center of the cluster
#     """
#     # get the coordinates of the synapses
#     synapse_coords = synapses_with_seg_info[columns].values
#     # get the distance of each synapse from the cluster center
#     distances = np.linalg.norm(synapse_coords - cluster_center, axis=1)
#     # get the indices of the synapses that are within 3 std of the cluster center
#     cluster_indices = np.where(distances < 3*np.std(distances))[0]
#     # make sure the indices are unique across clusters
#     cluster_indices = np.unique(cluster_indices)
#     # return the indices of the synapses in the cluster
#     return cluster_indices.tolist()

# def generate_background_spike_trains():
#     pass

# def generate_cluster_spike_trains():
#     pass