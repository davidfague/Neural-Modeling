'''
Modules/synapses_file.py
'''
from tokenize import group
import pandas as pd
import numpy as np
from Modules.pre_sim.spike_generator import PoissonTrainGenerator
import os
from functools import partial
import pickle
from Modules.logger import Logger

# for build_synapses_onto_cell_obj #TODO: move build_synapses_onto_cell_obj to cell_builder
import Modules.cell_model.synapse as synapse
from Modules.cell_model.synapse import Synapse
from Modules.cell_model.cell_model import CellModel
from Modules.cell_model.cell_builder import CellBuilder, SkeletonCell
from Modules.logger import Logger
import numpy as np
from neuron import h
import json
import h5py

#@TODO: clean up random seeding, clean up parameter loading.
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

### SAVING FG TRACES ###
def _fg_key(src: str, fg_id: int, syn_type: str) -> str:
    # A compact key for each FG
    return f"{syn_type}__{src}__fg{fg_id}"

def save_fg_traces_h5(fg_traces_store: dict, path: str):
    with h5py.File(path, "w") as f:
        for (syn_type, src, fg_id), lam in fg_traces_store.items():
            if lam is None or fg_id < 0:
                continue
            grp = f.require_group(f"{syn_type}/{src}/fg_{int(fg_id)}")
            dset = grp.create_dataset("lambda", data=np.asarray(lam, dtype=float), compression="gzip")
            dset.attrs["syn_type"] = syn_type
            dset.attrs["input_source"] = src
            dset.attrs["fg_id"] = int(fg_id)
            dset.attrs["length"] = int(len(lam))

def load_fg_traces_h5(path):
    """
    Returns:
      traces: dict keyed by (syn_type, input_source, fg_id) -> np.ndarray
      meta:   list of dicts with attrs for each FG
    """
    traces = {}
    meta = []
    with h5py.File(path, "r") as f:
        # walk all groups and pick those that end with fg_*/lambda
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("/lambda"):
                # name like: "exc/tuft_local_L5/fg_3/lambda"
                parts = name.split("/")
                syn_type, input_source, fg_part, _ = parts[-4], parts[-3], parts[-2], parts[-1]
                fg_id = int(fg_part.split("_")[1])

                lam = obj[()]  # load the array
                traces[(syn_type, input_source, fg_id)] = lam

                # grab attributes saved with the dataset (optional)
                attrs = dict(obj.attrs)
                # ensure required keys are present even if attrs weren’t set
                attrs.setdefault("syn_type", syn_type)
                attrs.setdefault("input_source", input_source)
                attrs.setdefault("fg_id", fg_id)
                attrs.setdefault("length", int(len(lam)))
                meta.append(attrs)

        f.visititems(_visit)

    return traces, meta

### ###

# define a class for generating synapses abstractly
class PreSimSynapseGenerator:
    def __init__(self, sim_dir):

        self.sim_dir = sim_dir

        self.segments = pd.read_csv(os.path.join(sim_dir, "segment_data.csv"))

        with open(os.path.join(sim_dir, "parameters.pickle"), 'rb') as file:
            self.parameters = pickle.load(file)

        if os.path.exists(os.path.join(self.sim_dir, "synapses.csv")):
            self.synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))
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
                # print("Mod files loaded successfully")
                pass
        except:
            # Already loaded
            pass 

    def generate_synapse_locations(self):
        for syn_properties_set, use_density, syn_mod, syn_params_choices in zip([self.parameters.exc_syn_properties, self.parameters.inh_syn_properties],
                                                            [self.parameters.exc_use_density, self.parameters.inh_use_density],
                                                            [self.parameters.exc_syn_mod, self.parameters.inh_syn_mod],
                                                            [self.parameters.exc_syn_params_choices, self.parameters.inh_syn_params_choices]):
            for input_source, synapse_properties in syn_properties_set.items():
                sec_type = synapse_properties['sec_type']
                self.logger.log(f"Generating synapses for {input_source.upper()} with properties: {synapse_properties}")
                segments_to_generate_on = self.get_segments_of_type(sec_type, self.segments) # get segments
                self.random_state = np.random.RandomState(self.parameters.numpy_random_state + synapse_properties['seed']['synapses'])
                np.random.seed(self.parameters.numpy_random_state + synapse_properties['seed']['synapses'])

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
                                        input_source = input_source,
                                        name = f"{synapse_properties['synapse_type']}_{input_source}"
                                        )
                self.synapses = pd.concat((self.synapses, synapses_this_sec_type), ignore_index=True)
        self.synapses = self.synapses.reset_index(drop=True) #@TODO: check if this is necessary and check self.synapses.
        self.synapses.to_csv(os.path.join(self.sim_dir, "synapses.csv"), index=False)
    def get_segments_of_type(self, sec_type, segments):
        return segments[segments['sec_type_precise'] == sec_type]
    
    def build_synapses_with_specs(self, segments_to_generate_on: pd.DataFrame, sec_type: str, synapse_type: str, use_density: bool,
                                  syn_number: int, syn_density: float, initial_weight_distribution: dict,
                                  release_probability_distribution: dict, name: str, syn_mod:str, input_source: str, syn_params_choices) -> pd.DataFrame:
        
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
            raise ValueError(f"Probabilities do not sum to 1 instead {segments_to_generate_on['probability'].sum()} for sec_type {sec_type}. Check your segment measurement for probabilities.\nsegments_to_generate_on: {segments_to_generate_on}")

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
                syn_params_made_choice = syn_params_choices['choices'][1] if segment['Distance'].values[0] > 100 else syn_params_choices['choices'][0] # perisomatic distance
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
                'input_source': input_source,
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

        fg_labels and pc_labels initialized to -1 (background).

        Loop Over All FGs in the Config:

            For each functional group (fg):

                matching_mask checks for synapses whose input_source matches the FG's input_source.

                If none match, continue to next FG.

                For matching synapses, check which are within the FG's spatial radius (in_fg_and_input).

                Assign FG index to those synapses.

                For each PC in the FG, repeat the spatial check and assign PC index.
        """
        fg_labels = -np.ones(len(synapses), dtype=int)
        pc_labels = -np.ones(len(synapses), dtype=int)
        # print("Assigning FGs for input_sources:", synapses['input_source'].unique())
        for fg_idx, fg in enumerate(clustering_config.get('functional_groups', [])):
            # print(f"FG {fg_idx} input_source: {fg.get('input_source')}")
            # Only assign synapses where input_source matches
            matching_mask = (synapses['input_source'].values == fg.get('input_source', None))
            # print(f"matching_mask: {matching_mask}, fg_idx: {fg_idx}, input_source: {fg.get('input_source', None)}")
            if not np.any(matching_mask):
                continue  # No synapses in this batch with this input_source
            
            fg_center = np.array(fg['center'])
            fg_radius = fg['radius']
            distances_to_fg = np.linalg.norm(coords - fg_center, axis=1)
            in_fg = distances_to_fg <= fg_radius

            # Only assign where both input_source AND spatial distance match
            in_fg_and_input = in_fg & matching_mask
            fg_labels[in_fg_and_input] = fg_idx

            pcs_cfg = fg.get('presynaptic_cells')
            if isinstance(pcs_cfg, dict) and pcs_cfg.get('mode') == 'dynamic':
                max_syn_per_pc = pcs_cfg.get('max_synapses_per_pc')
                if max_syn_per_pc is None:
                    raise ValueError(f"max_synapses_per_pc must be specified for dynamic mode in FG {fg_idx}.")

                locality = pcs_cfg.get('locality', None)  # None for legacy, 'nearest' for local batching

                idxs = np.where(in_fg_and_input)[0]
                n_syn = len(idxs)
                if n_syn == 0:
                    continue

                def sample_capacity():
                    """Return an integer >= 1 from max_syn_per_pc (callable/spec/number)."""
                    cap = max_syn_per_pc
                    if isinstance(max_syn_per_pc, dict) and 'dist' in max_syn_per_pc:
                        dist = max_syn_per_pc['dist']
                        if callable(dist):
                            cap = int(np.round(dist()))
                        elif isinstance(dist, dict):
                            kind = dist.get('kind')
                            if kind == 'uniform_int':
                                low = int(dist['low']); high = int(dist['high'])
                                try:
                                    cap = int(np.random.default_rng().integers(low, high + 1))
                                except AttributeError:
                                    cap = int(np.random.randint(low, high + 1))  # inclusive
                            elif kind == 'truncnorm_int':
                                m = float(dist['mean']); sd = float(dist['sd'])
                                low = int(dist['low']);  high = int(dist['high'])
                                val = int(np.round(np.random.normal(loc=m, scale=sd)))
                                cap = min(max(val, low), high)
                            else:
                                raise ValueError(f"Unknown divergence dist spec kind: {kind}")
                        else:
                            raise ValueError(f"'dist' must be callable or spec-dict, got {type(dist)}")
                    elif isinstance(max_syn_per_pc, (int, float)):
                        cap = int(round(max_syn_per_pc))
                    else:
                        # fall back to 1 if truly odd input to avoid infinite loop
                        cap = 1
                    return max(cap, 1)

                if locality == 'nearest':
                    # Spatially local batching
                    # Use coords only for the synapses in this FG
                    fg_coords = coords[idxs]
                    remaining = idxs.copy()
                    rem_coords = fg_coords.copy()

                    pc_idx_local = 0
                    while len(remaining) > 0:
                        # Seed = farthest from centroid (spreads clusters a bit)
                        centroid = rem_coords.mean(axis=0)
                        d2c = np.linalg.norm(rem_coords - centroid, axis=1)
                        seed_i = int(np.argmax(d2c))
                        seed_coord = rem_coords[seed_i:seed_i+1]

                        dists = np.linalg.norm(rem_coords - seed_coord, axis=1)
                        order = np.argsort(dists)

                        cap = sample_capacity()
                        take = order[:min(cap, len(order))]

                        pc_labels[remaining[take]] = pc_idx_local
                        pc_idx_local += 1

                        keep_mask = np.ones(len(remaining), dtype=bool)
                        keep_mask[take] = False
                        remaining = remaining[keep_mask]
                        rem_coords = rem_coords[keep_mask]

                    assert np.all(pc_labels[idxs] != -1), f"Some synapses in FG {fg_idx} weren't assigned a PC!"
                else:
                    # Legacy: sequential chunks (not spatially local)
                    idxs = np.sort(idxs)
                    pc_idx_seq = 0
                    cur = 0
                    while cur < n_syn:
                        cap = sample_capacity()
                        start = cur
                        end = min(cur + cap, n_syn)
                        pc_labels[idxs[start:end]] = pc_idx_seq
                        pc_idx_seq += 1
                        cur = end
                    assert np.all(pc_labels[idxs] != -1), f"Some synapses in FG {fg_idx} weren't assigned a PC!"

            else: # assign pcs statically and specifically from config
                for pc_idx, pc in enumerate(fg.get('presynaptic_cells', [])):
                    pc_center = np.array(pc['center'])
                    pc_radius = pc['radius']
                    distances_to_pc = np.linalg.norm(coords - pc_center, axis=1)
                    in_pc = (distances_to_pc <= pc_radius) & in_fg_and_input
                    pc_labels[in_pc] = pc_idx

        synapses['functional_group'] = fg_labels
        synapses['presynaptic_cell'] = pc_labels

        # Check if all synapses have been assigned to a functional group
        fg_input_sources = set(fg.get('input_source', None) for fg in clustering_config.get('functional_groups', []))
        syn_input_sources = set(synapses['input_source'].unique())
        assert len(syn_input_sources) == 1, f"Expected only one input_source in this batch, got: {syn_input_sources} synapses: {synapses}"
        assert syn_input_sources.issubset(fg_input_sources), f"Input source(s) in synapses {syn_input_sources} not present in clustering_config {fg_input_sources}"

        return synapses

    def generate_spike_trains_for_cell_assemblies(
        self,
        synapses: pd.DataFrame,
        parameters,
        h_tstop: int,
        input_source: str,
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

        props = getattr(parameters, f"{synapse_type}_syn_properties")[input_source]
        spike_train_mode = props.get('spike_train_mode', 'standard')
        mean_fr_dist = partial(
            props['mean_firing_rate_distribution']['function'],
            **props['mean_firing_rate_distribution']['params'], size=1
        )

        allowed_modes = {'standard', 'pink_noise', 'rhythmic', 'delay'}
        if isinstance(spike_train_mode, (list, tuple)):
            invalid = [m for m in spike_train_mode if m not in allowed_modes]
            if invalid:
                raise NotImplementedError(
                    f"spike_train_mode(s) {invalid} not implemented. Allowed: {sorted(allowed_modes)}"
                )
        else:
            if spike_train_mode not in allowed_modes:
                raise NotImplementedError(
                    f"spike_train_mode '{spike_train_mode}' is not implemented. Allowed: {sorted(allowed_modes)}"
                )

        clustering_config = getattr(parameters, f"{synapse_type}_clustering", {}).get(input_source, {})
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
            # print(f"[delay mode] FG={fg_id}, sec_type={input_source}, n_trains={len(trains)}, ex: {[len(t) for t in trains[:3]]}")


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
            # for idx, arr in enumerate(trains):
            #     print(f"train {idx} arr: {arr}: type={type(arr)}, hasattr_len={hasattr(arr,'__len__')}")
            #     if arr is None:
            #         print(f"Warning: train {idx} is None!")
            #     elif not hasattr(arr, '__len__'):
            #         print(f"Error: train {idx} has no length!")
            #     elif len(arr) == 0:
            #         print(f"Warning: train {idx} is empty!")
            # out_trains = [arr for arr in trains if arr is not None and hasattr(arr, "__len__") and len(arr) > 0]
            out_trains = [
                arr for arr in trains
                if arr is not None
                and hasattr(arr, '__len__')
                and not (isinstance(arr, np.ndarray) and arr.ndim == 0)
                and len(arr) > 0
            ]
            if not out_trains:
                raise ValueError(
                    f"No spike trains found for delay reference with mask: "
                    f"ref_synapse_type={ref_synapse_type}, ref_sec_type={ref_sec_type}, "
                    f"ref_fg_id={ref_fg_id}, ref_pc_id={ref_pc_id}")
            return out_trains

        def generate_fg_trace(fg_id=None, fg=None):
            """
            Build FG trace by composing the *source-level* base mode
            with an optional FG-level extra modulation (usually rhythmic).
            The FG should never override delay.
            """
            # 1) normalize the source-level mode into an ordered list
            if isinstance(spike_train_mode, (list, tuple)):
                base_modes = list(spike_train_mode)
            else:
                base_modes = [spike_train_mode]

            # 2) allow a single *additional* FG modulation (if present and not duplicate)
            fg_mod = (fg or {}).get('modulation_mode', None)
            mode_list = base_modes[:]  # e.g., ['delay','rhythmic'] for inh
            if fg_mod and fg_mod not in mode_list:
                mode_list.append(fg_mod)

            fg_trace = None
            for idx, mod in enumerate(mode_list):
                if idx == 0:
                    # BASE modulation
                    if mod == 'standard':
                        fg_trace = np.ones(h_tstop)
                    elif mod == 'pink_noise':
                        fg_trace = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
                            num=h_tstop, random_state=random_state
                        )
                        m = np.mean(fg_trace)
                        if m == 0:
                            raise ValueError("Mean of pink-noise trace is zero; cannot normalize.")
                        fg_trace = fg_trace / m
                    elif mod == 'delay':
                        shift = delay_config.get('delay_shift', None)
                        if shift is None:
                            raise ValueError("delay_shift must be specified for 'delay' mode.")
                        ref_synapse_type = delay_config.get('ref_synapse_type', 'exc')
                        ref_sec_type     = delay_config.get('ref_sec_type', input_source)
                        ref_fg_id        = delay_config.get('ref_fg_id', fg_id)
                        ref_pc_id        = delay_config.get('ref_pc_id', None)
                        trains = collect_reference_spike_trains(
                            ref_synapse_type, ref_sec_type, ref_fg_id, ref_pc_id
                        )
                        if not all(isinstance(t,(np.ndarray,list)) and len(t)>0 for t in trains):
                            raise ValueError("Delay refs must be non-empty arrays/lists.")
                        fg_trace = PoissonTrainGenerator.generate_lambdas_by_delaying(h_tstop, trains)
                    elif mod == 'rhythmic':
                        fg_trace = np.ones(h_tstop)
                        freq  = props.get('rhythmic_frequency', None)
                        depth = props.get('rhythmic_depth', None)
                        if freq is None or depth is None:
                            raise ValueError("Set rhythmic_frequency and rhythmic_depth.")
                        fg_trace = PoissonTrainGenerator.rhythmic_modulation(fg_trace, freq, depth, 1)
                    else:
                        raise NotImplementedError(f"Unrecognized base mode: {mod}")
                else:
                    # SECONDARY modulation: keep this tight (support rhythmic only)
                    if mod == 'rhythmic':
                        freq  = props.get('rhythmic_frequency', None)
                        depth = props.get('rhythmic_depth', None)
                        if freq is None or depth is None:
                            raise ValueError("Set rhythmic_frequency and rhythmic_depth.")
                        fg_trace = PoissonTrainGenerator.rhythmic_modulation(fg_trace, freq, depth, 1)
                    else:
                        # don’t allow 'delay' or 'pink_noise' as a *second* layer
                        continue
            return fg_trace

        unique_fgs = np.unique(synapses['functional_group'])
        for fg_id in unique_fgs:
            fg_mask = (synapses['functional_group'] == fg_id)
            fg = None
            if n_fg > 0 and fg_id >= 0:
                fg = clustering_config['functional_groups'][int(fg_id)]
            fg_trace = generate_fg_trace(fg_id=fg_id, fg=fg)
            if fg_trace is None or not isinstance(fg_trace, np.ndarray) or np.any(np.isnan(fg_trace)):
                raise ValueError(f"fg_trace is not valid for FG {fg_id}, got: {fg_trace}")
            if fg_id >= 0:
                fg_traces_store[(synapse_type, input_source, int(fg_id))] = fg_trace

            pcs_in_fg = np.unique(synapses.loc[fg_mask, 'presynaptic_cell'])
            for pc_id in pcs_in_fg:
                pc_mask = fg_mask & (synapses['presynaptic_cell'] == pc_id)
                if pc_id == -1:
                    for idx in synapses[pc_mask].index:
                        mean_fr = float(mean_fr_dist(size=1)) + props.get('fr_shift', 0)
                        if mean_fr <= 0:
                            mean_fr = 0
                        elif not np.isfinite(mean_fr):
                            raise ValueError(f"Background mean firing rate is infinite: {mean_fr}")
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
                mean_fr = float(mean_fr_dist(size=1)) + props.get('fr_shift', 0)
                if not np.isfinite(mean_fr) or mean_fr <= 0:
                    raise ValueError(f"PC mean firing rate is not positive: {mean_fr}")
                # Standard: shift mean, Delay: use fg_trace directly (already population-based)
                has_delay = (spike_train_mode == 'delay') or (
                    isinstance(spike_train_mode, (list, tuple)) and 'delay' in spike_train_mode
                )
                pc_trace = PoissonTrainGenerator.shift_mean_of_lambdas(fg_trace, mean_fr)
                if pc_trace is None or not isinstance(pc_trace, np.ndarray) or np.any(np.isnan(pc_trace)):
                    raise ValueError(f"pc_trace is not valid for FG {fg_id} PC {pc_id}, got: {pc_trace}")
                spike_train = PoissonTrainGenerator.generate_spike_train(pc_trace, random_state)
                for idx in synapses[pc_mask].index:
                    synapses.at[idx, 'spike_train'] = spike_train.spike_times
                    synapses.at[idx, 'pc_mean_firing_rate'] = mean_fr
                pc_spike_trains_store[(synapse_type, input_source, fg_id, pc_id)] = spike_train
        return synapses, fg_traces_store, pc_spike_trains_store

    def generate_spike_trains_for_synapses(self):
        """
        Main entrypoint: assigns FG/PC and generates spike trains for all synapses.
        Ensures all excitatory spike trains are generated before any delayed inhibition.

        Note: functional groups and presynaptic cells are assigned based on the synapse's input_source. The fg id and pc id are within input_source. (i.e. they reset between input_sources).
        This means that different input_sources can have the same fg id and pc id, but they will not be the same functional group or presynaptic cell.
        """
        synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))
        # print(f"unique input sources in synapses before generating spikes: {synapses['input_source'].unique()}")

        segments = self.segments
        parameters = self.parameters
        h_tstop = self.parameters.h_tstop

        columns = ['pc_0', 'pc_1', 'pc_2']
        synapses_with_seg_info = synapses.merge(
            segments, on='seg_id', how='left', suffixes=('','_seg'))
        synapse_coords = synapses_with_seg_info[columns].values

        # synapses['input_source'] = ''   # Initialize new column

        # Pass 1: Assign all FG/PCs for ALL synapses
        all_fg_labels = np.full(len(synapses), -1, dtype=int) # default -1 for background/not assigned
        all_pc_labels = np.full(len(synapses), -1, dtype=int) # default -1 for background/not assigned
        # print(f"Assigning functional groups and presynaptic cells for synapses")
        for synapse_type in ['exc', 'inh']:
            properties_set = getattr(parameters, f"{synapse_type}_syn_properties") # get synapse property config for this type
            clustering_dict = getattr(parameters, f"{synapse_type}_clustering", {}) # get synapse clustering config for this type
            for input_source, props in properties_set.items(): # iterate over all input sources for this synapse type
                sec_type = props['sec_type'] # get the section type for this input source
                group = clustering_dict.get(input_source) or clustering_dict.get(sec_type, {}) # Find the group for this sec_type in the clustering config

                matched_fgs = []
                # Find the functional groups whose input_source matches the input_source, hanfle multiple functional groups per input_source
                for fg in group.get('functional_groups', []):
                    if fg['input_source'] == input_source:
                        matched_fgs.append(fg)
                if not matched_fgs:
                    # print(f"Warning: No FG found for synapse_type={synapse_type}, sec_type={sec_type}, input_source={input_source}")
                    continue
                syn_mask = ( # filter synapses for this synapse type and input source
                    synapses['name'].str.contains(synapse_type, na=False)
                    & (synapses['input_source'] == input_source)
                )
                if syn_mask.sum() == 0:
                    # print(f"Warning: No synapses found for {synapse_type} {input_source} in synapses.csv. Skipping.")
                    continue
                coords_this_type = synapse_coords[syn_mask]
                # print(f"Assigning for {synapse_type}, sec_type={sec_type}, input_source={input_source}")
                # print(f"Functional group input_source: {[fg['input_source'] for fg in matched_fgs]}")
                # print(f"Synapses input_sources (unique): {synapses.loc[syn_mask, 'input_source'].unique()}")
                # print(f"Number of synapses: {syn_mask.sum()}")
                clustering_for_assignment = {'functional_groups': matched_fgs}
                assigned_synapses = self.assign_synapses_to_cell_assemblies( # assign synapses to functional groups and presynaptic cells
                    synapses.loc[syn_mask].copy(), coords_this_type, clustering_for_assignment
                )
                all_fg_labels[syn_mask] = assigned_synapses['functional_group'].values
                all_pc_labels[syn_mask] = assigned_synapses['presynaptic_cell'].values
                # synapses.loc[syn_mask, 'input_source'] = input_source # should already be set, but just in case
        synapses['functional_group'] = all_fg_labels
        synapses['presynaptic_cell'] = all_pc_labels

        all_synapses_full = synapses.copy()

        fg_traces_store = {}
        pc_spike_trains_store = {}

        # --- Pass 2: non-delayed first ---
        for synapse_type in ['exc', 'inh']:
            properties_set = getattr(parameters, f"{synapse_type}_syn_properties")
            for input_source, props in properties_set.items():
                mode = props.get('spike_train_mode', 'standard')
                has_delay = (mode == 'delay') or (isinstance(mode, (list, tuple)) and 'delay' in mode)
                if has_delay:
                    continue  # defer to Pass 3
                syn_mask = (
                    synapses['name'].str.contains(synapse_type, na=False) &
                    (synapses['input_source'] == input_source)
                )
                assigned_synapses = synapses.loc[syn_mask].copy()
                seed = props['seed']['synapses']
                random_state = np.random.RandomState(parameters.numpy_random_state + seed)
                assigned_synapses, fg_traces_store, pc_spike_trains_store = self.generate_spike_trains_for_cell_assemblies(
                    assigned_synapses, parameters, h_tstop, input_source, synapse_type, random_state,
                    fg_traces_store=fg_traces_store, pc_spike_trains_store=pc_spike_trains_store,
                    all_synapses_full=all_synapses_full
                )
                for col in ['spike_train', 'pc_mean_firing_rate', 'functional_group', 'presynaptic_cell']:
                    synapses.loc[assigned_synapses.index, col] = assigned_synapses[col]
                all_synapses_full.loc[assigned_synapses.index, 'spike_train'] = assigned_synapses['spike_train']

        # Prepare arrays for delay ref
        all_synapses_full['spike_train'] = all_synapses_full['spike_train'].apply(deserialize_spike_train)

        # --- Pass 3: delayed sources ---
        for synapse_type in ['exc', 'inh']:
            properties_set = getattr(parameters, f"{synapse_type}_syn_properties")
            for input_source, props in properties_set.items():
                mode = props.get('spike_train_mode', 'standard')
                has_delay = (mode == 'delay') or (isinstance(mode, (list, tuple)) and 'delay' in mode)
                if not has_delay:
                    continue
                syn_mask = (
                    synapses['name'].str.contains(synapse_type, na=False)
                    & (synapses['input_source'] == input_source)
                )
                assigned_synapses = synapses.loc[syn_mask].copy()
                seed = props['seed']['synapses']
                random_state = np.random.RandomState(parameters.numpy_random_state + seed)
                # NOTE: pass input_source here (see bug #3)
                assigned_synapses, fg_traces_store, pc_spike_trains_store = self.generate_spike_trains_for_cell_assemblies(
                    assigned_synapses, parameters, h_tstop, input_source, synapse_type, random_state,
                    fg_traces_store=fg_traces_store, pc_spike_trains_store=pc_spike_trains_store,
                    all_synapses_full=all_synapses_full
                )
                for col in ['spike_train', 'pc_mean_firing_rate', 'functional_group', 'presynaptic_cell']:
                    synapses.loc[assigned_synapses.index, col] = assigned_synapses[col]
                all_synapses_full.loc[assigned_synapses.index, 'spike_train'] = assigned_synapses['spike_train']

        synapses['spike_train'] = synapses['spike_train'].apply(serialize_spike_train)
        self.synapses = synapses.reset_index(drop=True)  # reset index after all operations
        self.synapses.to_csv(os.path.join(self.sim_dir, "synapses.csv"), index=False)
        save_fg_traces_h5(fg_traces_store, os.path.join(self.sim_dir, "fg_traces.h5"))



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

## The following functions are utilities later added to AA_pre_sim and have yet to be fully integrated into the class above.
def replace_N_synapses(sim_dir, N):
    """
    In <sim_dir>/synapses.csv, pick N rows and reset:
      - spike_train -> []          (empty; written as '[]' string)
      - pc_mean_firing_rate -> 0
      - presynaptic_cell (or 'presynaptifc_cell') -> -2
      - functional_group -> -2
    """
    synapses_path = os.path.join(sim_dir, "synapses.csv")
    synapses = pd.read_csv(synapses_path)

    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if N > len(synapses):
        raise ValueError(f"N ({N}) is greater than the number of synapses ({len(synapses)}).")

    # --- column setup (handle misspelling, create if absent) ---
    if "presynaptic_cell" in synapses.columns:
        presyn_col = "presynaptic_cell"
    else:
        presyn_col = "presynaptic_cell"
        synapses[presyn_col] = np.nan

    for col in ("pc_mean_firing_rate", "functional_group", "spike_train"):
        if col not in synapses.columns:
            synapses[col] = np.nan

    if "needs_new_spike_train" not in synapses.columns:
        synapses["needs_new_spike_train"] = False

    # --- pick rows and update ---
    rng = np.random.default_rng(42)
    picked_pos = rng.choice(len(synapses), size=N, replace=False)     # integer positions
    picked_idx = synapses.index[picked_pos]                            # index labels

    # scalar/broadcast-safe assignments
    synapses.loc[picked_idx, "pc_mean_firing_rate"] = 0
    synapses.loc[picked_idx, "functional_group"]    = -2
    synapses.loc[picked_idx, presyn_col]            = -2
    synapses.loc[picked_idx, "needs_new_spike_train"] = True

    # --- safe spike_train assignment ---
    # Option A (simple & CSV-friendly): write string '[]' and avoid shape issues
    synapses.loc[picked_idx, "spike_train"] = "[]"

    # If you truly want Python empty lists in-memory instead, use an index-aligned Series:
    # synapses.loc[picked_idx, "spike_train"] = pd.Series([[]]*len(picked_idx), index=picked_idx, dtype="object")

    synapses.to_csv(synapses_path, index=False)

from Modules.post_sim import analysis
def update_spike_trains(
        syn_type: str, 
        rate: float, 
        df: pd.DataFrame, 
        sim_duration_ms: int, 
        base_seed: int = 12345, 
        spike_train_modulation: str = 'constant', 
        modulation_params: dict = {}, 
        region:str = '') -> pd.DataFrame:
    if not 'needs_new_spike_train' in df.columns:
        # raise ValueError("DataFrame must contain 'needs_new_spike_train' column to update spike trains.")
        Warning("DataFrame must contain 'needs_new_spike_train' column to update spike trains.")  
        return df
    
    # Ensure array-capable column
    if df['spike_train'].dtype != 'object':
        df['spike_train'] = df['spike_train'].astype('object')

    # Mask: name has syn_type, -2 gate, and needs new train
    # Tighter, cheaper mask than .str.contains: match prefix "exc_" / "inh_"
    type_mask = df['name'].str.startswith(f'{syn_type}_', na=False) # df['name'].str.contains(syn_type, case=False, na=False)
    region_mask = df['name'].str.contains(region, case=False, na=False)
    # print(region_mask)
    mask = (
        type_mask
        & region_mask
        & df['needs_new_spike_train'].astype(bool)
        & (df['functional_group'] == -2)
        & (df['presynaptic_cell'] == -2)
    )

    idx = df.index[mask]
    if idx.empty:
        return df  # nothing to do

    # Reuse a single λ vector across all rows (uniform rate)
    if not np.isfinite(rate) or rate <= 0:
        lambda_vec = None  # means: return empty trains below
    else:
        lambda_vec = np.full(sim_duration_ms, float(rate), dtype=float)

    if spike_train_modulation == 'rhythmic':
        if lambda_vec is None:
            raise(ValueError(f"lambda_vec: {lambda_vec} should not be None"))
        lambda_vec_to_use = PoissonTrainGenerator.rhythmic_modulation(
            lambdas=lambda_vec,
            frequency=modulation_params.get('frequency'),
            depth_of_mod=modulation_params.get('depth_of_mod'),
            delta_t=modulation_params.get('delta_t', 0.1)
        )
    elif spike_train_modulation == 'constant':
        lambda_vec_to_use = lambda_vec
    else:
        raise ValueError(f"Notimplemented spike_train_modulation: {spike_train_modulation} choose 'rhythmic' or 'constant'")
    # Stable per-row seeds:
    # - Fast: base_seed + integer index (good if index is stable)
    # - If you need stability across reindexing, hash a stable key instead (slower).
    seeds = base_seed + pd.Index(range(len(idx))).to_numpy(dtype=np.int64)


    def build_train(seed: int):
        # keep empty if rate <= 0 or NaN
        if lambda_vec_to_use is None:
            return np.array([], dtype=np.int32)
        rng = np.random.RandomState(int(seed))
        st = PoissonTrainGenerator.generate_spike_train(lambda_vec_to_use, rng)
        # print(f"Generated spike train {st.spike_times} with seed {seed} and rate {rate}")
        return st.spike_times#.astype(int)  # only need the spike_times

    # List comprehension is usually fastest in pure Python for this pattern
    new_trains = [build_train(s) for s in seeds]

    # Aligned assignment
    df.loc[idx, 'spike_train'] = pd.Series(new_trains, index=idx, dtype='object')
    df.loc[idx, 'needs_new_spike_train'] = False
    return df

def update_spike_trains_for_sim(sim_dir, inh_bg_rate, exc_bg_rate):
    synapses = pd.read_csv(os.path.join(sim_dir, "synapses.csv"))
    params = analysis.DataReader.load_parameters(sim_dir)
    # print(params)
    # print(params.inh_syn_properties)
    # print(params.inh_syn_properties['perisomatic'])
    # print(params.inh_syn_properties['perisomatic']['rhythmic_depth'])
    h_tstop = params.h_tstop
    # # not rhythmic background
    # synapses = update_spike_trains('inh', inh_bg_rate, synapses, sim_duration_ms=h_tstop, base_seed=12345,
    #                             spike_train_modulation='constant',
    #                             modulation_params={},
    #                             region='')
    # rhythmic background
    synapses = update_spike_trains('inh', inh_bg_rate, synapses, sim_duration_ms=h_tstop, base_seed=12345,
                            spike_train_modulation='rhythmic',
                            modulation_params={'frequency': params.inh_syn_properties['perisomatic']['rhythmic_frequency'],
                                               'depth_of_mod': params.inh_syn_properties['perisomatic']['rhythmic_depth'],
                                               'delta_t': 0.1},
                            region='perisomatic')
    synapses = update_spike_trains('inh', inh_bg_rate, synapses, sim_duration_ms=h_tstop, base_seed=12345,
                            spike_train_modulation='rhythmic',
                            modulation_params={'frequency': params.inh_syn_properties['distal_basal']['rhythmic_frequency'],
                                               'depth_of_mod': params.inh_syn_properties['distal_basal']['rhythmic_depth'],
                                               'delta_t': 0.1},
                            region='distal_basal|tuft|trunk|nexus|oblique')
    
    # Ensure array-capable column
    synapses = update_spike_trains('exc', exc_bg_rate, synapses, sim_duration_ms=h_tstop, base_seed=12345)
    synapses.to_csv(os.path.join(sim_dir, "synapses.csv"), index=False)