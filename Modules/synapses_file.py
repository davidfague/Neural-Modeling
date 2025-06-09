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
            # add row to dataframe
            self.synapses = pd.concat((self.synapses, pd.DataFrame({
                'name': f"{name}_{_}",
                'modfile': syn_mod,
                # 'initW': syn_params_this_syn["initW"],
                # 'gmax': syn_params_this_syn["gmax"],
                # 'release_probability': syn_params_this_syn["release_probability"],
                'P_0': syn_params_this_syn["P_0"],
                'initW': syn_params_this_syn["initW"],
                'cell2cell_type': syn_params_this_syn["syn_params_choice"],

                'seg_id': syn_params_this_syn["seg_id"], 
                **gbar_params,
                # **syn_params_this_syn 
                # #TODO: (SHOULD BE DONE) use a string to indicate which syn_params to use instead of storing all of them in the DataFrame. Also will need to pull out the syn_params that were added to syn_params in this snippet (such as initW, location, release_probability.) and give them their own column.
            }, index=[0])), ignore_index=True)

    def generate_spike_trains_for_synapses(self)->None:
        synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))

        synapses_with_seg_info = synapses.merge( # TODO: alternative could be used to save memory.
            self.segments, 
            on='seg_id', 
            how='left',               # carry along all synapses even if a seg_id is missing
            suffixes=('','_seg')      # e.g. if both have a 'length' column
        )

        columns = ['pc_0', 'pc_1', 'pc_2']

        # get the coordinates of the synapses
        synapse_coords = synapses_with_seg_info[columns].values

        random_state = np.random.RandomState(self.parameters.precell_spikes_seeds['inh'])
        np.random.seed(self.parameters.precell_spikes_seeds['inh'])

        synapses['spike_train'] = np.empty(len(synapses), dtype=object)  # gives dtype=object
        synapses['pc_mean_firing_rate'] = np.nan            # dtype float, which is fine
        synapses['functional_group'] = np.nan               # track which functional group each synapse belongs to
        synapses['presynaptic_cell'] = np.nan               # track which presynaptic cell each synapse belongs to

        # First, generate background firing rate profile for all synapses
        background_firing_rate_timecourse = np.ones(self.parameters.h_tstop)

        # Process both excitatory and inhibitory synapses
        for synapse_type in ['exc', 'inh']:
            # Get the appropriate random state for this synapse type
            random_state = np.random.RandomState(getattr(self.parameters, f"{synapse_type}_syn_properties")[list(getattr(self.parameters, f"{synapse_type}_syn_properties").keys())[0]]['seed']['synapses'])
            np.random.seed(getattr(self.parameters, f"{synapse_type}_syn_properties")[list(getattr(self.parameters, f"{synapse_type}_syn_properties").keys())[0]]['seed']['synapses'])

            for cluster_sec_type in getattr(self.parameters, f"{synapse_type}_syn_properties").keys():
                # Get the mean firing rate distribution for this synapse type and section type
                mean_firing_rate_distribution = getattr(self.parameters, f"{synapse_type}_syn_properties")[cluster_sec_type]['mean_firing_rate_distribution']
                mean_firing_rate_distribution = partial(mean_firing_rate_distribution['function'], **mean_firing_rate_distribution['params'], size=1)

                # Get synapses of this type and section
                synapses_this_sec_and_syn_type = (
                    synapses['name'].str.contains(synapse_type, na=False) 
                    & synapses['name'].str.contains(cluster_sec_type, na=False)
                )
                
                # Get coordinates for these synapses
                coords_this_type = synapse_coords[synapses_this_sec_and_syn_type]
                
                # Get clustering configuration for this synapse type and section
                clustering_config = getattr(self.parameters, f"{synapse_type}_clustering", {}).get(cluster_sec_type, {})
                
                # Initialize functional group and presynaptic cell labels (-1 for background)
                functional_group_labels = -np.ones(len(coords_this_type))
                presynaptic_cell_labels = -np.ones(len(coords_this_type))
                
                # Process each functional group
                for fg_idx, fg in enumerate(clustering_config.get('functional_groups', [])):
                    fg_center = np.array(fg['center'])
                    fg_radius = fg['radius']
                    
                    # Calculate distances from functional group center
                    distances_to_fg = np.sqrt(np.sum((coords_this_type - fg_center)**2, axis=1))
                    
                    # Get synapses within this functional group
                    fg_mask = distances_to_fg <= fg_radius
                    functional_group_labels[fg_mask] = fg_idx
                    
                    # Process each presynaptic cell in this functional group
                    for pc_idx, pc in enumerate(fg.get('presynaptic_cells', [])):
                        pc_center = np.array(pc['center']) #+ fg_center  # PC center can be relative to FG center
                        pc_radius = pc['radius']
                        
                        # Calculate distances from presynaptic cell center
                        distances_to_pc = np.sqrt(np.sum((coords_this_type - pc_center)**2, axis=1))


                        # Get synapses within this presynaptic cell
                        pc_mask = (distances_to_pc <= pc_radius) & fg_mask
                        if not np.any(pc_mask):  # Warn if no synapses found in this PC
                            # self.logger.log(f"Warning: No synapses found in presynaptic cell {pc_idx} of functional group {fg_idx} for synapse type {synapse_type} and section {cluster_sec_type}.")
                            print(f"Warning: No synapses found in presynaptic cell {pc_idx} of functional group {fg_idx} for synapse type {synapse_type} and section {cluster_sec_type}.")
                            print(f"Coordinates of synapses in this section and synapse type: {coords_this_type[fg_mask]}")
                            print(f"Coordinates of presynaptic cell center: {pc_center}, radius: {pc_radius}")

                        presynaptic_cell_labels[pc_mask] = pc_idx
                
                # Assign functional groups and presynaptic cells to synapses
                synapses.loc[synapses_this_sec_and_syn_type, 'functional_group'] = functional_group_labels
                synapses.loc[synapses_this_sec_and_syn_type, 'presynaptic_cell'] = presynaptic_cell_labels
                
                # For each functional group (including noise points labeled as -1)
                unique_fgs = np.unique(functional_group_labels)
                for fg_id in unique_fgs:
                    # Get synapses in this functional group
                    fg_mask = (functional_group_labels == fg_id)
                    
                    if fg_id == -1:  # Background synapses
                        # Generate independent spike trains for background synapses
                        background_synapses = synapses[synapses_this_sec_and_syn_type][fg_mask]
                        for idx in background_synapses.index:
                            pc_mean_firing_rate = mean_firing_rate_distribution(size=1)
                            pc_firing_rate_timecourse = PoissonTrainGenerator.shift_mean_of_lambdas(
                                lambdas=background_firing_rate_timecourse, 
                                desired_mean=pc_mean_firing_rate
                            )
                            spike_train = PoissonTrainGenerator.generate_spike_train(
                                lambdas=pc_firing_rate_timecourse, 
                                random_state=random_state
                            )
                            
                            synapses.at[idx, 'spike_train'] = np.array(spike_train.spike_times)
                            synapses.at[idx, 'pc_mean_firing_rate'] = pc_mean_firing_rate
                    else:
                        # Generate a modulatory trace for this functional group
                        fg_firing_rate_profile = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
                            num=self.parameters.h_tstop,
                            random_state=random_state
                        )
                        fg_firing_rate_profile = fg_firing_rate_profile / np.mean(fg_firing_rate_profile)  # Normalize around 1
                        
                        # For each presynaptic cell in this functional group
                        unique_pcs = np.unique(presynaptic_cell_labels[fg_mask])
                        for pc_id in unique_pcs:
                            if pc_id == -1:  # These are synapses in the FG but not in any PC - treat as background
                                # Get these synapses
                                unassigned_synapses = synapses[synapses_this_sec_and_syn_type][(functional_group_labels == fg_id) & (presynaptic_cell_labels == -1)]
                                # Generate independent spike trains for them
                                for idx in unassigned_synapses.index:
                                    pc_mean_firing_rate = mean_firing_rate_distribution(size=1)
                                    pc_firing_rate_timecourse = PoissonTrainGenerator.shift_mean_of_lambdas(
                                        lambdas=background_firing_rate_timecourse, 
                                        desired_mean=pc_mean_firing_rate
                                    )
                                    spike_train = PoissonTrainGenerator.generate_spike_train(
                                        lambdas=pc_firing_rate_timecourse, 
                                        random_state=random_state
                                    )
                                    
                                    synapses.at[idx, 'spike_train'] = np.array(spike_train.spike_times)
                                    synapses.at[idx, 'pc_mean_firing_rate'] = pc_mean_firing_rate
                                continue
                                
                            # Get synapses in this presynaptic cell
                            pc_mask = (presynaptic_cell_labels == pc_id) & fg_mask
                            
                            # Sample a mean firing rate for this presynaptic cell
                            pc_mean_fr = mean_firing_rate_distribution(size=1)
                            
                            # Generate a base spike train for this presynaptic cell
                            base_firing_rate_timecourse = PoissonTrainGenerator.shift_mean_of_lambdas(
                                lambdas=fg_firing_rate_profile,
                                desired_mean=pc_mean_fr
                            )
                            base_spike_train = PoissonTrainGenerator.generate_spike_train(
                                lambdas=base_firing_rate_timecourse,
                                random_state=random_state
                            )
                            
                            # Assign the same spike train to all synapses in this presynaptic cell
                            cluster_synapses = synapses[synapses_this_sec_and_syn_type][pc_mask]
                            for idx in cluster_synapses.index:
                                synapses.at[idx, 'spike_train'] = np.array(base_spike_train.spike_times)
                                synapses.at[idx, 'pc_mean_firing_rate'] = pc_mean_fr

        # Check for any unprocessed synapses
        remaining_mask = synapses['spike_train'].isna()
        if remaining_mask.any():
            raise ValueError(f"Some synapses were not processed in the above loops: {synapses[remaining_mask]['name'].unique()}. Check your provided synapse names and parameters: {getattr(self.parameters, f'{synapse_type}_syn_properties')}")

        synapses.to_csv(os.path.join(self.sim_dir, "synapses.csv"), index=False)

    def build_synapses_onto_cell_obj(self)->CellModel:
        ## assumes already have parameters and sim_dir defined, even logger (use run_on_all_sims)

        # build synapses from synapses csv from simulation folder
        synapses = pd.read_csv(os.path.join(self.sim_dir, "synapses.csv"))

        # convert each string back to an array
        synapses["spike_train"] = synapses["spike_train"].apply(
            lambda s: np.fromstring(s.strip("[]"), sep=" ")
        )

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
#     # of synapses_with_seg_info['sec_type_precise'] == cluster_sec_type
#     cluster_indices = np.where(synapses_with_seg_info['sec_type_precise'] == cluster_sec_type)[0][cluster_indices] #TODO: CHECK

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