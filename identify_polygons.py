from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.core.structure import Structure, Lattice
import numpy as np
import pandas as pd
import time as time
import matplotlib.pyplot as plt
import pickle 
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
import networkx as nx
import os
from scipy.spatial.distance import cdist
import multiprocessing as mp
import logging

def stream_lammps_frames(filename):
    """Generator that yields one snapshot at a time without storing all in memory."""
    with open(filename, "r") as f:
        dump_iter = parse_lammps_dumps(f)
        for dump in dump_iter:
            yield dump  # each is a LammpsDump instance

def snapshot_dump_str(df, atom_dict):
    """
    Description:
        This function converts a dataframe containing the atom data at each timestep
        to a string in the lammps dump format.
    Args:
        df (dataframe): A dataframe containing the atom data at each timestep.
        atom_dict (dict): A dictionary containing the box bounds and tilt.
    Returns:
        dump_str (str): A string in the lammps dump format.
    """
    keys=df.keys()
    atom_data_str=str('ITEM: ATOMS ')
    for key in keys:
        atom_data_str+=str(key)+' '
    atom_data_str+='\n'
    # dump_str=''
    dump_str='ITEM: TIMESTEP\n'
    dump_str+=str(int(atom_dict['timestep']))+'\n'
    dump_str+='ITEM: NUMBER OF ATOMS\n'
    dump_str+=str(int(len(df)))+'\n'
    dump_str+='ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
    dump_str+=str(atom_dict['box']['bounds'][0][0])+' '+str(atom_dict['box']['bounds'][0][1])+' '
    if atom_dict['box']['tilt'] is None:
        dump_str+=str(0)+'\n'
    else:
        dump_str+=str(atom_dict['box']['tilt'][0])+'\n'
    dump_str+=str(atom_dict['box']['bounds'][1][0])+' '+str(atom_dict['box']['bounds'][1][1])+' '
    if atom_dict['box']['tilt'] is None:
        dump_str+=str(0)+'\n'
    else:
        dump_str+=str(atom_dict['box']['tilt'][1])+'\n'
    dump_str+=str(atom_dict['box']['bounds'][2][0])+' '+str(atom_dict['box']['bounds'][2][1])+' '
    if atom_dict['box']['tilt'] is None:
        dump_str+=str(0)+'\n'
    else:
        dump_str+=str(atom_dict['box']['tilt'][2])+'\n'
    dump_str+=atom_data_str
    for _, row in df.iterrows():
        output_str=''
        for key in keys:
            if ((key=='id') | (key=='type')):
                output_str+=str(int(row[key]))+' '
            else:
                output_str+=str(row[key])+' '
        output_str+='\n'
        dump_str+=output_str

    return dump_str


def construct_graph(df,max_bond_length=2.0):
    """
    Description:
        This function constructs a graph from a DataFrame containing atom data.
    Args:
        df (DataFrame): A DataFrame containing atom data.
        max_bond_length (float): The maximum bond length between atoms.
    Returns:    
        G (Graph): A graph object representing the system of atoms.
    """
    # Create an empty graph
    G = nx.Graph()
    # Add nodes to the graph
    for _, row in df.iterrows():
        G.add_node(row['id'], pos=(row['x'], row['y'], row['z']))
    # Calculate distances between all pairs of rows using vectorized operations
    node_coords = df[['x', 'y', 'z']].values
    distances = np.sqrt(np.sum((node_coords[:, np.newaxis] - node_coords) ** 2, axis=-1))
    # # Create a mask to exclude self-edges and edges that exceed the max_bond_length
    mask = (distances <= max_bond_length) & (np.arange(len(df))[:, np.newaxis] != np.arange(len(df)))
    # # Get the indices of nodes satisfying the conditions
    source_indices, target_indices = np.where(mask)
    # Create the graph and add edges
    for source_idx, target_idx in zip(source_indices, target_indices):
        G.add_edge(df.iloc[source_idx]['id'], df.iloc[target_idx]['id'])

    return G

def construct_node_specific_graph(df, node_id, region_radius, max_bond_length):
    """
    Description:
        This function constructs a graph from a DataFrame containing atom data, centered around a specified node.
    Args:
        df (DataFrame): A DataFrame containing atom data.
        node_id (int): The ID of the node around which to center the region.
        region_radius (float): The radius of the region around the specified node.
        max_bond_length (float): The maximum bond length between atoms.
    Returns:    
        G (Graph): A graph object representing the system of atoms in the region around the specified node.
    """
    # Get the coordinates of the specified node
    node_coords = df[df['id'] == node_id][['x', 'y', 'z']].values[0]
    # Filter the DataFrame to include only points within the region radius around the specified node
    df_subset = df[np.linalg.norm(df[['x', 'y', 'z']].values - node_coords, axis=1) <= region_radius]
    # Initialize an undirected graph
    G = nx.Graph()
    # Add nodes to the graph
    for _, row in df_subset.iterrows():
        G.add_node(row['id'], pos=(row['x'], row['y'], row['z']))
    # Compute pairwise distances
    distances = cdist(df_subset[['x', 'y', 'z']], df_subset[['x', 'y', 'z']])
    # Mask distances exceeding max_bond_length
    distances[distances > max_bond_length] = np.inf
    # Find indices of edges to add
    edge_indices = np.where(distances <= max_bond_length)
    # Add edges to the graph
    for i, j in zip(*edge_indices):
        if i < j:
            G.add_edge(df_subset.iloc[i]['id'], df_subset.iloc[j]['id'])

    return G

def plot_graph(G,fig_size=(22,8)):
    """
    Description:
        This function plots a graph object.
    Args:
        G (Graph): A graph object.
    """
    pos = {node: (coords[0], coords[1]) for node, coords in nx.get_node_attributes(G, 'pos').items()}
    plt.figure(figsize=fig_size)
    nx.draw(G, pos, with_labels=False, node_size=20, node_color='skyblue', font_size=8)
    plt.title('Top-Down View of Graph with Polygons Identified')
    plt.show()

def normalize_fig_dims(G,max_fig_dim):
    """
    Description:
        This function normalizes the dimensions of a figure to a specified maximum dimension.
    Args:
        G (Graph): A graph object.
        max_fig_dim (int): The maximum dimension of the figure.
    Returns:
        normalized_x (float): The normalized x dimension.
        normalized_y (float): The normalized y dimension.
    """
    pos = nx.get_node_attributes(G, 'pos')
    array_of_tuples = np.array(list(pos.values()), dtype=float)  # Convert coordinates to float if needed
    # Calculate the spans for both dimensions
    span_x = np.max(array_of_tuples[:, 0]) - np.min(array_of_tuples[:, 0])
    span_y = np.max(array_of_tuples[:, 1]) - np.min(array_of_tuples[:, 1])
    # Identify the larger span
    if span_x > span_y:
        # Normalize x dimension to 20 and adjust y dimension proportionally
        factor = 20 / span_x
        normalized_x = 20
        normalized_y = span_y * factor
    else:
        # Normalize y dimension to 20 and adjust x dimension proportionally
        factor = 20 / span_y
        normalized_y = 20
        normalized_x = span_x * factor

    return normalized_x, normalized_y

def identify_local_polygons(df, region_radius=2.0, max_bond_length=2.0, max_edges=12,plot=False,max_fig_dim=20):
    """
    Description:
        This function identifies local polygons around each atom in the system. It constructs
        a local graph around each atom within a specified region radius and identifies cycles
        (polygons) within that graph.`
    Args:
        df (DataFrame): A DataFrame containing the atom data.
        region_radius (float): The radius around each atom to consider for identifying polygons.
        max_bond_length (float): The maximum bond length between atoms.
        max_edges (int): The maximum number of edges in a polygon.
        plot (bool): A boolean indicating whether to plot the graph with polygons identified.
        max_fig_dim (int): The maximum dimension of the figure.
    Returns:
        df (DataFrame): The DataFrame with the local polygons identified.
        chain_counts (dict): A dictionary containing the counts of chain sizes.
        cycle_counts (dict): A dictionary containing the counts of cycle sizes.
    """
    chain_sizes = {} #will be for visualizing the chain sizes in OVITO
    chain_counts={} #will be for plotting the chain sizes
    cycle_sizes = {} #will be for visualizing the cycle sizes in OVITO
    cycle_counts = {} #will be for plotting the cycle sizes
    # Construct the graph for the entire system
    G = construct_graph(df,max_bond_length=max_bond_length)
    if plot == True:
        normalized_x, normalized_y = normalize_fig_dims(G,max_fig_dim)
        plot_graph(G,fig_size=(normalized_x, normalized_y))
    #Identify single atoms and chains of atoms
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(nx.cycle_basis(subgraph)) == 0:
            component_size = len(component)
            component=[int(comp) for comp in component]
            chain_counts[component_size] = chain_counts.get(component_size, 0) + 1
            for node in component:
                chain_sizes[node] = component_size

    unique_polygons = []

    for i,node_id in enumerate(df['id']):
        # Construct node-specific graph
        node_specific_graph = construct_node_specific_graph(df=df,node_id=node_id,region_radius=region_radius,max_bond_length=max_bond_length)
        cycles = list(nx.cycle_basis(node_specific_graph))
        for cycle in cycles:
            cycle = [int(node) for node in cycle]
            sorted_tuple = tuple(sorted(cycle))
            if sorted_tuple not in unique_polygons:
                unique_polygons.append(sorted_tuple)
                cycle_length = len(cycle)
                if cycle_length <= max_edges:
                    cycle_counts[cycle_length] = cycle_counts.get(cycle_length, 0) + 1 
                    for node in cycle:
                        if node not in cycle_sizes:
                            cycle_sizes[node] = [cycle_length]
                        else:
                            cycle_sizes[node].append(cycle_length)
    #go through cycle_sizes and change the list entries to integers, have this be the preference: 5, 7, min length of remaining list
    for key in cycle_sizes.keys():
        if 5 in cycle_sizes[key]:
            cycle_sizes[key]=5
        elif ((7 in cycle_sizes[key]) and (5 not in cycle_sizes[key])):
            cycle_sizes[key]=7
        else:
            cycle_sizes[key]=min(cycle_sizes[key])
    
    df['cycle_size'] = df['id'].map(cycle_sizes).fillna(0).astype(int)
    df['chain_size'] = df['id'].map(chain_sizes).fillna(0).astype(int)

    return df, chain_counts, cycle_counts, unique_polygons 


def assign_layers_local_xz_with_local_snap(df, x_bin_size=1.0, y_bin_size=1.0, eps=1.5,
                                     min_samples=2, z_outlier_thresh=5.0, snap_thresh=1.0):
    """
    Assigns layer IDs to carbon atoms locally in (x, y) bins based on (x,z) clustering,
    then refines assignments by snapping points to the nearest fitted layer line within the same bin.
    """

    df_out = df.copy()
    df_out['layer_id'] = 0

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    if x_bin_size <= 0 or y_bin_size <= 0:
        return df_out

    # 🔹 Handle degenerate range
    if np.isnan(x_min) or np.isnan(x_max) or x_max <= x_min:
        return df_out
    if np.isnan(y_min) or np.isnan(y_max) or y_max <= y_min:
        return df_out

    x_bins = np.arange(x_min, x_max + x_bin_size, x_bin_size)
    y_bins = np.arange(y_min, y_max + y_bin_size, y_bin_size)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask = (
                (df['x'] >= x_bins[i]) & (df['x'] < x_bins[i+1]) &
                (df['y'] >= y_bins[j]) & (df['y'] < y_bins[j+1])
            )
            subset = df[mask]

            if subset.empty:
                continue

            coords = subset[['x','z']].values
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = db.labels_

            unique_labels = [lbl for lbl in set(labels) if lbl != -1]
            if not unique_labels:
                continue

            # Assign layer IDs based on z-mean ordering
            cluster_means = {lbl: subset['z'].values[labels == lbl].mean() for lbl in unique_labels}
            sorted_clusters = sorted(cluster_means.items(), key=lambda x: -x[1])
            label_to_layer = {lbl: idx+1 for idx, (lbl, _) in enumerate(sorted_clusters)}

            for k, lbl in enumerate(labels):
                if lbl == -1:
                    min_cluster_z = min(cluster_means.values())
                    if subset.iloc[k]['z'] < min_cluster_z - z_outlier_thresh:
                        df_out.loc[subset.index[k], 'layer_id'] = 0
                    else:
                        closest_lbl = min(cluster_means, key=lambda l: abs(subset.iloc[k]['z']-cluster_means[l]))
                        df_out.loc[subset.index[k], 'layer_id'] = label_to_layer[closest_lbl]
                else:
                    df_out.loc[subset.index[k], 'layer_id'] = label_to_layer[lbl]

            # === SNAP TO LINES WITHIN THIS BIN ===
            bin_layers = df_out.loc[subset.index, 'layer_id'].unique()
            line_models = {}
            for layer in bin_layers:
                if layer == 0:
                    continue
                layer_points = df_out.loc[subset.index][df_out.loc[subset.index, 'layer_id'] == layer]
                if len(layer_points) < 2:
                    continue
                X = layer_points[['x']].values
                y = layer_points['z'].values
                model = LinearRegression().fit(X, y)
                line_models[layer] = model

            # Snap points in this bin
            for idx, row in df_out.loc[subset.index].iterrows():
                if row['layer_id'] == 0:
                    continue
                point_x = row['x']
                point_z = row['z']
                current_layer = row['layer_id']

                best_layer = current_layer
                best_dist = float('inf')

                for layer, model in line_models.items():
                    pred_z = model.predict([[point_x]])[0]
                    dist = abs(pred_z - point_z)
                    if dist < best_dist:
                        best_dist = dist
                        best_layer = layer

                if best_dist <= snap_thresh and best_layer != current_layer:
                    df_out.at[idx, 'layer_id'] = best_layer
            

    return df_out
      

def frame_generator(filename, step=1, frames_of_interest=None, label_layers = False, cycles_by_layer = False):
    logging.info(f"Starting analysis. Filtering for duplicate timesteps and sampling every {step}th frame.")
    seen_timesteps = set()

    for snapshot_index, dump in enumerate(parse_lammps_dumps(filename)):
        if frames_of_interest is not None and snapshot_index not in frames_of_interest:
            continue

        atom_dict = dump.as_dict()
        ts = atom_dict.get('timestep', -1)

        if ts in seen_timesteps:
            logging.debug(f"Skipping duplicate timestep: {ts}")
            continue
        seen_timesteps.add(ts)

        if snapshot_index % step == 0:
            yield (snapshot_index, dump, label_layers, cycles_by_layer)

    logging.info(f"Frame generator finished. Processed {len(seen_timesteps)} unique timesteps.")



def process_snapshot_parallel(args):
    """
    Function to process a single pre-loaded snapshot (frame).
    This function executes on the worker processes.
    """
    begin = time.time()
    snapshot_index, dump, label_layers, cycles_by_layer = args
    polygon_counts_per_layer = {} 
    # Extract atomic data
    atom_dict = dump.as_dict()
    modified_df = dump.data
    if len(modified_df)!=dump.natoms:
        modified_df = modified_df[:dump.natoms]

    modified_df = modified_df[(modified_df['type']==1) & (modified_df['c_PE']<-7.2)] #Modify this line as needed for filtering

    modified_df, chain_counts, cycle_counts, unique_polygons = identify_local_polygons(
        modified_df, region_radius=4.0, max_bond_length=2.0, max_edges=8, plot=False, max_fig_dim=20)
    
    if label_layers == True:
        low_cycles = modified_df[modified_df['cycle_size'] < 5].copy()
        low_cycles['layer_id'] = 0
        high_cycles = modified_df[modified_df['cycle_size'] >= 5]
        labeled_df = assign_layers_local_xz_with_local_snap(high_cycles, x_bin_size=4.0, y_bin_size=5.0, eps=1.5,
                                        min_samples=3, z_outlier_thresh=4.5, snap_thresh=1.0)
        modified_df = pd.concat([labeled_df, low_cycles], ignore_index=True)
    
        if cycles_by_layer == True:
            layer_map = labeled_df.set_index('id')['layer_id'].to_dict()
            for poly in unique_polygons:
                if not all(atom_id in layer_map for atom_id in poly):
                    continue
                layer_ids = [layer_map.get(atom_id, 0) for atom_id in poly]
                most_common_layer = max(set(layer_ids), key=layer_ids.count)
                poly_size = len(poly)
                polygon_counts_per_layer.setdefault(most_common_layer, {})
                polygon_counts_per_layer[most_common_layer].setdefault(poly_size, 0)
                polygon_counts_per_layer[most_common_layer][poly_size] += 1

    
    dump_str = snapshot_dump_str(modified_df, atom_dict)
    
    elapsed_time = time.time() - begin
    logging.info(f'Finished processing snapshot {snapshot_index} in {elapsed_time:.2f} seconds.')

    return snapshot_index, modified_df, dump_str, chain_counts, cycle_counts, polygon_counts_per_layer


if __name__ == '__main__':
    # --- Configuration ---
    lammps_dump_input = 'surface.script'  # Input LAMMPS dump file
    output_dump_file = 'identified_polygons.script'  # Output file for modified dumps
    output_cycle_data = 'identified_polygons_cycles.pkl'  # Output file for cycle data
    output_cycle_data_by_layer = 'identified_cycles_by_layer.pkl'  # Output file for cycle data by layer

    NUM_WORKERS = mp.cpu_count() 
    CHUNK_SIZE = 10  # Number of *filtered* frames to send to a worker at once
    FRAME_STEP = 1  # Only process every 1st frame
    frames_of_interest = None
    label_layers = False
    cycles_by_layer = False
    logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Create the filtered generator
    frame_iter = frame_generator(lammps_dump_input, step=FRAME_STEP, frames_of_interest=frames_of_interest, 
                                 label_layers=label_layers, cycles_by_layer=cycles_by_layer)

    # Use imap with the specified chunk size
    with mp.Pool(processes=NUM_WORKERS) as pool:
        logging.info(f"Starting pool with {NUM_WORKERS} workers and chunksize={CHUNK_SIZE}.")
        
        # pool.imap takes chunks from the generator and distributes them
        results_iterator = pool.imap(
            func=process_snapshot_parallel, 
            iterable=frame_iter, 
            chunksize=CHUNK_SIZE # This is the key setting for batching
        )
        
        # Collect results (blocking call)
        results = list(results_iterator)

        logging.info(f"Processing complete. Collected {len(results)} results. On to writing output.")

    # Unpacking results
    modded_dfs, dump_strs, snapshot_cycle_counts, snapshot_chain_counts, snapshot_polygon_counts = [], [], {}, {}, {}

    for snapshot_index, modified_df, dump_str, chain_counts, cycle_counts, polygon_counts_per_layer in results:
        modded_dfs.append(modified_df)
        dump_strs.append(dump_str)
        snapshot_cycle_counts[snapshot_index] = cycle_counts
        snapshot_chain_counts[snapshot_index] = chain_counts
        snapshot_polygon_counts[snapshot_index] = polygon_counts_per_layer

    with open(output_dump_file, 'w') as f:
        for dump_str in dump_strs:
            f.write(dump_str)

    cycle_data = pd.DataFrame.from_dict(snapshot_cycle_counts, orient='index')
    chain_data = pd.DataFrame.from_dict(snapshot_chain_counts, orient='index')
    polygon_data = pd.DataFrame.from_dict(snapshot_polygon_counts, orient='index')
    # Fill missing values with zeros
    cycle_data.fillna(0, inplace=True)
    chain_data.fillna(0, inplace=True)
    polygon_data.fillna(0, inplace=True)
    cycle_data.to_pickle(output_cycle_data)
    if label_layers== True or cycles_by_layer==True:
        polygon_data.to_pickle(output_cycle_data_by_layer)