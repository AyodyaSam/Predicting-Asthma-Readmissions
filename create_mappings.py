import pickle
import pandas as pd
import numpy as np

def create_global_mapping(node_mappings):
    """Create global node mapping"""
    global_node_map = {}
    reverse_mappings = {}
    current_offset = 0

    for node_type, local_mapping in node_mappings.items():
        type_mapping = {}

        for og_identity, local_id in local_mapping.items():
            global_id = current_offset + local_id
            type_mapping[og_identity] = global_id
            reverse_mappings[global_id] = (node_type, og_identity)

        global_node_map[node_type] = type_mapping
        current_offset += len(local_mapping)
        
        print(node_type)
        print(len(local_mapping))

    return global_node_map, reverse_mappings

# need a global edge list
def create_global_edge_mapping(edge_lists, global_node_map, node_mappings):
    """Convert edges from list of original ids to global ids

    Args:
        edge_lists (Dict): dict with edge type -> list of tuples
        global_node_map (Dict): global node mappings
        node_mappings (Dict): original node mappings

    Returns:
        Dict: Dict with edge type -> list of tuples using global IDs
    """

    global_edge_lists = {}

    # edge_type_map = {
    #     'patient_medication': ('patients','medications'),
    #     'patient_comorbidity': ('patients','comorbidities'),
    #     'patient_age': ('patients','age_groups'),
    #     'patient_gender': ('patients','genders')
    # }

    edge_type_map = {
        'patient_medication': ('patient','medication'),
        'patient_comorbidity': ('patient','comorbidity'),
        'patient_similarity': ('patient','patient'),
        'medication_cooccurrence': ('medication','medication'),
        'comorbidity_cooccurrence':('comorbidity','comorbidity')
    }

    for edge_type, edge_list in edge_lists.items():
        print(f"Processing {edge_type}: {len(edge_list)} edges")

        if edge_type not in edge_type_map:
            print(f"Warning: Unknown edge type {edge_type}")
            continue


        source_node_type, target_node_type = edge_type_map[edge_type]

        global_edges = []

        for edge in edge_list:
            try:
                source_local_id, target_local_id = edge

                source_entities = list(node_mappings[source_node_type].keys())
                target_entities = list(node_mappings[target_node_type].keys())

                source_entity = source_entities[source_local_id]
                target_entity = target_entities[target_local_id]

                source_glob_id = global_node_map[source_node_type][source_entity]
                target_glob_id = global_node_map[target_node_type][target_entity]


                global_edges.append((source_glob_id,target_glob_id))
                
            except (IndexError, KeyError) as e:
                print(f"Error in processing edge {edge} in {edge_type}: {e}")
                continue

        global_edge_lists[edge_type] = global_edges

    return global_edge_lists

if __name__=="__main__":
    with open('datasets/mimic_asthma_graph_data_icd_full_v5.pkl', 'rb') as f:
        data = pickle.load(f)

    # create the global mappings
    global_node_map, reverse_map = create_global_mapping(data['node_mappings'])
    global_edge_lists = create_global_edge_mapping(data['edge_lists'], global_node_map, data['node_mappings'])

    # add them data structure
    data['global_node_map'] = global_node_map
    data['reverse_map'] = reverse_map
    data['global_edge_lists'] = global_edge_lists

    # resave the updated pickle
    with open('datasets/mimic_asthma_graph_data_v5.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("Updated pickle file with global mappings!")    