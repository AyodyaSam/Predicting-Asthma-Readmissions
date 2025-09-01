import torch
import torch_geometric
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd

class GraphBuilder:
    """
    Build HeteroData object for asthma patient prediction using Pytorch Geometric
    Using Decagon structure as a reference (https://github.com/mims-harvard/decagon)
    """

    def __init__(self, graph_data):
        self.patient_data = graph_data['patient_data']
        self.node_mappings = graph_data['node_mappings']
        self.global_edge_lists = graph_data['global_edge_lists']
        self.global_node_map = graph_data['global_node_map']

        # initialise HeteroData object
        self.graph = HeteroData()

    def create_node_features(self):
        """Create feature matrices for each node type
        """

        print("Creating node features...")

        patient_features = self._create_patient_features()
        self.graph['patient'].x = patient_features

        medication_features = self._create_medication_features()
        self.graph['medication'].x = medication_features

        comorbidity_features = self._create_comorbidity_features()
        self.graph['comorbidity'].x = comorbidity_features

        # age_group_features = self._create_age_group_features()
        # self.graph['age_group'].x = age_group_features

        # gender_features = self._create_gender_features()
        # self.graph['gender'].x = gender_features

        print("Node features created!")

        return self.graph

    def _create_patient_features(self):
        """Create features for patient nodes
        """

        # num_patients = len(self.node_mappings['patients'])

        features = []

        # patients_sorted = sorted(self.node_mappings['patients'].items(),
        #                          key=lambda x:self.global_node_map['patients'][x[0]])
        patients_sorted = sorted(self.node_mappings['patient'].items(),
                                 key=lambda x:self.global_node_map['patient'][x[0]])
        # patients_sorted = sorted(self.node_mappings['patient'].items(),
        #                 key=lambda x: x[1]) 
        self.patient_data = self.patient_data.fillna(0)
        for patient_id, _ in patients_sorted:

            pat_row = self.patient_data[
                self.patient_data['SUBJECT_ID'] == patient_id
            ]

            if len(pat_row) == 0:
                features.append([0.0,0.0,0.0,0.0])
                continue

            pat_row = pat_row.iloc[0]

            feature_vec = [
                float(pat_row.get('AGE',0.0)),
                1.0 if pat_row.get('GENDER') == 'M' else 0.0,
                float(pat_row.get('HOSPITAL_EXPIRE_FLAG',0.0)),
                float(len(pat_row.get('MEDICATION_CATEGORY',[]))),
                float(pat_row.get('glucose_mean', 0.0)),
                float(pat_row.get('wbc_count', 0.0)), 
                float(pat_row.get('eosinophils_pct_high', 0.0)), # high beos 
                float(pat_row.get('hemoglobin_a1c_mean', 0.0)),
                float(pat_row.get('triglycerides_mean', 0.0)),
                float(pat_row.get('ige_mean', 0.0)),
                float(pat_row.get('creatinine_mean', 0.0)),
                float(pat_row.get('sodium_mean', 0.0)),
                float(pat_row.get('hemoglobin_mean', 0.0)),
                float(pat_row.get('potassium_mean', 0.0))
            ]

            features.append(feature_vec)

        return torch.FloatTensor(features)

    def _create_medication_features(self):
        """Create features for medication nodes
        """

        # get medications
        # medications = list(self.node_mappings['medications'].keys())
        medications = list(self.node_mappings['medication'].keys())

        features = [] # begin with simple categorical features

        for med in medications:
            # to consider for later
            # - drug class encoding
            # - route of administration
            # - frequency prescribed

            feature_vec = [
                1.0,
                len(med),
                1.0 if 'steroid' in med.lower() else 0.0,# drug class indicator
                1.0 if 'inhaler' in med.lower() else 0.0, # route indicator
                1.0 if 'bronchodilators' in med.lower() else 0.0, # add bronchodilator
                1.0 if 'leukotriene_modifiers' in med.lower() else 0.0 # add ltra modifier 
            ]
            print(med)
            features.append(feature_vec)

        return torch.FloatTensor(features)

    def _create_comorbidity_features(self):
        """Create features for comorbidity nodes"""

        # comorbidities = list(self.node_mappings['comorbidities'].keys())
        comorbidities = list(self.node_mappings['comorbidity'].keys())

        features = []

        for comorb in comorbidities:

            feature_vec = [
                1.0,
                1.0 if comorb == "hypertension" else 0.0,
                1.0 if comorb == "anxiety" else 0.0,
                1.0 if comorb == "diabetes" else 0.0,
                1.0 if comorb == "allergies" else 0.0,
            ]

            features.append(feature_vec)

        return torch.FloatTensor(features)

    def _create_age_group_features(self):
        """Create features for age group nodes"""

        # age_groups = list(self.node_mappings['age_groups'].keys())
        age_groups = list(self.node_mappings['age_group'].keys())

        age_order = {'pediatric':0, 'adult':1, 'elderly':2}

        features = []

        for age_group in age_groups:

            feature_vec = [
                float(age_order.get(age_group,1)),
                1.0
            ]

            features.append(feature_vec)

        return torch.FloatTensor(features)

    def _create_gender_features(self):
        """Create features for gender features"""

        genders = list(self.node_mappings['gender'].keys())

        features = []

        for gen in genders:

            feature_vec = [
                1.0,
                1.0 if gen == 'M' else 0.0,
                1.0 if gen == 'F' else 0.0
            ]

            features.append(feature_vec)

        return torch.FloatTensor(features)


    def create_edge_indices(self):
        """Convert edge lists to PyG endge_index format"""

        print("Creating edge indices...")

        edge_type_configs = {
            'patient_medication': ('patient', 'takes', 'medication'),
            'patient_comorbidity': ('patient', 'has', 'comorbidity'),
            'medication_cooccurrence': ('medication', 'coprescribed', 'medication'),
            'comorbidity_cooccurrence': ('comorbidity', 'cooccurs', 'comorbidity'),
            # 'patient_similarity': ('patient','similar','patient')
        }

        # edge_type_configs = {
        #     'patient_medication': ('patients', 'takes', 'medications'),
        #     'patient_comorbidity': ('patients', 'has', 'comorbidities'),
        #     'patient_age': ('patients', 'is_age', 'age_groups'),
        #     'patient_gender': ('patients', 'is_gender', 'genders')
        # }

        for edge_list_name, (source_type, relation, target_type) in edge_type_configs.items():
            if edge_list_name not in self.global_edge_lists:
                print(f"Warning {edge_list_name} not found in edge lists")
                continue

            edges = self.global_edge_lists[edge_list_name]

            if not edges:
                print(f"Warning: {edge_list_name} is empty")
                continue

            # convert to pyg format -[2, num_edges]
            edge_index = self._convert_to_edge_index(edges, source_type, target_type)

            self.graph[source_type, relation, target_type].edge_index = edge_index

            print(f"Added {edge_index.shape[1]} edges for {source_type}-{relation}-{target_type}")

        self._add_reverse_edges()

        print("Edge indes created!")

        return self.graph

    def _convert_to_edge_index(self, edges, source_type, target_type):
        """Convert edge list to PyG edge index tensor"""

        source_edges = []
        target_edges = []

        source_offset = self._get_node_type_offset(source_type)
        target_offset = self._get_node_type_offset(target_type)

        for source_glob, target_glob in edges:
            source_local = source_glob - source_offset
            target_local = target_glob - target_offset

            source_edges.append(source_local)
            target_edges.append(target_local)

            if source_local < 0 or target_local < 0:
                print(f"Negative local ID: {source_local}, {target_local}")
                continue

        edge_index = torch.tensor([source_edges,target_edges],dtype=torch.long)

        return edge_index

    def _get_node_type_offset(self, node_type):
        """Get starting global ID for a node type"""

        if not self.global_node_map[node_type]:
            return 0

        return min(self.global_node_map[node_type].values())


    def _add_reverse_edges(self):
        """Add reverse edges for dymmetric relationships"""

        relations_to_reverse = [
            ('patient', 'takes', 'medication', 'prescribed_to'),
            ('patient', 'has', 'comorbidity', 'affects'),
            # ('patient','similar','patient','similar')
        ]

        # relations_to_reverse = [
        #     ('patients', 'takes', 'medications', 'prescribed_to'),
        #     ('patients', 'has', 'comorbidities', 'affects'),
        # ]

        for source_type, relation, target_type, reverse_relation in relations_to_reverse:
            if (source_type, relation, target_type) in self.graph.edge_types:
                forward_edges = self.graph[source_type, relation, target_type].edge_index

                reverse_edges = torch.stack([forward_edges[1],forward_edges[0]])

                self.graph[target_type, relation, source_type].edge_index = reverse_edges

                print(f"Added reverse edges {target_type}-{reverse_relation}-{source_type}")

    def build_graph(self):
        """Build the complete hetero graph"""

        print("Build heterogenous graph")

        # node features
        self.create_node_features()

        # edge features
        self.create_edge_indices()

        # print summary
        self.print_graph_summary()

        return self.graph

    def print_graph_summary(self):
        """Print summary of constructed graph"""

        print("\n" + "="*50)
        print("GRAPH SUMMARY")
        print("\n" + "="*50)

        print("\nNode types:")
        for node_type in self.graph.node_types:
            num_nodes = self.graph[node_type].x.shape[0]
            num_features = self.graph[node_type].x.shape[1]
            print(f"  {node_type}: {num_nodes} nodes, {num_features} features")

        print("\nEdge types:")
        for edge_type in self.graph.edge_types:
            num_edges = self.graph[edge_type].edge_index.shape[1]
            print(f"  {edge_type}: {num_edges} edges")

        print(f"Total nodes: {sum(self.graph[nt].x.shape[0] for nt in self.graph.node_types)}")
        print(f"Total edges: {sum(self.graph[et].edge_index.shape[1] for et in self.graph.edge_types)}")


if __name__=="__main__":
    import pickle

    with open('datasets/mimic_asthma_graph_data_v5.pkl', 'rb') as f:
        data = pickle.load(f)

    graph_builder = GraphBuilder(data)

    hetero_graph = graph_builder.build_graph()

    reproducible_data = {
        'graph_object': hetero_graph,
        'node_mappings': data['node_mappings'], 
        'global_node_map': data['global_node_map'],
        'global_edge_lists': data['global_edge_lists'],
        'patient_data': data['patient_data'],
        'readmission_labels': data['readmission_labels'],
        'build_config': {
            'pytorch_geometric_version': torch_geometric.__version__,
            'torch_version': torch.__version__
        }
    }
    # save the graph locally
    # torch.save(hetero_graph,'datasets/asthma_hetero_graph_v6.pt')
    torch.save(reproducible_data,'datasets/asthma_hetero_graph_complete.pt')

    # print("Graph saved locally!")
