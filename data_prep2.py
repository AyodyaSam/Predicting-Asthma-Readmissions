import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class MIMICDataPreparator:
    """
    Prepare MIMIC-III data for heterogeneous graph construction
    Focus on asthma patients and related comorbidities
    """
    
    def __init__(self, mimic_path):
        """
        Initialise with path to MIMIC-III CSV files
        
        Args:
            mimic_path: Path to directory containing MIMIC-III CSV files
        """
        self.mimic_path = mimic_path
        self.asthma_patients = set()
        
        # Define asthma-related medications (generic names)
        self.asthma_medications = {
            'bronchodilators': ['albuterol', 'levalbuterol', 'ipratropium', 'tiotropium'],
            'corticosteroids': ['prednisone', 'prednisolone', 'methylprednisolone', 
                              'fluticasone', 'budesonide', 'beclomethasone'],
            'leukotriene_modifiers': ['montelukast', 'zafirlukast'],
            'combination_inhalers': ['fluticasone/salmeterol', 'budesonide/formoterol'],
            'rescue_inhalers': ['albuterol', 'levalbuterol']
        }
        
        # Comorbidity keywords for text mining (avoiding ICD codes)
        self.comorbidity_keywords = {
            'allergies': ['allergy', 'allergic', 'rhinitis', 'eczema', 'atopic'],
            'obesity': ['obesity', 'obese', 'bmi', 'overweight'],
            'gerd': ['gerd', 'reflux', 'heartburn', 'gastroesophageal'],
            'anxiety': ['anxiety', 'anxious', 'panic', 'depression', 'depressed']
        }
        
    def load_core_tables(self):
        """Load essential MIMIC-III tables"""
        print("Loading MIMIC-III tables...")
        
        # Core patient information
        self.patients = pd.read_csv(f"{self.mimic_path}/PATIENTS.csv").rename(columns=str.upper)
        self.admissions = pd.read_csv(f"{self.mimic_path}/ADMISSIONS.csv").rename(columns=str.upper)
        
        # Medication data
        self.prescriptions = pd.read_csv(f"{self.mimic_path}/PRESCRIPTIONS.csv").rename(columns=str.upper)
        
        # Clinical notes (for comorbidity detection)
        self.noteevents = pd.read_csv(f"{self.mimic_path}/NOTEEVENTS.csv").rename(columns=str.upper)
        
        # Lab events (for clinical indicators)
        self.labevents = pd.read_csv(f"{self.mimic_path}/LABEVENTS.csv").rename(columns=str.upper)
        
        # Lab items dictionary for lab names
        try:
            self.d_labitems = pd.read_csv(f"{self.mimic_path}/D_LABITEMS.csv").rename(columns=str.upper)
            print("Loaded lab items dictionary")
        except FileNotFoundError:
            print("D_LABITEMS.csv.gz not found - will use item IDs directly")
            self.d_labitems = None
        
        # Diagnoses to cross-reference 
        # https://en.wikipedia.org/wiki/List_of_ICD-9_codes_460%E2%80%93519:_diseases_of_the_respiratory_system
        self.diagnoses = pd.read_csv(f"{self.mimic_path}/DIAGNOSES_ICD.csv").rename(columns=str.upper)
        
        print(f"Loaded {len(self.patients)} patients, {len(self.admissions)} admissions")
        print(f"Loaded {len(self.prescriptions)} prescriptions, {len(self.noteevents)} notes")
        print(f"Loaded {len(self.labevents)} lab events")

    def identify_asthma_patients(self):
        """Identify asthma patients using ICD-9 codes"""
        # Asthma ICD-9 codes: 493.xx series
        asthma_codes = self.diagnoses[
            self.diagnoses['ICD9_CODE'].str.startswith('493', na=False)
        ]
        self.asthma_patients = set(asthma_codes['SUBJECT_ID'].unique())
        print(f"Identified {len(self.asthma_patients)} asthma patients")
        return self.asthma_patients

    def extract_patient_features(self):
        """Extract patient demographic and clinical features"""
        print("Extracting patient features...")
        
        # Filter to asthma patients
        asthma_patient_data = self.patients[
            self.patients['SUBJECT_ID'].isin(self.asthma_patients)
        ].copy()
        
        # Calculate age at admission
        admissions_with_age = self.admissions.merge(
            asthma_patient_data[['SUBJECT_ID', 'DOB','GENDER']], 
            on='SUBJECT_ID'
        )
        
        def safe_age_calculation(row):
            try:
                dob = pd.to_datetime(row['DOB'])
                admittime = pd.to_datetime(row['ADMITTIME'])
                age = (admittime - dob).days / 365.25
                return min(age, 90)  # Cap at 90 years
            except (OverflowError, ValueError):
                return 90  # Return 90 for problematic dates

        admissions_with_age['DOB'] = pd.to_datetime(admissions_with_age['DOB'])
        admissions_with_age['ADMITTIME'] = pd.to_datetime(admissions_with_age['ADMITTIME'])
        admissions_with_age['AGE'] = admissions_with_age.apply(safe_age_calculation, axis=1)
        
        # age groups
        def categorise_age(age):
            if age < 18:
                return 'pediatric'
            elif age < 65:
                return 'adult'
            else:
                return 'elderly'
        
        admissions_with_age['AGE_GROUP'] = admissions_with_age['AGE'].apply(categorise_age)
        
        # Aggregate patient features
        patient_features = admissions_with_age.groupby('SUBJECT_ID').agg({
            'AGE': 'mean',
            'AGE_GROUP': 'first',  # first admission age group
            'GENDER': 'first',
            'HOSPITAL_EXPIRE_FLAG': 'max'  # Any hospital death
        }).reset_index()

        patient_features['GENDER_M'] = (patient_features['GENDER'] == 'M').astype(int)
        patient_features['GENDER_F'] = (patient_features['GENDER'] == 'F').astype(int)
        
        return patient_features
    
    def extract_medications(self):
        """Extract and categorise medications for asthma patients"""
        print("Extracting medications...")
        
        # Filter prescriptions for asthma patients
        asthma_prescriptions = self.prescriptions[
            self.prescriptions['SUBJECT_ID'].isin(self.asthma_patients)
        ].copy()
        
        # Clean and standardise drug names
        asthma_prescriptions['DRUG_CLEAN'] = (
            asthma_prescriptions['DRUG']
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        
        # Categorise medications
        def categorise_medication(drug_name):
            drug_lower = drug_name.lower()
            
            for category, meds in self.asthma_medications.items():
                if any(med in drug_lower for med in meds):
                    return category
            
            # Check for other relevant medication classes
            if any(word in drug_lower for word in ['omeprazole', 'lansoprazole', 'pantoprazole']):
                return 'ppi'
            elif any(word in drug_lower for word in ['loratadine', 'cetirizine', 'diphenhydramine']):
                return 'antihistamine'
            elif any(word in drug_lower for word in ['sertraline', 'fluoxetine', 'citalopram']):
                return 'antidepressant'
            else:
                return 'other'
        
        asthma_prescriptions['MEDICATION_CATEGORY'] = asthma_prescriptions['DRUG_CLEAN'].apply(
            categorise_medication
        )
        
        # Create patient-medication relationships
        patient_medications = asthma_prescriptions.groupby('SUBJECT_ID').agg({
            'MEDICATION_CATEGORY': lambda x: list(x.unique()),
            'DRUG_CLEAN': lambda x: list(x.unique())
        }).reset_index()
        
        return patient_medications, asthma_prescriptions
    
    def extract_comorbidities(self):
        """Extract comorbidities using ICD-9 codes"""
        # Define ICD-9 code mappings
        icd9_comorbidities = {
            'allergies': ['V15.0', 'V15.1', 'V15.2', '477', '691'],  # Allergy history, rhinitis, eczema
            'obesity': ['278.0', '278.00', '278.01'],  # Obesity
            'gerd': ['530.81', '530.11'],  # GERD, reflux esophagitis
            'anxiety': ['300.0', '300.00', '300.01', '311'],  # Anxiety, depression
            'diabetes': ['250'],  # Diabetes codes start with 250
            'hypertension': ['401', '402', '403', '404', '405']  # Hypertension codes
        }
        
        # Filter diagnoses for asthma patients
        asthma_diagnoses = self.diagnoses[
            self.diagnoses['SUBJECT_ID'].isin(self.asthma_patients)
        ]
        
        # Detect comorbidities
        patient_comorbidities = []
        for patient_id in self.asthma_patients:
            patient_codes = asthma_diagnoses[
                asthma_diagnoses['SUBJECT_ID'] == patient_id
            ]['ICD9_CODE'].tolist()
            
            detected_conditions = []
            for condition, codes in icd9_comorbidities.items():
                if any(any(patient_code.startswith(code) for code in codes) 
                    for patient_code in patient_codes):
                    detected_conditions.append(condition)
            
            patient_comorbidities.append({
                'SUBJECT_ID': patient_id,
                'DETECTED_COMORBIDITIES': detected_conditions
            })
        
        return pd.DataFrame(patient_comorbidities)

    def extract_lab_indicators(self):
        """Extract relevant lab values that might indicate comorbidities"""
        print("Extracting lab indicators...")
        
        # Filter lab events for asthma patients
        asthma_labs = self.labevents[
            self.labevents['SUBJECT_ID'].isin(self.asthma_patients)
        ].copy()
        
        if len(asthma_labs) == 0:
            print("No lab data found for asthma patients")
            return pd.DataFrame()
        
        # Add lab names if dictionary available
        if self.d_labitems is not None:
            asthma_labs_with_names = asthma_labs.merge(
                self.d_labitems[['ITEMID', 'LABEL']], 
                on='ITEMID', 
                how='left'
            )
        else:
            asthma_labs_with_names = asthma_labs.copy()
            asthma_labs_with_names['LABEL'] = asthma_labs_with_names['ITEMID'].astype(str)
        
        # Define relevant lab patterns (flexible matching)
        relevant_lab_patterns = {
            'glucose': ['glucose', 'gluc'],
            'hemoglobin_a1c': ['hba1c', 'hemoglobin a1c', 'glycosylated'],
            'cholesterol': ['cholesterol', 'chol'],
            'triglycerides': ['triglyceride', 'trig'],
            'eosinophils': ['eosinophil', 'eos'],
            'ige': ['ige', 'immunoglobulin e'],
            'wbc': ['white blood cell', 'wbc', 'leukocyte'],
            'creatinine': ['creatinine', 'creat'],
            'bun': ['bun', 'urea nitrogen'],
            'sodium': ['sodium', 'na'],
            'potassium': ['potassium', 'k+'],
            'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
            'hematocrit': ['hematocrit', 'hct'],
            'feno': ['feno','Fractional Exhaled Nitric Oxide','FeNO'],
        }
        
        # Find matching lab items for each category
        lab_item_mapping = {}
        
        for lab_category, patterns in relevant_lab_patterns.items():
            matching_items = []
            
            for pattern in patterns:
                matches = asthma_labs_with_names[
                    asthma_labs_with_names['LABEL'].str.contains(
                        pattern, case=False, na=False
                    )
                ]['ITEMID'].unique()
                matching_items.extend(matches)
            
            if matching_items:
                lab_item_mapping[lab_category] = list(set(matching_items))
                print(f"Found {len(lab_item_mapping[lab_category])} items for {lab_category}")
        
        if not lab_item_mapping:
            print("No relevant lab items found, returning basic aggregation") # need to comment out debug code
            return self._basic_lab_aggregation(asthma_labs)
        
        # Extract features for each relevant lab category
        patient_lab_features = []
        
        for patient_id in self.asthma_patients:
            patient_labs = asthma_labs_with_names[
                asthma_labs_with_names['SUBJECT_ID'] == patient_id
            ]
            
            patient_features = {'SUBJECT_ID': patient_id}
            
            for lab_category, item_ids in lab_item_mapping.items():
                category_labs = patient_labs[
                    patient_labs['ITEMID'].isin(item_ids)
                ]['VALUENUM'].dropna()
                
                if len(category_labs) > 0:
                    # summary statistics
                    patient_features[f'{lab_category}_mean'] = category_labs.mean()
                    patient_features[f'{lab_category}_median'] = category_labs.median()
                    patient_features[f'{lab_category}_std'] = category_labs.std()
                    patient_features[f'{lab_category}_count'] = len(category_labs)
                    patient_features[f'{lab_category}_min'] = category_labs.min()
                    patient_features[f'{lab_category}_max'] = category_labs.max()
                    
                    # Clinical indicators (abnormal ranges)
                    patient_features.update(
                        self._get_clinical_indicators(lab_category, category_labs)
                    )
                else:
                    # if no lab values for this category
                    patient_features[f'{lab_category}_mean'] = np.nan
                    patient_features[f'{lab_category}_median'] = np.nan
                    patient_features[f'{lab_category}_std'] = np.nan
                    patient_features[f'{lab_category}_count'] = 0
                    patient_features[f'{lab_category}_min'] = np.nan
                    patient_features[f'{lab_category}_max'] = np.nan
            
            patient_lab_features.append(patient_features)
        
        lab_features_df = pd.DataFrame(patient_lab_features)
        
        print(f"Extracted lab features for {len(lab_features_df)} patients")
        lab_cols = [col for col in lab_features_df.columns if col != 'SUBJECT_ID']
        print(f"Lab feature columns: {lab_cols[:10]}...")  # show top 10
        
        return lab_features_df

    def _get_clinical_indicators(self, lab_category, lab_values):
        """Get clinical indicators based on normal ranges"""
        indicators = {}
        
        # Define normal ranges (approximate)
        normal_ranges = {
            'glucose': (70, 100),  # mg/dL fasting
            'hemoglobin_a1c': (4.0, 5.6),  # %
            'cholesterol': (0, 200),  # mg/dL total
            'triglycerides': (0, 150),  # mg/dL
            'eosinophils': (1, 4),  # % of WBC
            'wbc': (4.0, 11.0),  # K/uL
            'creatinine': (0.6, 1.2),  # mg/dL
            'bun': (7, 20),  # mg/dL
            'sodium': (136, 145),  # mEq/L
            'potassium': (3.5, 5.0),  # mEq/L
            'hemoglobin': (12.0, 16.0),  # g/dL (approximate for both genders)
            'hematocrit': (36, 46)  # % (approximate for both genders)
        }
        
        if lab_category in normal_ranges:
            low_threshold, high_threshold = normal_ranges[lab_category]
            
            # Calculate percentage of abnormal values
            low_values = (lab_values < low_threshold).sum()
            high_values = (lab_values > high_threshold).sum()
            total_values = len(lab_values)
            
            indicators[f'{lab_category}_pct_low'] = low_values / total_values if total_values > 0 else 0
            indicators[f'{lab_category}_pct_high'] = high_values / total_values if total_values > 0 else 0
            indicators[f'{lab_category}_ever_abnormal'] = 1 if (low_values + high_values) > 0 else 0
            
            # Most recent value (if we have multiple values)
            if total_values > 0:
                most_recent = lab_values.iloc[-1]  # Assumes sorted by time
                indicators[f'{lab_category}_recent_low'] = 1 if most_recent < low_threshold else 0
                indicators[f'{lab_category}_recent_high'] = 1 if most_recent > high_threshold else 0
        
        return indicators

    def _basic_lab_aggregation(self, asthma_labs):
        """Fallback basic lab aggregation if no specific labs found"""
        lab_features = asthma_labs.groupby('SUBJECT_ID').agg({
            'VALUENUM': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        lab_features.columns = ['SUBJECT_ID', 'lab_mean', 'lab_std', 'lab_count', 'lab_min', 'lab_max']
        
        return lab_features
    
    def create_graph_data_structure(self):
        """Combine all extracted data into graph-ready format"""
        print("Creating graph data structure...")
        
        # Get all components
        patient_features = self.extract_patient_features()
        patient_medications, prescription_details = self.extract_medications()
        patient_comorbidities = self.extract_comorbidities()
        
        # NEW: Extract lab features
        lab_features = self.extract_lab_indicators()
        
        # Merge all patient data
        graph_data = patient_features.merge(
            patient_medications, on='SUBJECT_ID', how='left'
        ).merge(
            patient_comorbidities, on='SUBJECT_ID', how='left'
        )
        
        # Merge lab features if available
        if not lab_features.empty:
            graph_data = graph_data.merge(lab_features, on='SUBJECT_ID', how='left')
            lab_cols = [col for col in lab_features.columns if col != 'SUBJECT_ID']
            print(f"Added {len(lab_cols)} lab features")
        
        # Fill missing values
        graph_data['MEDICATION_CATEGORY'] = graph_data['MEDICATION_CATEGORY'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        graph_data['DETECTED_COMORBIDITIES'] = graph_data['DETECTED_COMORBIDITIES'].apply(
            lambda x: x if isinstance(x, list) else []
        )

        # Create node mappings
        node_mappings = self.create_node_mappings(graph_data)

        # Create edge lists
        edge_lists = self.create_edge_lists(graph_data, node_mappings)

        # Create readmission labels
        readmission_labels = self.create_readmission_labels(days_threshold=90)

        return {
            'patient_data': graph_data,
            'node_mappings': node_mappings,
            'edge_lists': edge_lists,
            'prescription_details': prescription_details,
            'readmission_labels': readmission_labels,
            'lab_features': lab_features  # Include separately for analysis
        }
    
    def create_node_mappings(self, graph_data):
        """Create mappings from entities to node IDs"""
        node_mappings = {}
        
        # Patient nodes
        patients = graph_data['SUBJECT_ID'].unique()
        node_mappings['patient'] = {pid: i for i, pid in enumerate(patients)}
        
        # Medication category nodes
        all_meds = set()
        for med_list in graph_data['MEDICATION_CATEGORY'].dropna():
            all_meds.update(med_list)
        node_mappings['medication'] = {med: i for i, med in enumerate(all_meds)}
        
        # Comorbidity nodes
        all_comorbidities = set()
        for comorbidity_list in graph_data['DETECTED_COMORBIDITIES'].dropna():
            all_comorbidities.update(comorbidity_list)
        node_mappings['comorbidity'] = {cond: i for i, cond in enumerate(all_comorbidities)}
        
        # Feature nodes (age groups, gender)
        # age_groups = graph_data['AGE_GROUP'].unique()
        # genders = graph_data['GENDER'].unique()
        
        # node_mappings['age_group'] = {age: i for i, age in enumerate(age_groups)}
        # node_mappings['gender'] = {gender: i for i, gender in enumerate(genders)}
        
        return node_mappings
    
    def create_edge_lists(self, graph_data, node_mappings):
        """Create edge lists for different relationship types"""
        edge_lists = {}
        
        # Patient-medication edges
        patient_med_edges = []
        for _, row in graph_data.iterrows():
            patient_id = node_mappings['patient'][row['SUBJECT_ID']]
            for med in row['MEDICATION_CATEGORY']:
                if med in node_mappings['medication']:
                    med_id = node_mappings['medication'][med]
                    patient_med_edges.append((patient_id, med_id))
        
        edge_lists['patient_medication'] = patient_med_edges
        
        # Patient-comorbidity edges
        patient_comorbidity_edges = []
        for _, row in graph_data.iterrows():
            patient_id = node_mappings['patient'][row['SUBJECT_ID']]
            for condition in row['DETECTED_COMORBIDITIES']:
                if condition in node_mappings['comorbidity']:
                    condition_id = node_mappings['comorbidity'][condition]
                    patient_comorbidity_edges.append((patient_id, condition_id))
        
        edge_lists['patient_comorbidity'] = patient_comorbidity_edges
        
        # Patient-feature edges
        # patient_age_edges = []
        # patient_gender_edges = []
        
        # for _, row in graph_data.iterrows():
        #     patient_id = node_mappings['patient'][row['SUBJECT_ID']]
            
        #     # Age group edge
        #     if pd.notna(row['AGE_GROUP']):
        #         age_id = node_mappings['age_group'][row['AGE_GROUP']]
        #         patient_age_edges.append((patient_id, age_id))
            
        #     # Gender edge
        #     if pd.notna(row['GENDER']):
        #         gender_id = node_mappings['gender'][row['GENDER']]
        #         patient_gender_edges.append((patient_id, gender_id))
        
        # edge_lists['patient_age'] = patient_age_edges
        # edge_lists['patient_gender'] = patient_gender_edges
        
        # # NEW: Co-occurrence edges
        med_cooccurrence_edges = self.create_medication_cooccurrence_edges(graph_data, node_mappings)
        edge_lists['medication_cooccurrence'] = med_cooccurrence_edges
        
        comorbidity_cooccurrence_edges = self.create_comorbidity_cooccurrence_edges(graph_data, node_mappings)
        edge_lists['comorbidity_cooccurrence'] = comorbidity_cooccurrence_edges
        
        # NEW: Patient similarity edges (based on shared medications/comorbidities)
        patient_similarity_edges = self.create_patient_similarity_edges(graph_data, node_mappings)
        edge_lists['patient_similarity'] = patient_similarity_edges
        
        return edge_lists

    def create_medication_cooccurrence_edges(self, graph_data, node_mappings, min_cooccurrence=3):
        """Create medication-medication edges based on co-prescription patterns"""
        print("Creating medication co-occurrence edges...")
        
        # Count how often each medication pair appears together
        medication_pairs = defaultdict(int)
        
        for _, row in graph_data.iterrows():
            medications = row['MEDICATION_CATEGORY']
            
            # Create pairs from medications for this patient
            for i, med1 in enumerate(medications):
                for j, med2 in enumerate(medications):
                    if i < j and med1 in node_mappings['medication'] and med2 in node_mappings['medication']:
                        # Sort to ensure consistent ordering
                        pair = tuple(sorted([med1, med2]))
                        medication_pairs[pair] += 1
        
        # Create edges for pairs that co-occur frequently enough
        med_cooccurrence_edges = []
        
        for (med1, med2), count in medication_pairs.items():
            if count >= min_cooccurrence:
                med1_id = node_mappings['medication'][med1]
                med2_id = node_mappings['medication'][med2]
                # Add bidirectional edges
                med_cooccurrence_edges.extend([(med1_id, med2_id), (med2_id, med1_id)])
        
        print(f"Created {len(med_cooccurrence_edges)} medication co-occurrence edges from {len(medication_pairs)} pairs")
        return med_cooccurrence_edges

    def create_comorbidity_cooccurrence_edges(self, graph_data, node_mappings, min_cooccurrence=5):
        """Create comorbidity-comorbidity edges based on co-occurrence patterns"""
        print("Creating comorbidity co-occurrence edges...")
        
        # Count how often each comorbidity pair appears together
        comorbidity_pairs = defaultdict(int)
        
        for _, row in graph_data.iterrows():
            comorbidities = row['DETECTED_COMORBIDITIES']
            
            # Create pairs from comorbidities for this patient
            for i, cond1 in enumerate(comorbidities):
                for j, cond2 in enumerate(comorbidities):
                    if i < j and cond1 in node_mappings['comorbidity'] and cond2 in node_mappings['comorbidity']:
                        # Sort to ensure consistent ordering
                        pair = tuple(sorted([cond1, cond2]))
                        comorbidity_pairs[pair] += 1
        
        # Create edges for pairs that co-occur frequently enough
        comorbidity_cooccurrence_edges = []
        
        for (cond1, cond2), count in comorbidity_pairs.items():
            if count >= min_cooccurrence:
                cond1_id = node_mappings['comorbidity'][cond1]
                cond2_id = node_mappings['comorbidity'][cond2]
                # Add bidirectional edges
                comorbidity_cooccurrence_edges.extend([(cond1_id, cond2_id), (cond2_id, cond1_id)])
        
        print(f"Created {len(comorbidity_cooccurrence_edges)} comorbidity co-occurrence edges from {len(comorbidity_pairs)} pairs")
        return comorbidity_cooccurrence_edges

    def create_patient_similarity_edges(self, graph_data, node_mappings, similarity_threshold=0.5):
        """Create patient-patient edges based on shared medications and comorbidities"""
        print("Creating patient similarity edges...")
        
        patient_similarity_edges = []
        patients = list(node_mappings['patient'].keys())
        
        # Create patient profiles for similarity comparison
        patient_profiles = {}
        for _, row in graph_data.iterrows():
            patient_id = row['SUBJECT_ID']
            medications = set(row['MEDICATION_CATEGORY'])
            comorbidities = set(row['DETECTED_COMORBIDITIES'])
            
            patient_profiles[patient_id] = {
                'medications': medications,
                'comorbidities': comorbidities,
                'age_group': row.get('AGE_GROUP'),
                'gender': row.get('GENDER')
            }
        
        # Calculate similarity between patient pairs
        similarity_count = 0
        total_comparisons = 0
        
        for i, patient1 in enumerate(patients):
            for j, patient2 in enumerate(patients):
                if i < j:  # Avoid duplicate pairs and self-loops
                    total_comparisons += 1
                    
                    profile1 = patient_profiles[patient1]
                    profile2 = patient_profiles[patient2]
                    
                    # Calculate Jaccard similarity for medications
                    med_intersection = len(profile1['medications'] & profile2['medications'])
                    med_union = len(profile1['medications'] | profile2['medications'])
                    med_similarity = med_intersection / med_union if med_union > 0 else 0
                    
                    # Calculate Jaccard similarity for comorbidities
                    cond_intersection = len(profile1['comorbidities'] & profile2['comorbidities'])
                    cond_union = len(profile1['comorbidities'] | profile2['comorbidities'])
                    cond_similarity = cond_intersection / cond_union if cond_union > 0 else 0
                    
                    # Bonus for same demographics
                    demo_bonus = 0
                    if profile1['age_group'] == profile2['age_group']:
                        demo_bonus += 0.1
                    if profile1['gender'] == profile2['gender']:
                        demo_bonus += 0.1
                    
                    # Combined similarity score
                    overall_similarity = (med_similarity + cond_similarity + demo_bonus) / 2
                    
                    # Create edge if similarity is above threshold
                    if overall_similarity >= similarity_threshold:
                        patient1_id = node_mappings['patient'][patient1]
                        patient2_id = node_mappings['patient'][patient2]
                        # Add bidirectional edges
                        patient_similarity_edges.extend([(patient1_id, patient2_id), (patient2_id, patient1_id)])
                        similarity_count += 1
        
        print(f"Created {len(patient_similarity_edges)} patient similarity edges from {similarity_count} similar pairs")
        print(f"Patient similarity rate: {similarity_count/total_comparisons:.1%} of {total_comparisons} comparisons")
        return patient_similarity_edges

    def create_readmission_labels(self, days_threshold=30):
        "define what we're going to predict - looking at readmissions in this case"

        asthma_admissions = self.admissions[
            self.admissions['SUBJECT_ID'].isin(self.asthma_patients)
        ].copy()

        # sort by patient and admission time
        asthma_admissions['ADMITTIME'] = pd.to_datetime(asthma_admissions['ADMITTIME'])
        asthma_admissions['DISCHTIME'] = pd.to_datetime(asthma_admissions['DISCHTIME'])
        asthma_admissions = asthma_admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])

        # time between consecutive admissions
        asthma_admissions['NEXT_ADMISSION'] = asthma_admissions.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
        asthma_admissions['DAYS_TO_NEXT'] = (
            asthma_admissions['NEXT_ADMISSION'] - asthma_admissions['DISCHTIME']
        ).dt.days

        # Create readmission label (1 if readmitted within threshold, 0 otherwise)
        asthma_admissions['READMITTED'] = (
            asthma_admissions['DAYS_TO_NEXT'] <= days_threshold
        ).fillna(False).astype(int)

        return asthma_admissions

    def run_full_pipeline(self):
        """Run the complete data preparation pipeline"""
        print("Starting MIMIC-III data preparation pipeline...")
        
        # Load data
        self.load_core_tables()
        
        # Identify asthma patients
        self.identify_asthma_patients()
        
        # Create graph data structure
        graph_data = self.create_graph_data_structure()
        
        print("\nData preparation complete!")
        print(f"Graph contains:")
        print(f"  - {len(graph_data['node_mappings']['patient'])} patients")
        print(f"  - {len(graph_data['node_mappings']['medication'])} medication categories")
        print(f"  - {len(graph_data['node_mappings']['comorbidity'])} comorbidities")
        print(f"  - {len(graph_data['edge_lists']['patient_medication'])} patient-medication edges")
        print(f"  - {len(graph_data['edge_lists']['patient_comorbidity'])} patient-comorbidity edges")
        print("New edge types:")
        print(f"- Medication co-occurrence: {len(graph_data['edge_lists']['medication_cooccurrence'])} edges")
        print(f"- Comorbidity co-occurrence: {len(graph_data['edge_lists']['comorbidity_cooccurrence'])} edges") 
        print(f"- Patient similarity: {len(graph_data['edge_lists']['patient_similarity'])} edges")
        
        return graph_data


if __name__ == "__main__":
    mimic_path = "mimic-iii-clinical-database-demo-1.4" # Update this path with path to mimic-iii
    preparator = MIMICDataPreparator(mimic_path)

    graph_data = preparator.run_full_pipeline()
    
    import pickle
    with open('datasets/mimic_asthma_graph_data_icd_full_v5.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    
    print("Data preparation code ready!")
    