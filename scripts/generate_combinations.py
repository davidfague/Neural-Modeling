import itertools
import json

def generate_combinations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"Data from JSON: {data}")  # Debug statement
    
    variable_sets = data.get('variable_sets', [])
    for variable_set in variable_sets:
        keys = list(variable_set.keys())
        values = list(variable_set.values())

        # Separate 'complex_cell_biophys_hoc_name' values
        hoc_name_values = variable_set.get('complex_cell_biophys_hoc_name', [])
        other_values = [value for key, value in zip(keys, values) if key != 'complex_cell_biophys_hoc_name']

        # Create combinations for other values
        for other_combination in itertools.product(*other_values):
            for hoc_name in hoc_name_values:
                current_combination = dict(zip(keys, other_combination))
                current_combination['complex_cell_biophys_hoc_name'] = hoc_name
                yield current_combination

if __name__ == '__main__':
    json_file_path = 'constants_to_update.json'  # Specify your JSON file path here
    for combination in generate_combinations(json_file_path):
        print(json.dumps(combination))
        #print(f"Generated combination: {combination}")  # Debug statement

