import os
from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.utils import Utils
import json
import subprocess

class AllenCell():
    def __init__(self, h):
        self.soma = list(h.soma)
        self.all = [sec for sec in h.allsec()]
        self.apic = list(h.apic)
        self.dend = list(h.dend)
        self.axon = list(h.axon)

def update_missing_passive_values(utils, user_specs_dict):
  # update missing properties to user_specs_dict if they're not already in the allen specifications
  if "e_pas" not in utils.description.data["passive"][0].keys():
    utils.description.data["passive"][0]["e_pas"] = user_specs_dict["e_pas"]
    print(f"e_pas not in Allen specs, updating to {user_specs_dict['e_pas']}")
  
  if "cm" not in utils.description.data["passive"][0].keys():
    utils.description.data["passive"][0]["cm"] = user_specs_dict["cm"]
    print(f"cm not in Allen specs, updating to {user_specs_dict['cm']}")

  if "ra" not in utils.description.data["passive"][0].keys():
    utils.description.data["passive"][0]["ra"] = user_specs_dict["ra"]
    print(f"ra not in Allen specs, updating to {user_specs_dict['ra']}")

  return utils

def load_dictionary_from_json(file_path):
  """Loads a dictionary from a JSON file.

  Args:
    file_path: The path to the JSON file.

  Returns:
    A dictionary loaded from the JSON file.

  Raises:
    ValueError: If the JSON file is invalid or cannot be parsed.
    FileNotFoundError: If the specified file does not exist.
  """
  try:
    with open(file_path, 'r') as file:
      try:
        data = json.load(file)
        if not isinstance(data, type({})):
          raise ValueError("JSON file does not contain a dictionary.")
        print(f"loaded user_specs from {file_path}")
        return data
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
  except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")


def load_skeleton_cell_from_allen(allen_cell_dir):

    # # compile the downloaded modfiles
    # if not os.path.exists(os.path.join(allen_cell_dir,"x86_64")):
    #     print("compiling modfiles")
    #     subprocess.run("nrnivmodl modfiles", shell=True, check=True)
    # else:
    #     print("modfiles already compiled. skipping")

    # Create the h object
    print(f"loading manifest from {allen_cell_dir}")
    
    curr_dir = os.getcwd()
    os.chdir(allen_cell_dir)

    description = Config().load('manifest.json')
    utils = Utils(description)
    h = utils.h

    # convert 'values' from string to float
    for dict in utils.description.data['genome']:
        for key,item in dict.items():
            if key == 'value':
                dict[key] = float(item)

    # load user specifications
    # print(f"the directory where we find /user_specifications: {os.path.join(allen_cell_dir, "../user_specifications.json")}")
    # user_specs_dict = load_dictionary_from_json(os.path.join(allen_cell_dir, "../user_specifications.json"))
    user_specs_dict = load_dictionary_from_json("../user_specifications.json")

    if user_specs_dict:
        utils = update_missing_passive_values(utils, user_specs_dict)

    # read morphology
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY')
    utils.generate_morphology(morphology_path.encode('ascii', 'ignore'))

    # build the cell. Its parts will be assigned to the h object
    utils.load_cell_parameters()
    skeleton_cell = AllenCell(utils.h)

    os.chdir(curr_dir)
    return skeleton_cell