from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.starcraft2.maps import smac_maps

extra_params = {
    "1o_10b_vs_1r": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_10b_vs_1r_v2": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "2o_20b_vs_2r": {
        "n_agents": 22,
        "n_enemies": 2,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_2r_vs_4r": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },

    "5z_vs_1ul": {
        "n_agents": 5,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "bane_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },

    "bane_vs_hM_origin": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },
    
    "bane_vs_hM_hard": {
        "n_agents": 4,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },
    
    "bane_vs_hM2": {
        "n_agents": 6,
        "n_enemies": 4,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },    

    
    # "corridor": {
    #     "n_agents": 6,
    #     "n_enemies": 24,
    #     "limit": 400,
    #     "a_race": "P",
    #     "b_race": "Z",
    #     "unit_type_bits": 0,
    #     "map_type": "zealots",
    # },

}

smac_maps.map_param_registry.update(extra_params)

def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]

for name in smac_maps.map_param_registry.keys() and extra_params.keys():
    globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))
