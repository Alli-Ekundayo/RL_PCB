# Graph Report - .  (2026-04-15)

## Corpus Check
- 160 files · ~152,934 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 854 nodes · 1394 edges · 88 communities detected
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]

## God Nodes (most connected - your core abstractions)
1. `environment` - 26 edges
2. `tracker` - 25 edges
3. `main()` - 24 edges
4. `ReplayMemory` - 16 edges
5. `RL_PCB Project` - 15 edges
6. `h_annealer()` - 13 edges
7. `graph()` - 12 edges
8. `getComponent()` - 12 edges
9. `parameters` - 12 edges
10. `save_report_config()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `:param model: Select model to use for evaluation. There are two options` --uses--> `environment`  [INFERRED]
  src/training/callbacks.py → src/training/core/environment/environment.py
- `Logs a 2D tensor of video frames to tensorboard         :param vids: 2D video te` --uses--> `environment`  [INFERRED]
  src/training/callbacks.py → src/training/core/environment/environment.py
- `Log experiment settings to tensorboard         :param settings: DESCRIPTION, def` --uses--> `environment`  [INFERRED]
  src/training/callbacks.py → src/training/core/environment/environment.py
- `tracker` --uses--> `environment`  [INFERRED]
  src/training/core/environment/tracker.py → src/training/core/environment/environment.py
- `tracker` --uses--> `Iterates through all agents and prints their information          Returns`  [INFERRED]
  src/training/core/environment/tracker.py → src/training/core/environment/environment.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (83): add_edge_from_string_long(), add_edge_from_string_short(), add_node_from_string_long(), add_node_from_string_short(), build_info(), build_info_as_string(), calc_full_hpwl(), calc_hpwl() (+75 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (22): log_and_eval_callback, :param model: Select model to use for evaluation. There are two options, Logs a 2D tensor of video frames to tensorboard         :param vids: 2D video te, Log experiment settings to tensorboard         :param settings: DESCRIPTION, def, dataAugmenter, A class that performs data augmentation on a graph.      This class provides met, environment, Iterates through all agents and prints their information          Returns (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (47): annealer(), build_info(), build_info_as_string(), cell_overlap(), cell_overlap_partial(), check_entrapment(), check_move(), cost() (+39 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (29): areaRectangle(), buildKicadPcb(), find_quadrilaterals(), generate_board(), generate_edges(), generate_nodes(), getBoardBoundaryByEdgeCuts(), getBoardBoundaryByPinLocation() (+21 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (11): GaussianPolicy, QNetwork, SAC, Adds the latest samples from another buffer to this buffer.          Args:, Shuffles the transitions in the replay memory., Samples a batch of transitions from the replay memory.          Args:, Samples a batch of transitions from the latest elements in the\               re, A replay memory buffer used in reinforcement learning algorithms to store\ (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (19): checkIfDirectory(), compare_optimals(), find_file_with_ext(), get_id_by_kicad_pcb_file(), get_id_from_filename(), get_keywords_filenames(), get_kicad_pcb_filenames(), get_padShape() (+11 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (2): agent, tracker

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (21): calculate_resultant_vector(), compute_pad_referenced_distance_vectors_v2(), compute_sum_of_euclidean_distances(), compute_sum_of_euclidean_distances_between_pads(), compute_vector_to_group_midpoint(), cosine_distance_for_two_terminal_component(), deg2rad(), distance_between_two_points() (+13 more)

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (18): append_pcb_file_from_graph_and_board(), append_pcb_file_from_individual_files(), append_pcb_file_from_individual_files_and_optimals(), append_pcb_file_from_pcb(), build_info(), build_info_as_string(), check_for_file_existance(), dependency_info() (+10 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (15): draw_board_from_board_and_graph(), draw_board_from_board_and_graph_multi_agent(), draw_board_from_board_and_graph_with_debug(), draw_los(), draw_node_name(), draw_ratsnest(), draw_ratsnest_with_board(), get_los_and_ol_multi_agent() (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (0): 

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (4): object, Actor, Critic, TD3

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (12): build_info(), get_build_time(), get_cpp_standard(), get_library_version(), kicadParser(), parseKicadPcb(), readNodeName(), readQuotedString() (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (5): This module provides a data augmentation functionality for graphs.  The `dataAug, graph(), pcb, cmdline_args(), configure_seed()

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (17): MIT License, A* Based Routing Algorithm, Cellular Automata Inspiration, CUDA 11.7, DATE 24 Paper, Euclidean Wirelength (EW), Half-Perimeter Wirelength (HPWL), KiCad PCB File Parsing (+9 more)

### Community 15 - "Community 15"
Cohesion: 0.27
Nodes (9): create_from_string_long(), create_from_string_short(), format_string_long(), get_area(), get_inst_bb_centre_size(), get_inst_bb_coords(), node(), print() (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.28
Nodes (10): create_from_string_long(), create_from_string_short(), edge(), format_string_long(), get_pos(), get_size(), print(), print_to_console() (+2 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (7): layerChange(), setParameterPl(), setPos(), setRotation(), str2orient(), updateCoordinates(), wrap_orientation()

### Community 18 - "Community 18"
Cohesion: 0.2
Nodes (15): gen_default_hyperparameters(), gen_default_sb3_hyperparameters(), hyperparmeters_off_policy(), hyperparmeters_on_policy(), load_hyperparameters_from_file(), Sets specific parameters as user attributes to faciliate report generation     b, Method to test whether the given hyperparameters correspond to an on_policy, Method to test whether the given hyperparameters correspond to an     off_policy (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (8): pinShapeToOctagon(), printPolygon(), relativeStartEndPointsForSegment(), rotateShapeCoordsByAngles(), segmentToOctagon(), segmentToRelativeOctagon(), shape_to_coords(), testShapeToCoords()

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (4): PageNumCanvas, http://code.activestate.com/recipes/546511-page-x-of-y-with-reportlab/     http:, On a page break, add information to the list, Add the page number to each page (page x of y)

### Community 21 - "Community 21"
Cohesion: 0.26
Nodes (2): load_report_config(), save_report_config()

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (7): board(), get_board_size(), get_fields(), print(), process_board_file(), process_line(), write_to_file()

### Community 23 - "Community 23"
Cohesion: 0.23
Nodes (10): create_hierarchy(), get_leaf_module_from_id(), get_level_module_from_id(), init_hierarchy(), propagate_geometries(), propagate_netlist(), set_module_geometries(), set_netlist_hierarchy() (+2 more)

### Community 24 - "Community 24"
Cohesion: 0.17
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (5): printExterior(), printParameter(), setParameterPl(), setPos(), updateCoordinates()

### Community 26 - "Community 26"
Cohesion: 0.24
Nodes (6): cmdline_args(), main(), PageNumCanvas, http://code.activestate.com/recipes/546511-page-x-of-y-with-reportlab/     http:, On a page break, add information to the list, Add the page number to each page (page x of y)

### Community 27 - "Community 27"
Cohesion: 0.2
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 0.33
Nodes (3): create_from_string(), format_string(), optimal()

### Community 29 - "Community 29"
Cohesion: 0.56
Nodes (6): kicad_rotate(), mirror_y_then_rotate(), rotate(), round_down(), round_nearest(), round_up()

### Community 30 - "Community 30"
Cohesion: 0.22
Nodes (0): 

### Community 31 - "Community 31"
Cohesion: 0.57
Nodes (3): init_arguments(), parse_args(), parse_opt()

### Community 32 - "Community 32"
Cohesion: 0.32
Nodes (7): The module for setting up and initializing reinforcement learning models.  This, Setup function for the TD3 model.      Args:         train_env: The training env, Setup function for the SAC model.      Args:         train_env: The training env, Setup function to create a model based on the specified model type.      Args:, sac_model_setup(), setup_model(), td3_model_setup()

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (2): getPadstack(), hasBottomCrtyd()

### Community 34 - "Community 34"
Cohesion: 0.29
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 0.53
Nodes (4): get_2d(), get_rect(), get_value(), ss_reset()

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 0.33
Nodes (5): kicad_rotate(), kicad_rotate_around_point(), This module provides utility functions for performing rotation operations using, Rotate a point (x, y) by an angle a (in degrees) using the Kicad rotation     fo, Rotate a point (x, y) around a center point (cx, cy) by an angle     a (in degre

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 0.6
Nodes (3): lib_info_in_paragraphs(), machine_info_in_paragraphs(), This module provides functions for collecting and displaying machine and library

### Community 40 - "Community 40"
Cohesion: 0.5
Nodes (4): command_line_args(), main(), Module for parsing command line arguments, Parses command-line arguments for a Python script that generates a PNG image

### Community 41 - "Community 41"
Cohesion: 0.5
Nodes (3): Module, Node, pPin

### Community 42 - "Community 42"
Cohesion: 0.5
Nodes (3): Unit tests for pcb_vector_utils module, Tests rare condition of having both inputs 0.0 which if unchecked, will \, test_calculate_resultant_vector()

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 0.67
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 0.67
Nodes (1): GridBasedPlacer

### Community 46 - "Community 46"
Cohesion: 0.67
Nodes (2): Hierarchy, Level

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (2): linreg(), sqr()

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (1): This module provides a utility function to retrieve the number of PCBs (Printed

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): HPlacerUtils

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): PlaceDB

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): pPin

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Logger

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): PlaceObj

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (0): 

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Module

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (0): 

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): PlaceObj

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): pPin

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (0): 

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (0): 

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (0): 

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (0): 

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (0): 

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (0): 

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (0): 

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (0): 

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (0): 

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (0): 

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (0): 

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (0): 

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (0): 

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (0): 

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (0): 

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (0): 

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (0): 

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (0): 

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (0): 

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (0): 

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (0): 

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (0): 

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (0): 

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **63 isolated node(s):** `Node`, `pPin`, `Module`, `HPlacerUtils`, `PlaceDB` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 50`** (2 nodes): `util.h`, `utilParser()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (2 nodes): `via.h`, `ViaType()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (2 nodes): `object.h`, `ObjectType()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (2 nodes): `HPlacerUtils.hpp`, `HPlacerUtils`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (2 nodes): `PlaceDB.hpp`, `PlaceDB`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (2 nodes): `Constraint.hpp`, `pPin`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (2 nodes): `Logger.hpp`, `Logger`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (2 nodes): `PlaceObj.hpp`, `PlaceObj`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (2 nodes): `plotter.h`, `Plotter()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (2 nodes): `Module.hpp`, `Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (2 nodes): `globalParam.h`, `GlobalParam()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (2 nodes): `Congestion.hpp`, `PlaceObj`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (2 nodes): `Pin.hpp`, `pPin`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `parse_graph_from_pcb_file.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `utils.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `argparse.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `shape.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `pcbBoost.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `rule.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `layer.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `tree.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `argparse.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `utils.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `argparse.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `linreg.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `argparse.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `utils.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `argparse.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `graph()` connect `Community 13` to `Community 0`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `dataAugmenter` connect `Community 1` to `Community 13`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Why does `environment` connect `Community 1` to `Community 6`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `environment` (e.g. with `log_and_eval_callback` and `:param model: Select model to use for evaluation. There are two options`) actually correct?**
  _`environment` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `tracker` (e.g. with `agent` and `environment`) actually correct?**
  _`tracker` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ReplayMemory` (e.g. with `QNetwork` and `GaussianPolicy`) actually correct?**
  _`ReplayMemory` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RL_PCB Project` (e.g. with `Half-Perimeter Wirelength (HPWL)` and `Euclidean Wirelength (EW)`) actually correct?**
  _`RL_PCB Project` has 2 INFERRED edges - model-reasoned connections that need verification._