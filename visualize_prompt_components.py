import argparse

from os import makedirs
from os.path import join

import numpy as np
import matplotlib.pyplot as plt


def main(opt) :
    # PromptBind Model Configuration
    best_epoch_dict = {8:12, 16:12, 32:22, 48:40}
    
    # Create Directory for Saving Prompt Component Visualization Results
    save_path = f"prompt_comp_visualization/prompt_{opt.prompt_num}"
    makedirs(save_path, exist_ok=True)
    
    # Load Prompt Components
    prompt_comp_root = f"results/prompt_{opt.prompt_num}/prompt_components/epoch-{best_epoch_dict[opt.prompt_num]}"
    
    pocket_prompt_node_components = np.load(join(prompt_comp_root, "pocket_prompt_node_components.npy"))
    pocket_prompt_coord_components = np.load(join(prompt_comp_root, "pocket_prompt_coord_components.npy"))
    complex_prompt_node_components = np.load(join(prompt_comp_root, "complex_prompt_node_components.npy"))
    complex_prompt_coord_components = np.load(join(prompt_comp_root, "complex_prompt_coord_components.npy"))
    
    # Start Prompt Component Visualization
    print(f"Prompt Component Visualization [PromptBind-{opt.prompt_num}]")

    # Save Pocket Node Prompt Component Visualization Result
    print(f"[Pocket Node Prompt Comp.] < (Mean) : {pocket_prompt_node_components.mean()} || (Std) : {pocket_prompt_node_components.std()} >")
    plt.imshow(pocket_prompt_node_components, interpolation=None)
    plt.colorbar()
    plt.savefig(join(save_path, "prompt_node_components(pocket).png"), bbox_inches="tight")
    plt.clf()

    # Save Pocket Coord. Prompt Component Visualization Result
    print(f"[Pocket Coord. Prompt Comp.] < (Mean) : {pocket_prompt_coord_components.mean()} || (Std) : {pocket_prompt_coord_components.std()} >")
    plt.imshow(pocket_prompt_coord_components, interpolation=None)
    plt.colorbar()
    plt.savefig(join(save_path, "prompt_coord_components(pocket).png"), bbox_inches="tight")
    plt.clf()

    # Save Complex Node Prompt Component Visualization Result
    print(f"[Complex Node Prompt Comp.] < (Mean) : {complex_prompt_node_components.mean()} || (Std) : {complex_prompt_node_components.std()} >")
    plt.imshow(complex_prompt_node_components, interpolation=None)
    plt.colorbar()
    plt.savefig(join(save_path, "prompt_node_components(complex).png"), bbox_inches="tight")
    plt.clf()

    # Save Complex Coord. Prompt Component Visualization Result
    print(f"[Complex Coord. Prompt Comp.] < (Mean) : {complex_prompt_coord_components.mean()} || (Std) : {complex_prompt_coord_components.std()} >")
    plt.imshow(complex_prompt_coord_components, interpolation=None)
    plt.colorbar()
    plt.savefig(join(save_path, "prompt_coord_components(complex).png"), bbox_inches="tight")
    plt.clf()


if __name__ == "__main__" :
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-num", type=int, choices=[8,16,32,48])
    opt = parser.parse_args()
    
    # Save Prompt Component Visualization Result
    main(opt)