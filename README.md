# Nutrient_depletion_code
temporary code for nutrient depletion

## Installation
To run the code, one extra package needs to be installed. In your terminal, type in
```bash
pip install opencv-python
```

## How to use
1. Execute `auxiliary_funcs.py`, `simulation_funcs.py`;
2. Create an initialization file, or modify `initialization_for_fig4.py` so that you have your corresponding parameters. Execute the file.
3. Go to `main_script.py`. Import the initialization file you just created under `step 1`;
4. Create two directories under the current folder, named `data_for_video`, and `storage_results`. They can be changed to reflect the parameters you use in this particular simulation, but you need to change the corresponding code.
5. Comment out everything under `step 3`. Execute everything under `step 2`. This will generate a video for you. Also, corresponding data matrices are stored under the `storage_results` folder.
6. Comment out everything under `step 2`. Execute everything under `step 3`. This reloads all you data. You should be able to plot whatever you want, by adding code under `step 3`.
7. Finally, if you want to re-run the simulation, say if you want to change parameters, please make sure files under `data_for_video` and `storage_results` are manually deleted.
