# Catenary 3D Visualization Project

This project provides a visualization of the catenary model in 3D. It includes functions to compute the standard catenary shape and apply rotations for augmented catenary models. The visualization is created using Matplotlib.

## Project Structure

```
catenary-3d-visualization
├── src
│   ├── catenary_model.py  # Contains functions for catenary calculations
│   └── visualize.py       # Responsible for 3D visualization of the catenary model
├── requirements.txt       # Lists the project dependencies
└── README.md              # Documentation for the project
```

## Installation

To set up the project, ensure you have Python installed on your machine. Then, install the required dependencies by running:

```
pip install -r requirements.txt
```

## Usage

To visualize the catenary model in 3D, run the `visualize.py` script located in the `src` directory:

```
python src/visualize.py
```

This will generate a 3D plot of the catenary model based on the parameters defined in the script.

## Dependencies

The project requires the following Python packages:

- numpy
- matplotlib
- scipy

Make sure these packages are installed to ensure the project runs smoothly.

## License

This project is licensed under the MIT License.