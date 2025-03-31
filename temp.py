import pandas as pd

input_file = 'Data/L_dynamique1x100dis2_0013.csv'
output_file = 'Data/L_dynamique1x100dis2_0013_mod.csv'

selected_columns = ['Gamma', 'Theta', 'Time',
    'cable_cor_on_gamma_plane_15 X', 'cable_cor_on_gamma_plane_15 Y', 'cable_cor_on_gamma_plane_15 Z',
    'robot_cable_attach_point X', 'robot_cable_attach_point Y', 'robot_cable_attach_point Z',
    'rob_speed X', 'rob_speed Y', 'rob_speed Z',
    'rod_end X', 'rod_end Y', 'rod_end Z'
]

# Read the file with column name stripping
df = pd.read_csv(input_file, usecols=lambda x: x.strip() in selected_columns)

# Save filtered data
df.to_csv(output_file, index=False)
print('Done')
