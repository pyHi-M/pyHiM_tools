#!/bin/bash

# Define the reference file
reference_file="scan_001_RT1_000_ROI_converted_decon_ch00.tif"

# Prompt the user to input numbers corresponding to the filenames
echo "Enter the numbers corresponding to the filenames to be analyzed (e.g., 16 18 20):"
read -r input_numbers

# Convert the input into an array
IFS=' ' read -r -a numbers <<< "$input_numbers"

# Iterate through each number provided by the user
for number in "${numbers[@]}"; do
  # Construct the file pattern using the provided number
  file_pattern="scan_001_RT${number}_000_ROI_converted_decon_ch00.tif"
  echo
  echo 'Will process: '$file_pattern

  # Find files matching the pattern
  files=$(find . -name "$file_pattern")
  echo 'Found:'$files

  # Iterate through each file found
  for file in $files; do
    # Extract the base filename without the directory path
    base_filename=$(basename "$file")
    echo 'Will align:'$base_filename

    # Construct the displacement field filename
    displacement_field="${base_filename%.tif}_DF.h5"

    # Construct the output filename
    output_file="${base_filename%.tif}_aligned.tif"

    # Construct the log filename
    log_file="${base_filename%.tif}_DF.log"

    # Run the register_3D_deeds_blocks.py command with the constructed arguments and log the output
    echo 'Running register_3D_deeds_blocks for '$file
    register_3D_deeds_blocks.py --reference "$reference_file" --moving "$file" --displacement_field "$displacement_field" --output "$output_file" > "$log_file"

  done
done
