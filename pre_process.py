def extract_valid_ldr_lines(input_file, output_file, unique_bricks_file):
    valid_lines = []
    
    # Load the unique bricks into a set
    with open(unique_bricks_file, "r") as f:
        unique_bricks = set(line.strip() for line in f.readlines())
    
    with open(input_file, "r") as file:
        for line in file:
            line = line.strip()
            
            # Check if the line starts with "1", ends with ".dat", and has 15 entries
            if line.startswith("1") and line.endswith(".dat") and len(line.split()) == 15:
                # Extract the brick name
                brick_name = line.split()[-1]
                
                # Check if the brick is in the unique bricks list
                if brick_name in unique_bricks:
                    valid_lines.append(line)
    
    # Write the valid lines to the output file
    with open(output_file, "w") as file:
        file.write("\n".join(valid_lines))
    
    print(f"Extracted {len(valid_lines)} valid lines to {output_file}")

# Replace these with the actual file paths
input_file = 'test_input.mpd'
output_file = 'test_output.ldr'
unique_bricks_file = 'unique_bricks.txt'

extract_valid_ldr_lines(input_file, output_file, unique_bricks_file)
