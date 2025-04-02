import os
import re
import glob
import imageio

def create_gif_for_plot_type(results_folder, plot_type, output_filename, fps=2):
    """
    Creates an animated GIF for a given plot type by reading all PNG files with the specified suffix.
    
    Parameters:
        results_folder (str): Path to the results folder.
        plot_type (str): Suffix for the plot type (e.g., "mean_a", "mean_psi11", "variance_sum").
        output_filename (str): Filename for the output GIF.
        fps (int): Frames per second for the GIF.
    """
    # Search for PNG files ending with the specified plot type
    pattern = os.path.join(results_folder, f"*_{plot_type}.png")
    file_list = glob.glob(pattern)
    
    # Regex to extract the Gamma value from filenames, expected pattern "Gamma_<gamma>"
    gamma_pattern = re.compile(r"Gamma_([\d\.]+)")
    
    def extract_gamma(filename):
        basename = os.path.basename(filename)
        match = gamma_pattern.search(basename)
        if match:
            return float(match.group(1))
        else:
            return float('inf')
    
    # Sort files in ascending order by Gamma
    sorted_files = sorted(file_list, key=extract_gamma)
    
    if not sorted_files:
        print(f"No files found for plot type {plot_type} in {results_folder}")
        return
    
    images = []
    for file in sorted_files:
        try:
            img = imageio.imread(file)
            images.append(img)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not images:
        print(f"No valid images found for plot type {plot_type}.")
        return

    output_path = os.path.join(results_folder, output_filename)
    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved for {plot_type}: {output_path}")

def main():
    # Pfad zum "results"-Ordner; passe diesen Pfad ggf. an
    results_folder = "results"
    
    # Erstelle ein GIF für jeden Plot-Typ
    create_gif_for_plot_type(results_folder, "mean_a", "mean_a_gamma.gif", fps=2)
    create_gif_for_plot_type(results_folder, "mean_psi11", "mean_psi11_gamma.gif", fps=2)
    create_gif_for_plot_type(results_folder, "variance_sum", "variance_sum_gamma.gif", fps=2)

if __name__ == '__main__':
    main()
