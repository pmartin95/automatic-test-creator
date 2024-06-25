import json
from typing import List, Dict, Any
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
import argparse
from pylatex import Document, Section,  Command, Figure, Package
from pylatex.utils import  NoEscape
from tqdm import tqdm
import yaml
from pathlib import Path

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

IMAGE_DIR = config['directories']['images']
PROBLEM_DIR = config['directories']['problems']
LOG_FILE = config['files']['log']
DEFAULT_NUM_PROBLEMS = config['defaults']['num_problems']

def setup_logging(verbose: bool):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    handler = RotatingFileHandler(LOG_FILE, maxBytes=1e6, backupCount=5)
    logging.getLogger().addHandler(handler)


def import_json(filename: str) -> Dict[str, Any]:
    """
    Import data from a JSON file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: The parsed JSON data.

    Raises:
        FileNotFoundError: If the file is not found.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File '{filename}' not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"File '{filename}' contains invalid JSON.")
        raise

def select_problems(problems_dataframe: pd.DataFrame, tags: List[str] = [], num_problems: int = 1) -> pd.DataFrame:
    """
    Select problems from a DataFrame based on tags and desired number of problems.

    Args:
        problems_dataframe (pd.DataFrame): The input DataFrame containing problems.
        tags (List[str], optional): List of tags to filter by. Defaults to [].
        num_problems (int, optional): Number of problems to select. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame containing the selected problems.
    """
    df = problems_dataframe.copy()
    
    if 'tags' not in df.columns:
        logging.warning("'tags' column not found in the DataFrame. Returning random selection.")
        n = min(num_problems, len(df))
        return df.sample(n=n)
    
    if not tags:
        n = min(num_problems, len(df))
        return df.sample(n=n)
    
    df['tags'] = df['tags'].apply(lambda x: set(x) if isinstance(x, list) else set())
    filter_set = set(tags)
    
    filtered_df = df[df['tags'].apply(lambda x: filter_set.issubset(x))]
    
    if filtered_df.empty:
        logging.warning("No problems match the given tags.")
        return pd.DataFrame()
    
    n = min(num_problems, len(filtered_df))
    return filtered_df.sample(n=n)

def collect_packages(problems_df: pd.DataFrame) -> List[str]:
    """
    Collect unique packages from all selected problems.

    Args:
        problems_df (pd.DataFrame): DataFrame containing selected problems.

    Returns:
        List[str]: List of unique package names.
    """
    all_packages = set()
    for _, row in problems_df.iterrows():
        if 'packages' in row and isinstance(row['packages'], list):
            all_packages.update(row['packages'])
    return list(all_packages)

def fill_document(doc: Document, problems_df: pd.DataFrame, images_dir: str):
    for _, row in tqdm(problems_df.iterrows(), total=len(problems_df), desc="Filling solution document"):
        if row["problem name"]:
            with doc.create(Section(NoEscape(row["problem name"]))):
                doc.append(NoEscape(row["problem statement"]))
                
                logging.info(f"Processing problem: {row['problem name']}")
                logging.info(f"Images: {row.get('images', [])}")
                
                if "images" in row and isinstance(row["images"], list):
                    for i, image_filename in enumerate(row["images"], 1):
                        image_path = os.path.join('./images', image_filename)  # Use relative path
                        logging.info(f"Attempting to add image: {image_path}")
                        try:
                            with doc.create(Figure(position='htbp')) as fig:
                                fig.add_image(image_path, width=NoEscape(r'0.8\textwidth'))
                                fig.add_caption(f"Image {i} for {row['problem name']}")
                            logging.info(f"Successfully added image: {image_path}")
                        except Exception as e:
                            logging.warning(f"Failed to add image {image_path}: {str(e)}")

def fill_solution_document(doc: Document, problems_df: pd.DataFrame, images_dir: str):
    for _, row in tqdm(problems_df.iterrows(), total=len(problems_df), desc="Filling solution document"):
        if row["problem name"] and "solution" in row:
            with doc.create(Section(f"Solution: {row['problem name']}")):
                doc.append(NoEscape(row["solution"]))
                
                # Check if the solution has associated images
                if "solution_images" in row and isinstance(row["solution_images"], list):
                    for i, image_filename in enumerate(row["solution_images"], 1):
                        image_path = os.path.join(images_dir, image_filename)
                        if os.path.exists(image_path):
                            with doc.create(Figure(position='htbp')) as fig:
                                fig.add_image(image_filename, width=NoEscape(r'0.8\textwidth'))
                                fig.add_caption(f"Solution Image {i} for {row['problem name']}")
                        else:
                            logging.warning(f"Solution image file not found: {image_path}")

def generate_pdf(doc: Document, filename: str):
    """Generate a PDF from a PyLaTeX document."""
    try:
        doc.generate_pdf(filename, clean_tex=False, compiler='pdflatex', compiler_args=['-interaction=nonstopmode', '-halt-on-error'])
        logging.info(f"Successfully generated {filename}.pdf")
    except Exception as e:
        logging.error(f"Failed to generate {filename}.pdf: {str(e)}")
        raise


def main(proj_dir: str, tags: List[str], num_problems: int) -> None:
    logging.info(f"Starting main function with proj_dir={proj_dir}, tags={tags}, num_problems={num_problems}")
    logging.info(f"Current working directory: {os.getcwd()}")
    
    images_dir = Path(proj_dir) / config['directories']['images']
    problems_dir = Path(proj_dir) / config['directories']['problems']
    
    logging.info(f"Images directory: {images_dir}")
    logging.info(f"Problems directory: {problems_dir}")
    
    try:
        problems_list = [import_json(os.path.join(root, file)) 
                        for root, _, files in os.walk(problems_dir) 
                        for file in files if file.endswith('.json')]
        
        logging.debug(f"Loaded {len(problems_list)} problem(s) from JSON files")
        
        problems_df = pd.DataFrame(problems_list)
        logging.info(f"DataFrame columns: {problems_df.columns}")
        logging.info(f"DataFrame shape: {problems_df.shape}")
        
        select_problems_df = select_problems(problems_df, tags, num_problems)
        logging.info(f"Selected {len(select_problems_df)} problem(s)")
        
        problem_packages = collect_packages(select_problems_df)
        logging.info(f"Collected packages: {problem_packages}")

        # Create problem document
        problem_doc = Document("problem_set")
        for package in problem_packages:
            problem_doc.packages.append(Package(package))
        problem_doc.packages.append(Package('graphicx'))
        problem_doc.packages.append(Package('amsmath'))
        problem_doc.preamble.append(Command('graphicspath', '{{./images/}}'))
        problem_doc.preamble.append(Command('title', 'Problem Set'))
        problem_doc.preamble.append(Command('author', 'Anonymous author'))
        problem_doc.preamble.append(Command('date', NoEscape(r'\today')))
        problem_doc.append(NoEscape(r'\maketitle'))
        
        logging.info("Filling problem document")
        fill_document(problem_doc, select_problems_df, images_dir)
        
        # logging.info("Generating problem set PDF")
        # problem_doc.generate_pdf('problem_set', clean_tex=False, compiler='pdflatex', compiler_args=['-interaction=nonstopmode', '-halt-on-error'])
        
        # Create solution document
        solution_doc = Document("solutions")
        solution_doc.packages.append(Package('graphicx'))
        solution_doc.packages.append(Package('amsmath'))
        for package in problem_packages:
            solution_doc.packages.append(Package(package))
        solution_doc.preamble.append(Command('graphicspath', '{./images/}'))
        solution_doc.preamble.append(Command('title', 'Solutions'))
        solution_doc.preamble.append(Command('author', 'Anonymous author'))
        solution_doc.preamble.append(Command('date', NoEscape(r'\today')))
        solution_doc.append(NoEscape(r'\maketitle'))
        
        logging.info("Filling solution document")
        fill_solution_document(solution_doc, select_problems_df, images_dir)
        
        # logging.info("Generating solutions PDF")
        # solution_doc.generate_pdf('solutions', clean_tex=False)
        generate_pdf(problem_doc, 'problem_set')
        generate_pdf(solution_doc, 'solutions')        
        logging.info("PDF generation complete")
        if os.path.exists('problem_set.pdf') and os.path.exists('solutions.pdf'):
            logging.info("PDF files successfully created.")
            logging.info(f"Problem set: {os.path.abspath('problem_set.pdf')}")
            logging.info(f"Solutions: {os.path.abspath('solutions.pdf')}")
        else:
            logging.error("PDF generation may have failed. Please check the output files.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and select problems based on tags.")
    parser.add_argument("proj_dir", help="Project directory path")
    parser.add_argument("--tags", nargs='*', default=[], help="Tags to filter problems")
    parser.add_argument("--num_problems", type=int, default=1, help="Number of problems to select")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    
    args = parser.parse_args()
    
    # Set up logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    main(args.proj_dir, args.tags, args.num_problems)


# Running this problem ex:
# python3 scripts/main.py . --num_problems 2 --tags math -v