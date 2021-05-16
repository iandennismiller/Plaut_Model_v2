"""
report_generator.py

=== SUMMARY ===
Description     : Automatically generate a PDF report based on configuration settings
Date Created    : July 19, 2020
Last Updated    : July 29, 2020

=== UPDATE NOTES ===
 > September 28, 2020
    - file created
"""

import jinja2
from common.constants import PlotTypes
from weasyprint import HTML
import yaml
import os
import shutil
from datetime import datetime
import pandas as pd


class ReportGenerator:
    TEMP_FOLDER = 'reports/tmp'
    COMPONENTS_FOLDER = 'reports/components'

    IMAGE_WIDTH = {
        'FULL': '100%',
        'HALF': '50%'
    }

    def __init__(self, config):
        """
        Initialize the Report Generator

        Arguments:
            config (str): filepath to configuration file
        """

        with open(config) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.contents = []

        self.templateLoader = jinja2.FileSystemLoader(searchpath="./")
        self.templateEnv = jinja2.Environment(loader=self.templateLoader)

    def add_heading(self, level, text):
        """
        Add a heading to the HTML template

        Arguments:
            level (int): the heading level (only 1 or 2 allowed for now)
            text (str): heading text

        Returns:
            None
        """
        heading_template = self.templateEnv.get_template(f'{ReportGenerator.COMPONENTS_FOLDER}/heading{level}.html')
        heading_output = heading_template.render(text=text)
        self.contents.append(heading_output)

    def add_text(self, text):
        """
        Add plain text to the HTML template

        Arguments:
            text (str): text to be added

        Returns:
            None
        """
        text_template = self.templateEnv.get_template(f'{ReportGenerator.COMPONENTS_FOLDER}/text.html')
        text_output = text_template.render(text=text)
        self.contents.append(text_output)

    def add_plot(self, img_path, width):
        """
        Add a plot to the HTML template

        Arguments:
            img_path (str): filepath to the desired plot image
            width (str): the width of the image (can be relative % or absolute px value)

        Returns:
            None
        """
        shutil.copy(img_path, f'{ReportGenerator.TEMP_FOLDER}/{hash(img_path)}.png')
        plot_template = self.templateEnv.get_template(f'{ReportGenerator.COMPONENTS_FOLDER}/plot.html')
        plot_output = plot_template.render(img_path=f'{hash(img_path)}.png', style=f"'width:{width};'")
        self.contents.append(plot_output)

    def add_content(self, content_type, params=None, sim_directory=None):
        """
        Wrapper function for adding content to the HTML template

        Arguments:
            content_type (str): specifies what type of content to add
            params (dict): additional parameters specific to content type
            sim_directory (str): filepath to root directory of simulation results

        Returns:
            None
        """
        if content_type == 'heading':
            if params['type'] == 'default':
                self.add_heading(level=2, text=sim_directory)

        elif content_type == 'plot':
            if params['type'] == 'loss':
                filepath = os.path.abspath(f"results/{sim_directory}/{PlotTypes.TRAINING_LOSS}.png")
            elif params['type'] == 'plaut_accuracy':
                filepath = os.path.abspath(f"results/{sim_directory}/{PlotTypes.PLAUT_ACCURACY}.png")
            elif params['type'] == 'anchor_accuracy':
                filepath = os.path.abspath(f"results/{sim_directory}/{PlotTypes.ANCHOR_ACCURACY}.png")
            elif params['type'] == 'probe_accuracy':
                filepath = os.path.abspath(f"results/{sim_directory}/{PlotTypes.PROBE_ACCURACY}.png")
            elif params['type'] == 'hidden_similarity':
                filepath = os.path.abspath(f"results/{sim_directory}/{PlotTypes.HIDDEN_SIMILARITY}.png")
            else:
                raise Exception('Plot Type is either not given or unrecognized')

            self.add_plot(img_path=filepath, width=ReportGenerator.IMAGE_WIDTH[params['width'].upper()])

        elif content_type == 'sim_config':
            with open(f'results/{sim_directory}/simulator_config.yaml') as file:
                sim_config = yaml.load(file, Loader=yaml.FullLoader)

            if params['basic']:
                temp_df = pd.Series(data=sim_config['training']).to_frame().T
                temp_df.columns = temp_df.columns.str.title().str.replace('_', ' ')
                self.contents.append(temp_df.to_html(index=False, justify='left'))

            if params['optimizers']:
                temp_df = pd.DataFrame(data=sim_config['optimizers']).T
                temp_df.columns = temp_df.columns.str.title().str.replace('_', ' ')
                temp_df.index.name = 'Starting Epoch'
                # sort columns
                temp_df = temp_df.reset_index(drop=True)
                temp_df = temp_df[['Start Epoch', 'Optimizer', 'Learning Rate', 'Momentum', 'Weight Decay']]
                self.contents.append(temp_df.to_html(index=False, justify='left'))

        elif content_type == 'break':
            self.contents.append('<p style="page-break-before: always">')

    def generate(self):
        """
        Generates the HTML template and PDF report

        Returns:
            None
        """

        # create temporary folder for copying image files
        try:
            os.mkdir(ReportGenerator.TEMP_FOLDER)
        except FileExistsError:
            shutil.rmtree(ReportGenerator.TEMP_FOLDER)
            os.mkdir(ReportGenerator.TEMP_FOLDER)

        # find all simulations to be included in report
        group_directory = self.config['group_directory']
        sim_directory = self.config['sim_directory']
        assert (group_directory or sim_directory) and not(group_directory and sim_directory), \
            "A group directory or a simulation directory must be specified, but not both"

        if group_directory:
            assert os.path.exists(f'results/{group_directory}'), "ERROR: Group directory does not exist"

            sim_list = os.listdir(f'results/{group_directory}')
            sim_label = group_directory.split('-')[0]
            sim_list = [f'{group_directory}/{x}' for x in sim_list if sim_label in x]

            # filters
            if self.config['filters']:
                
                seed_filters = self.config['filters']['seed']
                dilution_filters = self.config['filters']['dilution']
                order_filters = self.config['filters']['order']

                for symbol, filters in zip(['S', 'D', 'O'], [seed_filters, dilution_filters, order_filters]):
                    if type(filters) == int:
                        sim_list = [x for x in sim_list if f'{symbol}{filters}' in x.split('-')[2]]
                    elif type(filters) == list:
                        sim_list = [x for x in sim_list if any([f"{symbol}{f}" in x.split('-')[-2] for f in filters])]

                if len(sim_list) == 0:
                    raise Exception('Simulation filters resulted in no satisfactory simulations')

        else:
            assert os.path.exists(f'results/{sim_directory}'), "ERROR: Simulation directory does not exist"
            sim_list = [sim_directory]

        # REPORT HEADER
        self.add_heading(level=1, text=self.config['title'])
        now = datetime.now()
        self.add_text(text=f'Report generated on {now.strftime("%B %d, %Y")} at {now.strftime("%H:%M:%S")}')

        # REPORT CONTENTS
        for i, sim in enumerate(sim_list):
            for content in self.config['content']:
                content_type = list(content.keys())[0]
                params = content[content_type]

                self.add_content(content_type, params, sim)

            if i+1 < len(sim_list):
                self.add_content(content_type='break')

        # CREATE HTML TEMPLATE
        base_template = self.templateEnv.get_template(f'{ReportGenerator.COMPONENTS_FOLDER}/base.html')
        base_output = base_template.render(content='\n\n'.join(self.contents))
        with open(f'{ReportGenerator.TEMP_FOLDER}/template.html', 'w') as html_file:
            html_file.write(base_output)

        # CREATE PDF FROM TEMPLATE
        if group_directory:
            HTML(f'{ReportGenerator.TEMP_FOLDER}/template.html').write_pdf(
                f'results/{group_directory}/{self.config["title"]}.pdf', stylesheets=['reports/style.css'])
        else:
            HTML(f'{ReportGenerator.TEMP_FOLDER}/template.html').write_pdf(
                f'results/{sim_directory}/{self.config["title"]}.pdf', stylesheets=['reports/style.css'])

        # delete temp folder for storing image files
        shutil.rmtree(f'{ReportGenerator.TEMP_FOLDER}')
        return None

